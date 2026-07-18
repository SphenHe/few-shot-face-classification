"""Complete A to Z functions on the data."""
from glob import glob
from multiprocessing import Pool
from pathlib import Path
from random import getrandbits
from shutil import move
from typing import Any, List, Optional, Set

from tqdm import tqdm

from few_shot_face_classification.cache import load_or_build_embeddings_cache
from few_shot_face_classification.data import get_im_paths, load_single
from few_shot_face_classification.embed import (
    _configure_cpu_worker_threads,
    _embed_batch_worker,
    _embed_batch_with_boxes_worker,
    _init_embed_worker,
    embed,
    embed_batch,
    embed_batch_with_boxes,
    get_networks,
    resolve_device,
    resolve_num_workers,
    validate_face,
)
from few_shot_face_classification.exceptions import InvalidImageException
from few_shot_face_classification.similarity import export, get_classes
from few_shot_face_classification.utils import Conflict

_EXPORT_LABELED_PATHS = None
_EXPORT_LABELED_EMBS = None
_EXPORT_WRITE_F = None
_EXPORT_THR = 1.0
_EXPORT_DRAW_BOXES = True
_EXPORT_DEVICE = "cpu"


def _get_available_path(path: Path) -> Path:
    """Return a path that does not overwrite an existing file."""
    candidate = path
    while candidate.exists():
        candidate = candidate.with_name(f"{candidate.stem}_{getrandbits(16)}{candidate.suffix}")
    return candidate


def _move_invalid_label(path: Path, error_dir: Path) -> None:
    """Move an invalid labeled image without overwriting existing files."""
    error_dir.mkdir(exist_ok=True, parents=True)
    dest = _get_available_path(error_dir / path.name)
    print(f"Invalid image '{path}', moving to '{dest}'...")
    move(str(path), str(dest))


def recognise(
        path: Path,
        labeled_f: Path,
        thr: float = 1.,
        batch_size: int = 32,
        cache_file: Optional[Path] = None,
        use_cache: bool = True,
        device: str = "cpu",
        num_workers: Optional[int] = None,
) -> Set[str]:
    """Recognise all labeled faces present in the image, as specified by the provided path."""
    # Load in the image in which the faces are to be recognised
    im = load_single(path)
    
    # Detect faces and embed accordingly
    embs = embed(im, device=device)
    
    # Embed the data
    labeled_paths, labeled_embs = load_or_build_embeddings_cache(
        labeled_f,
        batch_size=batch_size,
        cache_file=cache_file,
        use_cache=use_cache,
        device=device,
        num_workers=num_workers,
    )
    
    # Detect and return all classes
    classes = get_classes(
            embs=embs,
            labeled_paths=labeled_paths,
            labeled_embs=labeled_embs,
            thr=thr,
    )
    return set(classes) - {None, }


def validate_labels(
        labeled_f: Path,
        conflict: Conflict = Conflict.MOVE,
        device: str = "cpu",
) -> None:
    """
    Validate if the labeled data is correct.
    
    :param labeled_f: Folder with labeled data
    :param conflict: How to handle conflict in the data (warn, remove, move, or crash execution)
    """
    labeled_f = Path(labeled_f)

    # Get all image paths to validate
    paths = get_im_paths(labeled_f)
    error_dir = labeled_f.parent / "error_data"
    
    # Load in networks used during validation
    mtcnn, vggface2 = get_networks(device=device)
    
    # Start validation
    for path in paths:
        im = load_single(path)
        if not validate_face(im, val_single=True, mtcnn=mtcnn, vggface2=vggface2, device=device):
            if conflict == Conflict.WARN:
                print(f"Image '{path}' is invalid!")
            elif conflict == Conflict.REMOVE:
                print(f"Invalid image '{path}', removing...")
                path.unlink(missing_ok=True)
            elif conflict == Conflict.MOVE:
                _move_invalid_label(path, error_dir)
            elif conflict == Conflict.CRASH:
                raise InvalidImageException(path)


def detect_and_export(
        raw_f: Path,
        labeled_f: Path,
        write_f: Path,
        batch_size: int = 32,
        thr: float = 1.,
        conflict: Conflict = Conflict.MOVE,
        draw_boxes: bool = True,
        cache_file: Optional[Path] = None,
        use_cache: bool = True,
        device: str = "cpu",
        num_workers: Optional[int] = None,
) -> None:
    """
    Detect all faces in the images and export them to the correct subfolder.
    
    :param raw_f: Folder with raw images to export / classify
    :param labeled_f: Folder with labeled images (faces)
    :param write_f: Folder to which results are written
    :param batch_size: Batch size used during the export
    :param thr: Distance threshold
    :param conflict: How to handle conflict in the data (warn, remove, move, or crash execution)
    :param draw_boxes: Whether to draw face boxes and names on output images
    :param device: Torch device for embedding ("cpu", "cuda", or "auto")
    """
    raw_f = Path(raw_f)
    labeled_f = Path(labeled_f)
    write_f = Path(write_f)

    # First, validate that all labels are indeed correct.
    resolved_device = resolve_device(device)
    validate_labels(labeled_f, conflict=conflict, device=device)

    # Embed the data (cached when possible)
    labeled_paths, labeled_embs = load_or_build_embeddings_cache(
        labeled_f,
        batch_size=batch_size,
        cache_file=cache_file,
        use_cache=use_cache,
        device=device,
        num_workers=num_workers,
    )
    
    # Embed and export by batch, load in images to export first
    paths = get_im_paths(raw_f)
    
    # Split the paths into batches
    chunks: List[Any] = []
    for i in range(0, len(paths), batch_size):
        chunks.append(paths[i:i + batch_size])

    if not chunks:
        return
    
    # Embed and export each chunk
    if resolved_device.type == "cuda":
        mtcnn, vggface2 = get_networks(device=device)
        args = (
            labeled_paths,
            labeled_embs,
            write_f,
            thr,
            draw_boxes,
            device,
            mtcnn,
            vggface2,
        )
        list(tqdm((_embed_and_export(chunk, *args) for chunk in chunks), total=len(chunks), desc="Exporting"))
        return

    workers = min(resolve_num_workers(num_workers), len(chunks)) if chunks else 1
    if workers == 1:
        _configure_cpu_worker_threads()
        mtcnn, vggface2 = get_networks(device=device)
        args = (
            labeled_paths,
            labeled_embs,
            write_f,
            thr,
            draw_boxes,
            device,
            mtcnn,
            vggface2,
        )
        list(tqdm((_embed_and_export(chunk, *args) for chunk in chunks), total=len(chunks), desc="Exporting"))
    else:
        initializer_args = (
            labeled_paths,
            labeled_embs,
            write_f,
            thr,
            draw_boxes,
            device,
        )
        with Pool(workers, initializer=_init_export_worker, initargs=initializer_args) as p:
            list(tqdm(p.imap(_embed_and_export_worker, chunks), total=len(chunks), desc="Exporting"))


def _init_export_worker(
        labeled_paths: List[Path],
        labeled_embs: List[Any],
        write_f: Path,
        thr: float,
        draw_boxes: bool,
        device: str,
) -> None:
    global _EXPORT_LABELED_PATHS, _EXPORT_LABELED_EMBS, _EXPORT_WRITE_F
    global _EXPORT_THR, _EXPORT_DRAW_BOXES, _EXPORT_DEVICE

    _init_embed_worker(device)
    _EXPORT_LABELED_PATHS = labeled_paths
    _EXPORT_LABELED_EMBS = labeled_embs
    _EXPORT_WRITE_F = write_f
    _EXPORT_THR = thr
    _EXPORT_DRAW_BOXES = draw_boxes
    _EXPORT_DEVICE = device


def _embed_and_export(
        paths: List[Path],
        labeled_paths: List[Path],
        labeled_embs: List[Any],
        write_f: Path,
        thr: float,
        draw_boxes: bool,
        device: str,
        mtcnn: Any = None,
        vggface2: Any = None,
) -> None:
    """Embed the given paths and export the results."""
    boxes = None
    if draw_boxes:
        paths, embs, boxes = embed_batch_with_boxes(
            paths,
            device=device,
            mtcnn=mtcnn,
            vggface2=vggface2,
        )
    else:
        paths, embs = embed_batch(
            paths,
            device=device,
            mtcnn=mtcnn,
            vggface2=vggface2,
        )
    
    # Export the results
    export(
            paths=paths,
            embs=embs,
            labeled_paths=labeled_paths,
            labeled_embs=labeled_embs,
            write_f=write_f,
            thr=thr,
            draw_boxes=draw_boxes,
            device=device,
            boxes=boxes,
    )


def _embed_and_export_worker(paths: List[Path]) -> None:
    if _EXPORT_LABELED_PATHS is None or _EXPORT_LABELED_EMBS is None or _EXPORT_WRITE_F is None:
        raise RuntimeError("Export worker was not initialized")

    boxes = None
    if _EXPORT_DRAW_BOXES:
        embedded_paths, embs, boxes = _embed_batch_with_boxes_worker(paths)
    else:
        embedded_paths, embs = _embed_batch_worker(paths)
    export(
            paths=embedded_paths,
            embs=embs,
            labeled_paths=_EXPORT_LABELED_PATHS,
            labeled_embs=_EXPORT_LABELED_EMBS,
            write_f=_EXPORT_WRITE_F,
            thr=_EXPORT_THR,
            draw_boxes=_EXPORT_DRAW_BOXES,
            device=_EXPORT_DEVICE,
            boxes=boxes,
    )


def add_none(
        path: Path,
        labeled_f: Path,
        device: str = "cpu",
) -> None:
    """Add every recognised face in the image to the 'None' class in the labeled folder."""
    # Get the face extraction network
    mtcnn, _ = get_networks(device=device)
    
    # Crop the images
    im = load_single(path)
    hsh = getrandbits(128)
    _ = mtcnn(
            im,
            save_path=str(Path.cwd() / f'{hsh}.png'),
    )
    
    # Move to labeled_f
    tmp_images = glob(str(Path.cwd() / f'{hsh}*.png'))
    n = len(glob(str(labeled_f / 'none_*')))
    for i, tmp_im in enumerate(tmp_images):
        move(
                tmp_im,
                labeled_f / f'none_{n + i + 1}.png',
        )
