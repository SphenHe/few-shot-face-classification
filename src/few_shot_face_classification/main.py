"""Complete A to Z functions on the data."""
import pickle
from glob import glob
from multiprocessing import Pool, cpu_count
from pathlib import Path
from random import getrandbits
from shutil import move
from typing import Any, List, Optional, Set, Tuple

from tqdm import tqdm

from few_shot_face_classification.data import get_im_paths, load_single
from few_shot_face_classification.embed import embed, embed_batch, embed_folder, get_networks, validate_face
from few_shot_face_classification.exceptions import InvalidImageException
from few_shot_face_classification.similarity import export, get_classes
from few_shot_face_classification.utils import Conflict


def _load_or_create_embeddings(
        labeled_f: Path,
        batch_size: int = 32,
        cache_file: Optional[Path] = None,
        use_cache: bool = True,
) -> Tuple[List[Path], List[Any]]:
    """Load labeled embeddings from cache when valid, otherwise compute and persist.

    The cache is considered valid when it exists and is newer than any file in the
    labeled folder. If loading fails, we transparently recompute and overwrite.
    """
    # Short-circuit if caching is disabled or no cache file provided
    if not use_cache or cache_file is None:
        return embed_folder(labeled_f, batch_size=batch_size)

    def _restore_paths(raw_paths):
        # Rebuild paths relative to labeled folder for cross-platform portability
        restored = []
        for p in raw_paths:
            restored.append(labeled_f / Path(p))
        return restored

    cache_valid = False
    if cache_file.exists():
        cache_mtime = cache_file.stat().st_mtime
        labeled_files = [f for f in labeled_f.glob("*") if f.is_file()]
        if labeled_files:
            newest_labeled = max(f.stat().st_mtime for f in labeled_files)
            cache_valid = cache_mtime > newest_labeled

    if cache_valid:
        try:
            with open(cache_file, "rb") as f:
                data = pickle.load(f)
            labeled_paths = _restore_paths(data["paths"])
            labeled_embs = data["embeddings"]
            return labeled_paths, labeled_embs
        except Exception:
            # If cache read fails, fall back to recompute
            pass

    labeled_paths, labeled_embs = embed_folder(labeled_f, batch_size=batch_size)

    try:
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        rel_paths = [p.relative_to(labeled_f).as_posix() for p in labeled_paths]
        with open(cache_file, "wb") as f:
            pickle.dump({"paths": rel_paths, "embeddings": labeled_embs}, f)
    except Exception:
        # Cache write failure should not block main flow
        pass

    return labeled_paths, labeled_embs


def recognise(
        path: Path,
        labeled_f: Path,
        thr: float = 1.,
) -> Set[str]:
    """Recognise all labeled faces present in the image, as specified by the provided path."""
    # Load in the image in which the faces are to be recognised
    im = load_single(path)
    
    # Detect faces and embed accordingly
    embs = embed(im)
    
    # Embed the data
    labeled_paths, labeled_embs = embed_folder(labeled_f)
    
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
        conflict: Conflict = Conflict.CRASH,
) -> None:
    """
    Validate if the labeled data is correct.
    
    :param labeled_f: Folder with labeled data
    :param conflict: How to handle conflict in the data (warn, remove, or crash execution)
    """
    # Get all image paths to validate
    paths = get_im_paths(labeled_f)
    
    # Load in networks used during validation
    mtcnn, vggface2 = get_networks()
    
    # Start validation
    for path in paths:
        im = load_single(path)
        if not validate_face(im, val_single=True, mtcnn=mtcnn, vggface2=vggface2):
            if conflict == Conflict.WARN:
                print(f"Image '{path}' is invalid!")
            elif conflict == Conflict.REMOVE:
                print(f"Invalid image '{path}', removing...")
                path.unlink(missing_ok=True)
            elif conflict == Conflict.CRASH:
                raise InvalidImageException(path)


def detect_and_export(
        raw_f: Path,
        labeled_f: Path,
        write_f: Path,
        batch_size: int = 32,
        thr: float = 1.,
        conflict: Conflict = Conflict.CRASH,
        draw_boxes: bool = True,
    cache_file: Optional[Path] = None,
    use_cache: bool = True,
) -> None:
    """
    Detect all faces in the images and export them to the correct subfolder.
    
    :param raw_f: Folder with raw images to export / classify
    :param labeled_f: Folder with labeled images (faces)
    :param write_f: Folder to which results are written
    :param batch_size: Batch size used during the export
    :param thr: Distance threshold
    :param conflict: How to handle conflict in the data (warn, remove, or crash execution)
    :param draw_boxes: Whether to draw face boxes and names on output images
    """
    # First, validate that all labels are indeed correct. On crash, move bad images aside and retry.
    if conflict == Conflict.CRASH:
        error_dir = labeled_f.parent / "error_data"
        error_dir.mkdir(exist_ok=True, parents=True)

        while True:
            try:
                validate_labels(labeled_f, conflict=conflict)
                break
            except InvalidImageException as exc:
                bad_path = getattr(exc, "path", None)
                if bad_path is None:
                    raise

                bad_path = Path(bad_path)
                dest = error_dir / bad_path.name
                while dest.exists():
                    dest = dest.with_name(f"{dest.stem}_{getrandbits(16)}{dest.suffix}")

                print(f"Invalid image '{bad_path}', moving to '{dest}' and retrying validation...")
                move(str(bad_path), dest)
    else:
        validate_labels(labeled_f, conflict=conflict)
    
        # Embed the data (cached when possible)
        labeled_paths, labeled_embs = _load_or_create_embeddings(
            labeled_f,
            batch_size=batch_size,
            cache_file=cache_file,
            use_cache=use_cache,
        )
    
    # Embed and export by batch, load in images to export first
    paths = get_im_paths(raw_f)
    
    # Split the paths into batches
    chunks: List[Any] = []
    for i in range(0, len(paths), batch_size):
        chunks.append((
            paths[i:i + batch_size],
            labeled_paths,
            labeled_embs,
            write_f,
            thr,
            draw_boxes,
        ))
    
    # Embed and export each chunk
    with Pool(cpu_count() - 2) as p:
        list(tqdm(p.imap(_embed_and_export, chunks), total=len(chunks), desc="Exporting"))


def _embed_and_export(
        args: List[Any],
) -> None:
    """Embed the given paths and export the results."""
    # Unfold the arguments
    paths, labeled_paths, labeled_embs, write_f, thr, draw_boxes = args
    
    # Create the embeddings
    paths, embs = embed_batch(paths)
    
    # Export the results
    export(
            paths=paths,
            embs=embs,
            labeled_paths=labeled_paths,
            labeled_embs=labeled_embs,
            write_f=write_f,
            thr=thr,
            draw_boxes=draw_boxes,
    )


def add_none(
        path: Path,
        labeled_f: Path,
) -> None:
    """Add every recognised face in the image to the 'None' class in the labeled folder."""
    # Get the face extraction network
    mtcnn, _ = get_networks()
    
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
