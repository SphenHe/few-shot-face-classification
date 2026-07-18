"""Methods to embed results."""
import os
import warnings
from multiprocessing import Pool, cpu_count
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
from PIL import Image
from facenet_pytorch import InceptionResnetV1, MTCNN
import torch
from tqdm import tqdm

from few_shot_face_classification.data import get_im_paths, load_single
from few_shot_face_classification.exceptions import MultipleFaceException, NoFaceException

# Filter out the user warnings
warnings.filterwarnings("ignore", category=UserWarning)

_DEFAULT_MAX_CPU_WORKERS = 4
_WORKER_MTCNN = None
_WORKER_VGGFACE2 = None
_WORKER_DEVICE = "cpu"


def resolve_device(device: str = "cpu") -> torch.device:
    """Resolve and validate the requested torch device."""
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    resolved = torch.device(device)
    if resolved.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA device requested, but PyTorch cannot access CUDA. "
            "Install a CUDA-enabled torch build and verify torch.cuda.is_available()."
        )
    return resolved


def _network_device(vggface2: InceptionResnetV1) -> torch.device:
    return next(vggface2.parameters()).device


def get_networks(device: str = "cpu") -> Tuple[MTCNN, InceptionResnetV1]:
    """Get all the networks for image detection."""
    resolved_device = resolve_device(device)

    # Create MTCNN network that extracts all potential faces from the images
    mtcnn = MTCNN(keep_all=True, device=resolved_device)
    
    # Use the VGGFace2 to create the embedding
    vggface2 = InceptionResnetV1(pretrained='vggface2').to(resolved_device).eval()
    return mtcnn, vggface2


def resolve_num_workers(num_workers: Optional[int] = None) -> int:
    """Resolve a conservative CPU worker count for model-heavy multiprocessing."""
    if num_workers is None:
        raw = os.getenv("FSFC_NUM_WORKERS")
        if raw:
            try:
                num_workers = int(raw)
            except ValueError:
                raise ValueError("FSFC_NUM_WORKERS must be an integer")

    if num_workers is not None:
        if num_workers < 1:
            raise ValueError("num_workers must be >= 1")
        return num_workers

    return max(1, min(cpu_count() - 2, _DEFAULT_MAX_CPU_WORKERS))


def _configure_cpu_worker_threads() -> None:
    """Avoid multiplying PyTorch/OpenMP threads inside each worker process."""
    raw = os.getenv("FSFC_TORCH_THREADS", "1")
    try:
        threads = max(1, int(raw))
    except ValueError:
        raise ValueError("FSFC_TORCH_THREADS must be an integer")

    torch.set_num_threads(threads)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass


def _init_embed_worker(device: str) -> None:
    global _WORKER_MTCNN, _WORKER_VGGFACE2, _WORKER_DEVICE
    _WORKER_DEVICE = device
    if resolve_device(device).type == "cpu":
        _configure_cpu_worker_threads()
    _WORKER_MTCNN, _WORKER_VGGFACE2 = get_networks(device=device)


def validate_face(
        im: Image,
        val_single: bool,
        mtcnn: Optional[MTCNN] = None,
        vggface2: Optional[InceptionResnetV1] = None,
        device: str = "cpu",
) -> bool:
    """
    Validate the image on detected faces.
    
    :param im: Image to validate
    :param val_single: Validate that strictly one image is present in the image
    :param mtcnn: MTCNN network for face extraction
    :param vggface2: VGGFace2 network to embed the face
    """
    # Create MTCNN network if not provided
    if mtcnn is None or vggface2 is None:
        mtcnn, vggface2 = get_networks(device=device)
    resolved_device = _network_device(vggface2)
    
    # Try to embed the face, catch exceptions if they happen
    try:
        # Check if image can be cropped
        img_cropped = mtcnn(im)
        
        # Check if strictly one face recognised
        if val_single and img_cropped.shape[0] == 0:
            print("No face")
            raise NoFaceException
        elif val_single and img_cropped.shape[0] > 1:
            print("Multi face")
            raise MultipleFaceException
        
        # Check if embedding happens correctly
        with torch.no_grad():
            for face_arr in img_cropped:
                _ = vggface2(face_arr.unsqueeze(0).to(resolved_device)).detach().cpu().numpy()[0]
    except KeyboardInterrupt:
        raise KeyboardInterrupt
    except Exception:
        return False
    return True


def embed(
        im: Image,
        mtcnn: Optional[MTCNN] = None,
        vggface2: Optional[InceptionResnetV1] = None,
        device: str = "cpu",
) -> List[np.ndarray]:
    """
    Create embeddings for every face detected by the algorithm.
    
    :param im: Image to embed
    :param mtcnn: MTCNN network for face extraction
    :param vggface2: VGGFace2 network to embed the face
    """
    # Create MTCNN network if not provided
    if mtcnn is None or vggface2 is None:
        mtcnn, vggface2 = get_networks(device=device)
    resolved_device = _network_device(vggface2)
    
    # Crop out the faces, return empty list if none detected
    img_cropped = mtcnn(im)
    if img_cropped is None:
        return []
    
    # Embed all detected faces
    embeddings = []
    with torch.no_grad():
        for face_arr in img_cropped:
            embeddings.append(
                    vggface2(face_arr.unsqueeze(0).to(resolved_device)).detach().cpu().numpy()[0]
            )
    return embeddings


def embed_folder(
        folder: Path,
        batch_size: int = 32,
        device: str = "cpu",
        num_workers: Optional[int] = None,
) -> Tuple[List[Path], List[np.ndarray]]:
    """Embed all the images in the requested folder."""
    # Load in all the files to embed
    paths = get_im_paths(folder)
    
    # Split the paths into batches
    chunks = []
    for i in range(0, len(paths), batch_size):
        chunks.append(paths[i:i + batch_size])
    
    return embed_paths(paths, batch_size=batch_size, device=device, num_workers=num_workers)


def embed_paths(
        paths: List[Path],
        batch_size: int = 32,
        device: str = "cpu",
        num_workers: Optional[int] = None,
) -> Tuple[List[Path], List[np.ndarray]]:
    """Embed the requested image paths."""
    chunks = []
    for i in range(0, len(paths), batch_size):
        chunks.append(paths[i:i + batch_size])

    resolved_device = resolve_device(device)
    if resolved_device.type == "cuda" or not chunks:
        results = [
            embed_batch(chunk, device=device)
            for chunk in tqdm(chunks, total=len(chunks), desc="Processing")
        ]
        return [x for y in results for x in y[0]], [x for y in results for x in y[1]]

    workers = min(resolve_num_workers(num_workers), len(chunks))
    if workers == 1:
        mtcnn, vggface2 = get_networks(device=device)
        results = [
            embed_batch(chunk, mtcnn=mtcnn, vggface2=vggface2, device=device)
            for chunk in tqdm(chunks, total=len(chunks), desc="Processing")
        ]
    else:
        with Pool(workers, initializer=_init_embed_worker, initargs=(device,)) as p:
            results = list(tqdm(p.imap(_embed_batch_worker, chunks), total=len(chunks), desc="Processing"))
    
    return [x for y in results for x in y[0]], [x for y in results for x in y[1]]


def embed_batch(
        paths: List[Path],
        device: str = "cpu",
        mtcnn: Optional[MTCNN] = None,
        vggface2: Optional[InceptionResnetV1] = None,
) -> Tuple[List[Path], List[np.ndarray]]:
    """Embed a batch of images as specified by their path, used in multiprocessing."""
    # Load in the networks
    if mtcnn is None or vggface2 is None:
        mtcnn, vggface2 = get_networks(device=device)
    
    # Embed all the images
    return_path, return_arr = [], []
    for path in paths:
        im = load_single(path)
        emb = embed(
                im=im,
                mtcnn=mtcnn,
                vggface2=vggface2,
                device=device,
        )
        return_path += [path] * len(emb)
        return_arr += emb
    return return_path, return_arr


def _embed_batch_worker(paths: List[Path]) -> Tuple[List[Path], List[np.ndarray]]:
    if _WORKER_MTCNN is None or _WORKER_VGGFACE2 is None:
        return embed_batch(paths)
    return embed_batch(paths, device=_WORKER_DEVICE, mtcnn=_WORKER_MTCNN, vggface2=_WORKER_VGGFACE2)
