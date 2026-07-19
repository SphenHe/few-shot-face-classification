"""Methods to embed results."""
import math
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

_APPROX_MODEL_MEMORY_GB_PER_WORKER = 1.5
_DEFAULT_MAX_AUTO_WORKERS = 32
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
    """Resolve a CPU worker count for model-heavy multiprocessing."""
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

    available_cpus = _available_cpu_count()
    torch_threads = resolve_torch_threads()

    # Each worker owns an MTCNN and embedding model, so process count should grow
    # sublinearly with available CPUs. Very large CPU machines are often limited
    # by memory bandwidth, image decode, and model duplication before they are
    # limited by core count, so cpu/2 is too aggressive at 192+ cores.
    cpu_workers = max(1, math.floor(available_cpus / torch_threads))
    if available_cpus <= 2:
        suggested_workers = 1
    elif available_cpus <= 4:
        suggested_workers = 2
    else:
        suggested_workers = max(2, math.ceil(math.sqrt(available_cpus) * 2))
    cpu_workers = min(cpu_workers, suggested_workers, resolve_max_auto_workers())

    memory_workers = _estimate_memory_limited_workers()
    if memory_workers is not None:
        cpu_workers = min(cpu_workers, memory_workers)
    return max(1, cpu_workers)


def resolve_torch_threads() -> int:
    """Resolve PyTorch intra-op thread count used inside CPU workers."""
    raw = os.getenv("FSFC_TORCH_THREADS", "1")
    try:
        threads = int(raw)
    except ValueError:
        raise ValueError("FSFC_TORCH_THREADS must be an integer")
    if threads < 1:
        raise ValueError("FSFC_TORCH_THREADS must be >= 1")
    return threads


def resolve_max_auto_workers() -> int:
    """Resolve the upper bound for automatic CPU worker selection."""
    raw = os.getenv("FSFC_MAX_AUTO_WORKERS")
    if not raw:
        return _DEFAULT_MAX_AUTO_WORKERS
    try:
        max_workers = int(raw)
    except ValueError:
        raise ValueError("FSFC_MAX_AUTO_WORKERS must be an integer")
    if max_workers < 1:
        raise ValueError("FSFC_MAX_AUTO_WORKERS must be >= 1")
    return max_workers


def resolve_pool_chunksize(total_chunks: int, workers: int) -> int:
    """Resolve multiprocessing imap chunksize for batched image work."""
    raw = os.getenv("FSFC_POOL_CHUNKSIZE")
    if raw:
        try:
            chunksize = int(raw)
        except ValueError:
            raise ValueError("FSFC_POOL_CHUNKSIZE must be an integer")
        if chunksize < 1:
            raise ValueError("FSFC_POOL_CHUNKSIZE must be >= 1")
        return chunksize

    if total_chunks <= 0 or workers <= 1:
        return 1

    # Keep enough tasks in flight for load balancing, but avoid paying an IPC
    # round trip for every small batch on large datasets.
    return max(1, min(8, math.ceil(total_chunks / (workers * 8))))


def resolve_face_batch_size() -> int:
    """Resolve how many cropped faces to embed in one network forward pass."""
    raw = os.getenv("FSFC_FACE_BATCH_SIZE", "64")
    try:
        batch_size = int(raw)
    except ValueError:
        raise ValueError("FSFC_FACE_BATCH_SIZE must be an integer")
    if batch_size < 1:
        raise ValueError("FSFC_FACE_BATCH_SIZE must be >= 1")
    return batch_size


def _available_cpu_count() -> int:
    """Return CPUs available to this process, respecting affinity and cgroups."""
    try:
        available = len(os.sched_getaffinity(0))
    except (AttributeError, OSError):
        available = cpu_count()

    quota_paths = (
        ("/sys/fs/cgroup/cpu.max", None),
        ("/sys/fs/cgroup/cpu/cpu.cfs_quota_us", "/sys/fs/cgroup/cpu/cpu.cfs_period_us"),
    )
    for quota_path, period_path in quota_paths:
        try:
            if period_path is None:
                quota_raw, period_raw = Path(quota_path).read_text().strip().split()
                if quota_raw == "max":
                    continue
            else:
                quota_raw = Path(quota_path).read_text().strip()
                period_raw = Path(period_path).read_text().strip()
            quota, period = int(quota_raw), int(period_raw)
            if quota > 0 and period > 0:
                available = min(available, max(1, math.ceil(quota / period)))
                break
        except (OSError, ValueError):
            continue
    return max(1, available)


def _estimate_memory_limited_workers() -> Optional[int]:
    """Estimate how many model-owning workers fit in currently available RAM."""
    try:
        with open("/proc/meminfo") as f:
            values = {}
            for line in f:
                key, value = line.split(":", 1)
                values[key] = int(value.strip().split()[0])
    except (OSError, ValueError):
        return None

    available_kb = values.get("MemAvailable")
    if available_kb is None:
        return None

    available_bytes = available_kb * 1024
    cgroup_available = _get_cgroup_available_memory()
    if cgroup_available is not None:
        available_bytes = min(available_bytes, cgroup_available)

    available_gb = available_bytes / 1024 / 1024 / 1024
    reserved_gb = max(1.0, available_gb * 0.2)
    usable_gb = max(0.0, available_gb - reserved_gb)
    return max(1, int(usable_gb // _APPROX_MODEL_MEMORY_GB_PER_WORKER))


def _get_cgroup_available_memory() -> Optional[int]:
    """Return remaining cgroup memory in bytes when a finite limit is set."""
    paths = (
        ("/sys/fs/cgroup/memory.max", "/sys/fs/cgroup/memory.current"),
        (
            "/sys/fs/cgroup/memory/memory.limit_in_bytes",
            "/sys/fs/cgroup/memory/memory.usage_in_bytes",
        ),
    )
    for limit_path, usage_path in paths:
        try:
            limit_raw = Path(limit_path).read_text().strip()
            if limit_raw == "max":
                continue
            limit = int(limit_raw)
            usage = int(Path(usage_path).read_text().strip())
            # cgroup v1 may expose an effectively unlimited, near-2^63 value.
            if 0 < limit < (1 << 60):
                return max(0, limit - usage)
        except (OSError, ValueError):
            continue
    return None


def _configure_cpu_worker_threads() -> None:
    """Avoid multiplying PyTorch/OpenMP threads inside each worker process."""
    threads = resolve_torch_threads()

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
    # Try to embed the face, catch exceptions if they happen
    try:
        embeddings, _ = embed_with_boxes(
            im,
            mtcnn=mtcnn,
            vggface2=vggface2,
            device=device,
        )

        # Check if strictly one face recognised
        if val_single and len(embeddings) == 0:
            print("No face")
            raise NoFaceException
        elif val_single and len(embeddings) > 1:
            print("Multi face")
            raise MultipleFaceException
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
    embeddings, _ = embed_with_boxes(
        im,
        mtcnn=mtcnn,
        vggface2=vggface2,
        device=device,
    )
    return embeddings


def embed_with_boxes(
        im: Image,
        mtcnn: Optional[MTCNN] = None,
        vggface2: Optional[InceptionResnetV1] = None,
        device: str = "cpu",
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """Detect faces once and return aligned embeddings and bounding boxes."""
    if mtcnn is None or vggface2 is None:
        mtcnn, vggface2 = get_networks(device=device)
    resolved_device = _network_device(vggface2)

    boxes, _ = mtcnn.detect(im)
    if boxes is None or len(boxes) == 0:
        return [], []

    img_cropped = mtcnn.extract(im, boxes, save_path=None)
    if img_cropped is None:
        return [], []
    if img_cropped.ndim == 3:
        img_cropped = img_cropped.unsqueeze(0)

    embeddings = _embed_face_tensor_batch(img_cropped, vggface2, resolved_device)
    return embeddings, [np.asarray(box) for box in boxes[:len(embeddings)]]


def _embed_face_tensor_batch(
        faces: torch.Tensor,
        vggface2: InceptionResnetV1,
        device: torch.device,
        face_batch_size: Optional[int] = None,
) -> List[np.ndarray]:
    """Embed cropped face tensors in model-sized batches."""
    if faces.numel() == 0:
        return []
    if faces.ndim == 3:
        faces = faces.unsqueeze(0)

    face_batch_size = face_batch_size or resolve_face_batch_size()
    outputs: List[np.ndarray] = []
    with torch.no_grad():
        for i in range(0, faces.shape[0], face_batch_size):
            batch = faces[i:i + face_batch_size].to(device)
            embedded = vggface2(batch).detach().cpu().numpy()
            outputs.extend(embedded)
    return outputs


def embed_folder(
        folder: Path,
        batch_size: int = 32,
        device: str = "cpu",
        num_workers: Optional[int] = None,
) -> Tuple[List[Path], List[np.ndarray]]:
    """Embed all the images in the requested folder."""
    paths = get_im_paths(folder)
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
    if not chunks:
        return [], []

    if resolved_device.type == "cuda":
        mtcnn, vggface2 = get_networks(device=device)
        results = [
            embed_batch(
                chunk,
                device=device,
                mtcnn=mtcnn,
                vggface2=vggface2,
            )
            for chunk in tqdm(chunks, total=len(chunks), desc="Processing")
        ]
        return [x for y in results for x in y[0]], [x for y in results for x in y[1]]

    workers = min(resolve_num_workers(num_workers), len(chunks))
    if workers == 1:
        _configure_cpu_worker_threads()
        mtcnn, vggface2 = get_networks(device=device)
        results = [
            embed_batch(chunk, mtcnn=mtcnn, vggface2=vggface2, device=device)
            for chunk in tqdm(chunks, total=len(chunks), desc="Processing")
        ]
    else:
        with Pool(workers, initializer=_init_embed_worker, initargs=(device,)) as p:
            results = list(tqdm(
                p.imap(_embed_batch_worker, chunks, chunksize=resolve_pool_chunksize(len(chunks), workers)),
                total=len(chunks),
                desc="Processing",
            ))
    
    return [x for y in results for x in y[0]], [x for y in results for x in y[1]]


def embed_batch(
        paths: List[Path],
        device: str = "cpu",
        mtcnn: Optional[MTCNN] = None,
        vggface2: Optional[InceptionResnetV1] = None,
) -> Tuple[List[Path], List[np.ndarray]]:
    """Embed a batch of images as specified by their path, used in multiprocessing."""
    return_paths, return_embeddings, _ = embed_batch_with_boxes(
        paths,
        device=device,
        mtcnn=mtcnn,
        vggface2=vggface2,
    )
    return return_paths, return_embeddings


def embed_batch_with_boxes(
        paths: List[Path],
        device: str = "cpu",
        mtcnn: Optional[MTCNN] = None,
        vggface2: Optional[InceptionResnetV1] = None,
) -> Tuple[List[Path], List[np.ndarray], List[np.ndarray]]:
    """Embed a batch and return one bounding box aligned with each embedding."""
    if mtcnn is None or vggface2 is None:
        mtcnn, vggface2 = get_networks(device=device)

    resolved_device = _network_device(vggface2)
    face_batch_size = resolve_face_batch_size()
    return_paths, return_boxes = [], []
    cropped_faces: List[torch.Tensor] = []
    for path in paths:
        im = load_single(path)
        boxes, _ = mtcnn.detect(im)
        if boxes is None or len(boxes) == 0:
            continue

        img_cropped = mtcnn.extract(im, boxes, save_path=None)
        if img_cropped is None:
            continue
        if img_cropped.ndim == 3:
            img_cropped = img_cropped.unsqueeze(0)

        face_count = min(len(boxes), img_cropped.shape[0])
        if face_count == 0:
            continue
        cropped_faces.append(img_cropped[:face_count])
        return_paths += [path] * face_count
        return_boxes += [np.asarray(box) for box in boxes[:face_count]]

    if not cropped_faces:
        return [], [], []

    faces = torch.cat(cropped_faces, dim=0)
    return_embeddings = _embed_face_tensor_batch(
        faces,
        vggface2,
        resolved_device,
        face_batch_size=face_batch_size,
    )
    return return_paths, return_embeddings, return_boxes


def _embed_batch_worker(paths: List[Path]) -> Tuple[List[Path], List[np.ndarray]]:
    if _WORKER_MTCNN is None or _WORKER_VGGFACE2 is None:
        return embed_batch(paths)
    return embed_batch(paths, device=_WORKER_DEVICE, mtcnn=_WORKER_MTCNN, vggface2=_WORKER_VGGFACE2)


def _embed_batch_with_boxes_worker(
        paths: List[Path],
) -> Tuple[List[Path], List[np.ndarray], List[np.ndarray]]:
    if _WORKER_MTCNN is None or _WORKER_VGGFACE2 is None:
        return embed_batch_with_boxes(paths)
    return embed_batch_with_boxes(
        paths,
        device=_WORKER_DEVICE,
        mtcnn=_WORKER_MTCNN,
        vggface2=_WORKER_VGGFACE2,
    )
