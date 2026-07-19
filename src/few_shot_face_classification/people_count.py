"""Count people in photos by detecting face boxes."""
from pathlib import Path
from typing import Dict, Optional

from facenet_pytorch import MTCNN
from tqdm import tqdm

from few_shot_face_classification.data import get_im_paths, load_single
from few_shot_face_classification.embed import resolve_device


def count_people(
        path: Path,
        mtcnn: Optional[MTCNN] = None,
        device: str = "cpu",
) -> int:
    """
    Count people in a photo by counting detected faces.

    :param path: Image path
    :param mtcnn: Optional reusable MTCNN detector
    :param device: Torch device for detection ("cpu", "cuda", or "auto")
    :return: Number of detected faces
    """
    if mtcnn is None:
        mtcnn = MTCNN(keep_all=True, device=resolve_device(device))

    im = load_single(Path(path))
    boxes, _ = mtcnn.detect(im)
    return 0 if boxes is None else len(boxes)


def count_people_in_folder(
        folder: Path,
        device: str = "cpu",
        show_progress: bool = True,
) -> Dict[Path, int]:
    """
    Count people in every image under a folder.

    :param folder: Folder containing photos
    :param device: Torch device for detection ("cpu", "cuda", or "auto")
    :param show_progress: Whether to show a progress bar
    :return: Mapping from image path to detected person count
    """
    folder = Path(folder)
    paths = get_im_paths(folder)
    mtcnn = MTCNN(keep_all=True, device=resolve_device(device))
    iterable = tqdm(paths, desc="Counting people") if show_progress else paths
    return {path: count_people(path, mtcnn=mtcnn, device=device) for path in iterable}
