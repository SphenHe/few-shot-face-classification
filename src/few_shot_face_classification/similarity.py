"""Check similarities between embeddings and operate accordingly."""

import os
from pathlib import Path
from shutil import copy
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

from few_shot_face_classification.utils import get_class

_EPS = 1e-12
_DEFAULT_CLASS_TOP_K = 3
_DEFAULT_CLASS_MARGIN = 0.08
_DEFAULT_MAX_MATCH_DISTANCE = 0.78
_DEFAULT_SINGLE_REFERENCE_MAX_DISTANCE = 0.70


def _resolve_class_top_k() -> int:
    raw = os.getenv("FSFC_CLASS_TOP_K", str(_DEFAULT_CLASS_TOP_K))
    try:
        top_k = int(raw)
    except ValueError:
        raise ValueError("FSFC_CLASS_TOP_K must be an integer")
    if top_k < 1:
        raise ValueError("FSFC_CLASS_TOP_K must be >= 1")
    return top_k


def _resolve_class_margin() -> float:
    raw = os.getenv("FSFC_CLASS_MARGIN", str(_DEFAULT_CLASS_MARGIN))
    try:
        margin = float(raw)
    except ValueError:
        raise ValueError("FSFC_CLASS_MARGIN must be a number")
    if margin < 0:
        raise ValueError("FSFC_CLASS_MARGIN must be >= 0")
    return margin


def _resolve_max_match_distance() -> float:
    raw = os.getenv("FSFC_MAX_MATCH_DISTANCE", str(_DEFAULT_MAX_MATCH_DISTANCE))
    try:
        distance = float(raw)
    except ValueError:
        raise ValueError("FSFC_MAX_MATCH_DISTANCE must be a number")
    if distance <= 0:
        raise ValueError("FSFC_MAX_MATCH_DISTANCE must be > 0")
    return distance


def _resolve_single_reference_max_distance() -> float:
    raw = os.getenv(
        "FSFC_SINGLE_REFERENCE_MAX_DISTANCE",
        str(_DEFAULT_SINGLE_REFERENCE_MAX_DISTANCE),
    )
    try:
        distance = float(raw)
    except ValueError:
        raise ValueError("FSFC_SINGLE_REFERENCE_MAX_DISTANCE must be a number")
    if distance <= 0:
        raise ValueError("FSFC_SINGLE_REFERENCE_MAX_DISTANCE must be > 0")
    return distance


def _as_2d_array(embs: List[np.ndarray]) -> np.ndarray:
    arr = np.asarray(embs, dtype=np.float32)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    return arr


def _l2_normalize(embs: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(embs, axis=1, keepdims=True)
    return embs / np.maximum(norms, _EPS)


def _normalized_euclidean_distances(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    # For L2-normalized face embeddings this is equivalent to ranking by cosine
    # similarity, while keeping the existing threshold scale.
    return np.sqrt(np.maximum(0.0, 2.0 - 2.0 * np.clip(left @ right.T, -1.0, 1.0)))


def _build_reference_sets(
    labeled_paths: List[Path],
    labeled_embs: List[np.ndarray],
) -> Tuple[Dict[str, np.ndarray], Optional[np.ndarray]]:
    labeled_classes = [get_class(p) for p in labeled_paths]
    normalized_embs = _l2_normalize(_as_2d_array(labeled_embs))

    class_refs: Dict[str, List[np.ndarray]] = {}
    negative_refs = []
    for cls, emb in zip(labeled_classes, normalized_embs):
        if cls is None:
            negative_refs.append(emb)
        else:
            class_refs.setdefault(cls, []).append(emb)

    grouped_refs = {cls: np.vstack(refs) for cls, refs in class_refs.items()}
    negatives = np.vstack(negative_refs) if negative_refs else None
    return grouped_refs, negatives


def _class_distances(
    query_embs: np.ndarray,
    class_refs: Dict[str, np.ndarray],
    top_k: int,
) -> Tuple[List[str], np.ndarray]:
    class_names = sorted(class_refs)
    distances = []
    for cls in class_names:
        refs = class_refs[cls]
        sample_distances = _normalized_euclidean_distances(query_embs, refs)
        k = min(top_k, refs.shape[0])
        nearest_mean = np.partition(sample_distances, kth=k - 1, axis=1)[:, :k].mean(
            axis=1
        )

        centroid = _l2_normalize(refs.mean(axis=0, keepdims=True))
        centroid_distance = _normalized_euclidean_distances(query_embs, centroid)[:, 0]

        distances.append(np.minimum(nearest_mean, centroid_distance))
    return class_names, np.vstack(distances).T


def _class_thresholds(
    class_refs: Dict[str, np.ndarray],
    negative_refs: Optional[np.ndarray],
    margin: float,
) -> Dict[str, float]:
    max_match_distance = _resolve_max_match_distance()
    single_reference_max_distance = min(
        max_match_distance,
        _resolve_single_reference_max_distance(),
    )
    thresholds = {
        cls: max_match_distance if refs.shape[0] > 1 else single_reference_max_distance
        for cls, refs in class_refs.items()
    }

    for cls, refs in class_refs.items():
        impostor_refs = []
        for other_cls, other_refs in class_refs.items():
            if other_cls != cls:
                impostor_refs.append(other_refs)
        if negative_refs is not None:
            impostor_refs.append(negative_refs)
        if not impostor_refs:
            continue

        impostors = np.vstack(impostor_refs)
        nearest_impostor_distance = _normalized_euclidean_distances(
            refs, impostors
        ).min()
        thresholds[cls] = min(
            thresholds[cls], max(0.0, nearest_impostor_distance - margin)
        )
    return thresholds


def get_classes(
    embs: List[np.ndarray],
    labeled_paths: List[Path],
    labeled_embs: List[np.ndarray],
    thr: float = 1.0,
) -> List[Optional[str]]:
    """
    Extract the best fitting classes, None if no good match.

    :param embs: Embeddings to classify
    :param labeled_paths: Paths of the labeled embeddings, used to derive class from
    :param labeled_embs: Embeddings of the labeled faces
    :param thr: Distance threshold, return None if no distance falls below it
    """
    if not embs:
        return []
    if not labeled_embs:
        return [None] * len(embs)

    class_refs, negative_refs = _build_reference_sets(labeled_paths, labeled_embs)
    if not class_refs:
        return [None] * len(embs)

    margin = _resolve_class_margin()
    class_thresholds = _class_thresholds(class_refs, negative_refs, margin)
    query_embs = _l2_normalize(_as_2d_array(embs))
    class_names, dist = _class_distances(
        query_embs, class_refs, top_k=_resolve_class_top_k()
    )

    best_indices = dist.argmin(axis=1)
    best_distances = dist[np.arange(dist.shape[0]), best_indices]

    second_distances = np.full(len(embs), np.inf, dtype=np.float32)
    if dist.shape[1] > 1:
        partitioned = np.partition(dist, kth=1, axis=1)
        second_distances = partitioned[:, 1]

    negative_distances = np.full(len(embs), np.inf, dtype=np.float32)
    if negative_refs is not None:
        negative_distances = _normalized_euclidean_distances(
            query_embs, negative_refs
        ).min(axis=1)

    classes: List[Optional[str]] = []
    for best_index, best_distance, second_distance, negative_distance in zip(
        best_indices,
        best_distances,
        second_distances,
        negative_distances,
    ):
        best_class = class_names[best_index]
        max_allowed_distance = min(thr, class_thresholds[best_class])
        if best_distance > max_allowed_distance:
            classes.append(None)
        elif negative_distance <= best_distance + margin:
            classes.append(None)
        elif second_distance - best_distance < margin:
            classes.append(None)
        else:
            classes.append(best_class)
    return classes


def export(
    paths: List[Path],
    embs: List[np.ndarray],
    labeled_paths: List[Path],
    labeled_embs: List[np.ndarray],
    write_f: Path,
    thr: float = 1.0,
    draw_boxes: bool = True,
    device: str = "cpu",
    boxes: Optional[List[np.ndarray]] = None,
) -> None:
    """
    Export (copy) all images to their corresponding class (recognised person).
    支持多个人脸的分别识别。

    :param paths: Paths of the raw images
    :param embs: Embeddings of the faces present in the raw images
    :param labeled_paths: Paths of the labeled images / faces
    :param labeled_embs: Embeddings of the corresponding labeled faces
    :param write_f: Folder to write results to (in corresponding subfolders)
    :param thr: Distance threshold
    :param draw_boxes: Whether to draw face boxes and names on the output images
    :param device: Torch device for embedding ("cpu", "cuda", or "auto")
    :param boxes: Bounding boxes aligned with paths and embeddings
    """
    # Derive all the labeled classes
    classes = get_classes(
        embs=embs,
        labeled_paths=labeled_paths,
        labeled_embs=labeled_embs,
        thr=thr,
    )

    images: Dict[Path, Dict[str, list]] = {}
    for index, (cls, path) in enumerate(zip(classes, paths)):
        item = images.setdefault(path, {"classes": [], "boxes": []})
        item["classes"].append(cls)
        if boxes is not None and index < len(boxes):
            item["boxes"].append(boxes[index])

    mtcnn = None
    if draw_boxes and boxes is None:
        # Backward-compatible fallback for direct callers that do not provide boxes.
        from facenet_pytorch import MTCNN
        from few_shot_face_classification.embed import resolve_device

        mtcnn = MTCNN(keep_all=True, device=resolve_device(device))

    for path, item in images.items():
        recognised_classes = list(
            dict.fromkeys(cls for cls in item["classes"] if cls is not None)
        )
        if not recognised_classes:
            continue

        output_paths = []
        for cls in recognised_classes:
            (write_f / cls).mkdir(parents=True, exist_ok=True)
            output_paths.append(write_f / cls / path.name)

        if not draw_boxes:
            for output_path in output_paths:
                copy(path, output_path)
            continue

        try:
            with Image.open(path) as im:
                face_boxes = item["boxes"]
                if not face_boxes and mtcnn is not None:
                    detected_boxes, _ = mtcnn.detect(im)
                    face_boxes = [] if detected_boxes is None else list(detected_boxes)

                if not face_boxes:
                    for output_path in output_paths:
                        copy(path, output_path)
                    continue

                face_names = [name if name else "Unknown" for name in item["classes"]]
                if len(face_names) < len(face_boxes):
                    face_names.extend(["Unknown"] * (len(face_boxes) - len(face_names)))
                image_with_boxes = _draw_faces_on_image(
                    im, np.asarray(face_boxes), face_names
                )
                for output_path in output_paths:
                    image_with_boxes.save(output_path)
        except Exception as exc:
            print(f"Warning: Could not draw boxes on {path}: {exc}")
            for output_path in output_paths:
                copy(path, output_path)


def _draw_faces_on_image(
    image,
    boxes: np.ndarray,
    names: List[str],
    box_color: tuple = (0, 255, 0),
    text_color: tuple = (0, 0, 0),
    text_bg_color: tuple = (0, 255, 0),
):
    """
    Draw face boxes and names on the image.
    支持中文文本显示，保持原始色彩不变。

    :param image: PIL Image or numpy array
    :param boxes: Face bounding boxes from MTCNN [[x1, y1, x2, y2], ...]
    :param names: List of names corresponding to each face
    :param box_color: BGR color tuple for boxes (default green)
    :param text_color: BGR color tuple for text (default white)
    :param text_bg_color: BGR color tuple for text background (default green)
    :return: PIL Image with drawn boxes and names
    """
    from PIL import Image, ImageDraw, ImageFont

    # Ensure image is PIL Image (not numpy array)
    if isinstance(image, np.ndarray):
        # Check if it's BGR or RGB
        if len(image.shape) == 3 and image.shape[2] == 3:
            # Assume it's RGB from PIL (correct format)
            image = Image.fromarray(image, mode="RGB")
        else:
            image = Image.fromarray(image)
    else:
        # Already PIL Image, just make a copy to avoid modifying original
        image = image.copy()

    # Create a copy to draw on
    result = image.copy()
    draw = ImageDraw.Draw(result)

    # Try to load a font that supports Chinese characters
    font_size = 20
    try:
        # Try common Chinese font paths on different platforms
        font_paths = [
            # Linux fonts (common locations)
            "/usr/share/fonts/truetype/NotoSansCJKsc-VF.otf",
            "/usr/share/fonts/truetype/SourceHanSansCN-Normal.otf",
            "/usr/share/fonts/truetype/SourceHanSansCN-Regular.otf",
            "/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf",
            "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
            "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
            "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
            "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
            # Windows fonts
            "C:\\Windows\\Fonts\\simhei.ttf",  # SimHei (黑体)
            "C:\\Windows\\Fonts\\simsun.ttc",  # SimSun (宋体)
            "C:\\Windows\\Fonts\\msyh.ttc",  # Microsoft YaHei (微软雅黑)
            # macOS fonts
            "/System/Library/Fonts/PingFang.ttc",
            "/Library/Fonts/Arial Unicode.ttf",
        ]
        font = None
        for font_path in font_paths:
            try:
                font = ImageFont.truetype(font_path, font_size)
                break
            except:
                continue

        if font is None:
            # Fallback to default font if no Chinese font found
            font = ImageFont.load_default()
    except:
        font = ImageFont.load_default()

    # Convert BGR colors to RGB for PIL
    box_color_rgb = (box_color[2], box_color[1], box_color[0])
    text_color_rgb = (text_color[2], text_color[1], text_color[0])
    text_bg_color_rgb = (text_bg_color[2], text_bg_color[1], text_bg_color[0])

    # Draw boxes and text
    for box, name in zip(boxes, names):
        x1, y1, x2, y2 = [int(v) for v in box]

        # Unknown faces use red boxes and labels; recognised faces keep the
        # configured colours.
        text = str(name) if name else "Unknown"
        is_unknown = text.strip().lower() == "unknown"
        current_box_color = (255, 0, 0) if is_unknown else box_color_rgb
        current_text_color = (255, 0, 0) if is_unknown else text_color_rgb
        current_text_bg_color = (255, 255, 255) if is_unknown else text_bg_color_rgb

        # Draw bounding box
        draw.rectangle([x1, y1, x2, y2], outline=current_box_color, width=2)

        # Get text bounding box for PIL
        try:
            bbox = draw.textbbox((0, 0), text, font=font)
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]
        except:
            text_width = len(text) * 10
            text_height = font_size

        # Calculate text position (above the face box)
        text_x = x1
        text_y = max(y1 - text_height - 10, 5)

        # Draw text background
        draw.rectangle(
            [text_x, text_y, text_x + text_width + 10, text_y + text_height + 10],
            fill=current_text_bg_color,
        )

        # Draw text
        draw.text((text_x + 5, text_y + 5), text, font=font, fill=current_text_color)

    return result
