"""Check similarities between embeddings and operate accordingly."""
from pathlib import Path
from shutil import copy
from typing import Dict, List, Optional

import numpy as np
from PIL import Image
from sklearn.metrics.pairwise import euclidean_distances

from few_shot_face_classification.utils import get_class


def get_classes(
        embs: List[np.ndarray],
        labeled_paths: List[Path],
        labeled_embs: List[np.ndarray],
        thr: float = 1.,
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

    # Get all classes that belong to the labeled embeddings
    labeled_classes = [get_class(p) for p in labeled_paths]
    
    # Calculate the distance between embeddings
    dist = euclidean_distances(embs, labeled_embs)
    
    # Derive the best suiting class
    classes = []
    for d in dist:
        classes.append(
                labeled_classes[np.where(d == min(d))[0][0]]
                if min(d) <= thr
                else None
        )
    return classes


def export(
        paths: List[Path],
        embs: List[np.ndarray],
        labeled_paths: List[Path],
        labeled_embs: List[np.ndarray],
        write_f: Path,
        thr: float = 1.,
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
        recognised_classes = list(dict.fromkeys(cls for cls in item["classes"] if cls is not None))
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
                image_with_boxes = _draw_faces_on_image(im, np.asarray(face_boxes), face_names)
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
            image = Image.fromarray(image, mode='RGB')
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
            "C:\\Windows\\Fonts\\msyh.ttc",    # Microsoft YaHei (微软雅黑)
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
            fill=current_text_bg_color
        )
        
        # Draw text
        draw.text(
            (text_x + 5, text_y + 5),
            text,
            font=font,
            fill=current_text_color
        )
    
    return result
