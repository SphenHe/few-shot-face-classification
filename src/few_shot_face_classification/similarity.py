"""Check similarities between embeddings and operate accordingly."""
from pathlib import Path
from shutil import copy
from typing import List, Optional

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
    """
    # Derive all the labeled classes
    classes = get_classes(
            embs=embs,
            labeled_paths=labeled_paths,
            labeled_embs=labeled_embs,
            thr=thr,
    )
    
    # Import MTCNN for face detection if drawing boxes
    if draw_boxes:
        from few_shot_face_classification.embed import get_networks, embed
        mtcnn, vggface2 = get_networks()
    
    # Assign images to correct class
    for cls, path in zip(classes, paths):
        # Ignore when no class is recognised
        if cls is None:
            continue
        
        # Ensure class-folder exists
        (write_f / cls).mkdir(parents=True, exist_ok=True)
        output_path = write_f / f"{cls}/{path.name}"
        
        # If drawing boxes is enabled, process the image
        if draw_boxes:
            try:
                # Load image using PIL for consistency with the rest of the code
                im = Image.open(path)
                
                # Detect faces and get their embeddings
                batch_boxes, _ = mtcnn.detect(im)
                
                # Only process if faces were detected
                if batch_boxes is not None and len(batch_boxes) > 0:
                    # Get embeddings for all detected faces
                    face_embs = embed(im, mtcnn=mtcnn, vggface2=vggface2)
                    
                    # Identify each face separately
                    face_names = get_classes(
                        embs=face_embs,
                        labeled_paths=labeled_paths,
                        labeled_embs=labeled_embs,
                        thr=thr,
                    )
                    
                    # Replace None with "Unknown"
                    face_names = [name if name else "Unknown" for name in face_names]
                    
                    # Draw boxes on the image
                    img_with_boxes = _draw_faces_on_image(im, batch_boxes, face_names)
                    
                    # Save the image with boxes
                    img_with_boxes.save(output_path)
                else:
                    # No faces detected, just copy
                    copy(path, output_path)
            except Exception as e:
                # If any error occurs during processing, just copy the original
                print(f"Warning: Could not draw boxes on {path}: {e}")
                copy(path, output_path)
        else:
            # Original behavior: just copy
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
        
        # Draw bounding box
        draw.rectangle([x1, y1, x2, y2], outline=box_color_rgb, width=2)
        
        # Prepare text
        text = str(name) if name else "Unknown"
        
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
            fill=text_bg_color_rgb
        )
        
        # Draw text
        draw.text(
            (text_x + 5, text_y + 5),
            text,
            font=font,
            fill=text_color_rgb
        )
    
    return result
