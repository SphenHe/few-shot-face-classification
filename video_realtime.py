import argparse
import pickle
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from src.few_shot_face_classification.embed import embed, embed_folder, get_networks
from src.few_shot_face_classification.similarity import get_classes, _draw_faces_on_image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Real-time face recognition with OpenCV video input")
    parser.add_argument("--labeled", type=Path, default=Path("data/labeled"), help="Folder with labeled reference faces")
    parser.add_argument("--threshold", type=float, default=1.0, help="Distance threshold for recognition")
    parser.add_argument("--camera", type=int, default=0, help="OpenCV camera index")
    parser.add_argument("--width", type=int, default=0, help="Optional camera width")
    parser.add_argument("--height", type=int, default=0, help="Optional camera height")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for loading labeled embeddings")
    parser.add_argument("--cache", type=Path, default=Path("data/embeddings_cache.pkl"), help="Cache file for embeddings")
    parser.add_argument("--no-cache", action="store_true", help="Force re-processing without using cache")
    return parser.parse_args()


def load_or_create_embeddings(labeled_folder: Path, cache_file: Path, batch_size: int, use_cache: bool = True):
    """Load embeddings from cache or create new ones."""
    def _restore_paths(raw_paths):
        # Rebuild paths relative to current labeled folder to stay cross-platform
        restored = []
        for p in raw_paths:
            # Accept stored as str or Path-like; treat as relative path
            rel = Path(p)
            restored.append(labeled_folder / rel)
        return restored

    # Check if cache exists and is newer than labeled folder
    cache_valid = False
    if use_cache and cache_file.exists():
        cache_mtime = cache_file.stat().st_mtime
        # Check if any file in labeled folder is newer than cache
        labeled_files = list(labeled_folder.glob("*"))
        if labeled_files:
            newest_labeled = max(f.stat().st_mtime for f in labeled_files if f.is_file())
            cache_valid = cache_mtime > newest_labeled
        
    if cache_valid:
        print(f"Loading embeddings from cache: {cache_file}")
        try:
            with open(cache_file, "rb") as f:
                data = pickle.load(f)
            labeled_paths = _restore_paths(data["paths"])
            labeled_embs = data["embeddings"]
            print(f"Loaded {len(labeled_embs)} cached embeddings")
            return labeled_paths, labeled_embs
        except Exception as e:
            print(f"Failed to load cache: {e}")
            print("Re-processing labeled images...")
    
    # Process labeled images
    print("Processing labeled images...")
    labeled_paths, labeled_embs = embed_folder(labeled_folder, batch_size=batch_size)
    
    # Save to cache
    try:
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        rel_paths = [p.relative_to(labeled_folder).as_posix() for p in labeled_paths]
        with open(cache_file, "wb") as f:
            pickle.dump({"paths": rel_paths, "embeddings": labeled_embs}, f)
        print(f"Saved embeddings to cache: {cache_file}")
    except Exception as e:
        print(f"Warning: Failed to save cache: {e}")
    
    return labeled_paths, labeled_embs


def main() -> None:
    args = parse_args()

    # Load or create labeled embeddings with caching
    labeled_paths, labeled_embs = load_or_create_embeddings(
        args.labeled, 
        args.cache, 
        args.batch_size, 
        use_cache=not args.no_cache
    )
    print(f"Ready with {len(labeled_embs)} labeled faces from {args.labeled}")

    # Load networks (auto-select GPU if available)
    mtcnn, vggface2 = get_networks()

    cap = cv2.VideoCapture(args.camera)
    if args.width:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    if args.height:
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)

    if not cap.isOpened():
        raise RuntimeError("Could not open video source")

    print("Press 'q' or 'ESC' to quit (make sure the video window is focused)")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Convert to PIL for the existing pipeline (keeps color correctness)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_im = Image.fromarray(rgb)

        # Detect faces and get bounding boxes
        boxes, _ = mtcnn.detect(pil_im)
        
        # Compute embeddings for detected faces
        embs = embed(pil_im, mtcnn=mtcnn, vggface2=vggface2)

        if boxes is not None and len(boxes) > 0:
            names = get_classes(embs, labeled_paths, labeled_embs, thr=args.threshold)
            names = [n if n else "Unknown" for n in names]

            # Draw boxes and names using the existing PIL-based helper
            annotated = _draw_faces_on_image(pil_im, boxes, names)
            frame = cv2.cvtColor(np.array(annotated), cv2.COLOR_RGB2BGR)

        cv2.imshow("Face Recognition", frame)
        key = cv2.waitKey(30) & 0xFF  # Increased delay for better key detection
        if key == ord("q") or key == ord("Q") or key == 27:  # q, Q, or ESC
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
