import argparse
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
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Preload labeled embeddings once
    print("Loading labeled embeddings...")
    labeled_paths, labeled_embs = embed_folder(args.labeled, batch_size=args.batch_size)
    print(f"Loaded {len(labeled_embs)} labeled faces from {args.labeled}")

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
