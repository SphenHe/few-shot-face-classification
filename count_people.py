"""Count people in one photo or all photos in a folder."""
import argparse
import csv
from pathlib import Path

from few_shot_face_classification import count_people, count_people_in_folder


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="统计照片上的人数")
    parser.add_argument("path", nargs="?", default="data/raw", help="图片路径或图片文件夹，默认 data/raw")
    parser.add_argument("--device", default="cpu", help='检测设备："cpu"、"cuda" 或 "auto"')
    parser.add_argument("--csv", dest="csv_path", help="可选：把统计结果写入 CSV 文件")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    path = Path(args.path)

    if path.is_dir():
        counts = count_people_in_folder(path, device=args.device)
    elif path.is_file():
        counts = {path: count_people(path, device=args.device)}
    else:
        raise FileNotFoundError(f"路径不存在: {path}")

    total_people = sum(counts.values())
    print("照片人数统计:")
    for image_path, people_count in counts.items():
        print(f"  - {image_path}: {people_count} 人")
    print(f"\n合计: {len(counts)} 张照片，检测到 {total_people} 人")

    if args.csv_path:
        csv_path = Path(args.csv_path)
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["path", "people_count"])
            for image_path, people_count in counts.items():
                writer.writerow([str(image_path), people_count])
        print(f"CSV 已保存: {csv_path}")


if __name__ == "__main__":
    main()
