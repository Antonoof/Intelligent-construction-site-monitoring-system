"""Generates submission.csv for AutoDetect2 from a trained YOLO checkpoint."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import yaml

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
COLUMNS = ["image_id", "class_id", "confidence", "x_center", "y_center", "width", "height"]


def collect_images(data_yaml: Path) -> list[Path]:
    with open(data_yaml, "r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}

    roots: list[str] = []
    for key in ("train", "val", "test"):
        entry = data.get(key)
        if isinstance(entry, str):
            roots.append(entry)
        elif isinstance(entry, list):
            roots.extend(entry)

    images: list[Path] = []
    seen: set[str] = set()
    for root in roots:
        base = Path(root)
        if not base.is_dir():
            continue
        for path in sorted(base.rglob("*")):
            if path.suffix.lower() in IMAGE_SUFFIXES and path.name not in seen:
                seen.add(path.name)
                images.append(path)
    return images


def run_predictions(
    weights: Path,
    data_yaml: Path,
    output: Path,
    conf: float = 0.001,
    iou: float = 0.7,
    max_det: int = 300,
    device=None,
    batch: int = 16,
) -> Path:
    from ultralytics import YOLO

    images = collect_images(Path(data_yaml))
    if not images:
        raise SystemExit(f"Не найдено изображений по путям из {data_yaml}")

    model = YOLO(str(weights))
    rows = 0
    print(f"predict {len(images)} images with {Path(weights).name}", flush=True)

    with open(output, "w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(COLUMNS)

        for start in range(0, len(images), batch):
            chunk = images[start : start + batch]
            results = model.predict(
                [str(p) for p in chunk],
                conf=conf,
                iou=iou,
                max_det=max_det,
                device=device,
                verbose=False,
            )
            for image, result in zip(chunk, results):
                boxes = result.boxes
                if boxes is None or not len(boxes):
                    continue
                height, width = result.orig_shape
                xywh = boxes.xywh.cpu().numpy()
                for (xc, yc, bw, bh), score, cls in zip(
                    xywh, boxes.conf.cpu().numpy(), boxes.cls.cpu().numpy()
                ):
                    writer.writerow(
                        [
                            image.name,
                            int(cls),
                            f"{float(score):.6f}",
                            f"{xc / width:.6f}",
                            f"{yc / height:.6f}",
                            f"{bw / width:.6f}",
                            f"{bh / height:.6f}",
                        ]
                    )
                    rows += 1
            print(f"  {min(start + batch, len(images))}/{len(images)}", flush=True)

    print(f"submission: {output} ({rows} боксов)", flush=True)
    return output


def main() -> int:
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="Сборка submission.csv")
    parser.add_argument("--weights", default=str(here / "result" / "best_0.pt"))
    parser.add_argument("--data", default=str(here / "data.yaml"))
    parser.add_argument("--output", default=str(here / "submission.csv"))
    parser.add_argument("--conf", type=float, default=0.001)
    parser.add_argument("--iou", type=float, default=0.7)
    parser.add_argument("--max-det", type=int, default=300)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    run_predictions(
        weights=Path(args.weights),
        data_yaml=Path(args.data),
        output=Path(args.output),
        conf=args.conf,
        iou=args.iou,
        max_det=args.max_det,
        device=args.device,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
