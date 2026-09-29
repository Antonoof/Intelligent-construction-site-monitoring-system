"""Оценка детектора на собственной тестовой выборке по условиям съёмки (ТЗ, раздел 6.2).

Отдельный тестовый набор организаторы не дают, поэтому тестовая выборка собирается скриптом 05 со
стратификацией по классам и условиям: день / сумерки / ночь, зима, дальний план. Здесь считаем AP50,
точность и полноту по каждому классу — в целом и для каждого условия, и пишем:
  docs/metrics_detector_v2.md и docs/metrics_detector_v2.csv

  python training/07_eval_conditions.py --weights weights/rfdetr_large_v2_best.pth
  python training/07_eval_conditions.py --predictions preds.json   # оценка готовых предсказаний без модели
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "training"))
from metrics import evaluate  # noqa: E402

CONDITIONS = ["day", "dusk", "night", "winter", "far", "near"]


def load_test(dataset: Path):
    d = json.loads((dataset / "test" / "_annotations.coco.json").read_text(encoding="utf-8"))
    cats = {c["id"]: c["name"] for c in d["categories"]}
    gt = defaultdict(list)
    for a in d["annotations"]:
        x, y, w, h = a["bbox"]
        gt[a["image_id"]].append((cats[a["category_id"]], x, y, x + w, y + h))
    tags = {}
    with open(dataset / "manifest.csv", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            tags[row["file"]] = row["tags"].split()
    images = {im["id"]: im for im in d["images"]}
    img_tags = {i: tags.get(f"test/{im['file_name']}", []) for i, im in images.items()}
    return images, gt, img_tags, [cats[k] for k in sorted(cats)]


def predict_all(weights: Path, dataset: Path, images: dict, threshold: float = 0.05) -> dict:
    from PIL import Image
    from rfdetr import RFDETR
    sys.path.insert(0, str(ROOT / "backend"))
    from app.methodology import get_methodology
    m = get_methodology()
    model = RFDETR.from_checkpoint(str(weights), trust_checkpoint=True)
    names = list(model.class_names)
    pred = defaultdict(list)
    for i, im in images.items():
        det = model.predict(Image.open(dataset / "test" / im["file_name"]).convert("RGB"), threshold=threshold)
        for k in range(len(det.xyxy)):
            cls = m.normalize_class(names[int(det.class_id[k])]) if int(det.class_id[k]) < len(names) else None
            if cls:
                pred[i].append((cls, *map(float, det.xyxy[k]), float(det.confidence[k])))
    return pred


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--dataset", default=str(ROOT / "datasets" / "oko_v2"))
    ap.add_argument("--weights", default=str(ROOT / "weights" / "rfdetr_large_v2_best.pth"))
    ap.add_argument("--predictions", help="JSON {image_id: [[cls, x1, y1, x2, y2, conf], ...]}")
    ap.add_argument("--out", default=str(ROOT / "docs" / "metrics_detector_v2"))
    a = ap.parse_args(argv)
    ds = Path(a.dataset)
    images, gt, img_tags, classes = load_test(ds)
    if a.predictions:
        raw = json.loads(Path(a.predictions).read_text(encoding="utf-8"))
        pred = {int(k): [tuple(b) for b in v] for k, v in raw.items()}
    else:
        pred = predict_all(Path(a.weights), ds, images)
    table = {"все снимки": evaluate(gt, pred, classes)}
    for cond in CONDITIONS:
        ids = [i for i, t in img_tags.items() if cond in t]
        if ids:
            table[cond] = evaluate({i: gt.get(i, []) for i in ids}, {i: pred.get(i, []) for i in ids}, classes)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out.with_suffix(".csv"), "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["условие", "класс", "AP50", "precision@0.5", "recall@0.5", "объектов"])
        for cond, res in table.items():
            for c in classes:
                r = res[c]
                w.writerow([cond, c, r["ap50"], r["precision"], r["recall"], r["n_gt"]])
    lines = ["# Детектор: AP50 по классам и условиям съёмки", "",
             "| класс | " + " | ".join(table) + " |", "|---|" + "---:|" * len(table)]
    for c in classes + ["__mean__"]:
        cells = []
        for res in table.values():
            v = res[c]["ap50"]
            cells.append("—" if v is None else f"{v:.3f}")
        lines.append(f"| {'среднее' if c == '__mean__' else c} | " + " | ".join(cells) + " |")
    out.with_suffix(".md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    return table


if __name__ == "__main__":
    main()
