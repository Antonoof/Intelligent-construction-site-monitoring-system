#!/usr/bin/env python3
"""Шаг 1. Кеш признаков: DINOv2 (CLS + 144 патч-токена) и RF-DETR (техника по классам).

Большие модели прогоняются по кадрам один раз, дальше голова учится на кеше за минуты.
На RTX 4090 с DINOv2 ViT-g в bf16: 4 692 кадра ≈ 12 минут, кеш ~2.1 ГБ.

    python training/01_build_features.py
    python training/01_build_features.py --limit 50      # быстрая проверка
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import cv2
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from common import Backbone, det_features, detect, load_config, load_detector, p, pick_device


def main(argv=None):
    ap = argparse.ArgumentParser(description="Кеш признаков DINOv2 и RF-DETR")
    ap.add_argument("--config", default=None)
    ap.add_argument("--limit", type=int, default=None, help="только первые N кадров (проверка)")
    args = ap.parse_args(argv)
    cfg = load_config(args.config)

    frames_dir = p(cfg["paths"]["frames_dir"])
    meta = pd.read_csv(frames_dir / "frames.csv")
    if args.limit:
        meta = meta.iloc[np.linspace(0, len(meta) - 1, args.limit).round().astype(int)].reset_index(drop=True)
    out = p(cfg["paths"]["features_dir"])
    out.mkdir(parents=True, exist_ok=True)

    device = pick_device()
    print(f"устройство: {device}; кадров: {len(meta)}")
    t0 = time.perf_counter()
    detector = load_detector(cfg, device)
    class_names = list(detector.class_names)
    backbone = Backbone(cfg, device)
    print(f"RF-DETR + {cfg['backbone']['name']} (dim {backbone.dim}, вход {backbone.hw}, "
          f"сетка {backbone.grid}) загружены за {time.perf_counter() - t0:.0f} с")

    n, g, d = len(meta), backbone.grid[0] * backbone.grid[1], backbone.dim
    cls_mm = np.lib.format.open_memmap(out / "cls.npy", "w+", np.float16, (n, d))
    patch_mm = np.lib.format.open_memmap(out / "patches.npy", "w+", np.float16, (n, g, d))
    det_mm = np.lib.format.open_memmap(out / "det.npy", "w+", np.float32, (n, len(class_names) * 4))
    boxes_f = (out / "detections.jsonl").open("w", encoding="utf-8")

    bs = cfg["backbone"]["batch_size"]
    thr = cfg["detector"]["threshold"]
    for start in tqdm(range(0, n, bs), unit="батч"):
        rows = meta.iloc[start:start + bs]
        rgb = [cv2.cvtColor(cv2.imread(str(frames_dir / "images" / r)), cv2.COLOR_BGR2RGB) for r in rows["image"]]
        cls, patches = backbone(rgb)
        dets = detect(detector, rgb, thr)
        cls_mm[start:start + len(rgb)] = cls
        patch_mm[start:start + len(rgb)] = patches
        for i, (im, dt, name) in enumerate(zip(rgb, dets, rows["image"])):
            h, w = im.shape[:2]
            det_mm[start + i] = det_features(dt, len(class_names), w, h)
            boxes_f.write(json.dumps({"image": name, "boxes": [
                [round(float(v), 1) for v in box] + [int(c), round(float(s), 4)]
                for box, c, s in zip(dt.xyxy, dt.class_id, dt.confidence)]}, ensure_ascii=False) + "\n")
    boxes_f.close()
    for mm in (cls_mm, patch_mm, det_mm):
        mm.flush()

    meta.to_csv(out / "meta.csv", index=False)
    (out / "info.json").write_text(json.dumps({
        "backbone": cfg["backbone"]["name"], "dim": d, "grid": list(backbone.grid), "image_size": list(backbone.hw),
        "class_names": class_names, "det_feats_per_class": 4, "n": n,
        "stages": [s["name"] for s in cfg["stages"]],
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    size = sum(f.stat().st_size for f in out.iterdir()) / 2**30
    print(f"готово: {n} кадров за {(time.perf_counter() - t0) / 60:.1f} мин, кеш {size:.2f} ГБ → {out}")


if __name__ == "__main__":
    sys.exit(main())
