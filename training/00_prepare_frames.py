#!/usr/bin/env python3
"""Шаг 0. Нарезка таймлапсов на кадры с метками прогресса и стадии.

Прогресс кадра = его положение в видео: первый кадр 0.0, последний 1.0.
Весь набор идёт в train: валидации и теста нет (см. head.holdout_video в конфиге, если понадобится).

    python training/00_prepare_frames.py
    python training/00_prepare_frames.py --per-video 600
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import cv2
import numpy as np
from tqdm.auto import tqdm

from common import load_config, p, stage_bounds, stage_of

VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv"}


def main(argv=None):
    ap = argparse.ArgumentParser(description="Нарезка видео на обучающие кадры")
    ap.add_argument("--config", default=None)
    ap.add_argument("--per-video", type=int, default=None)
    args = ap.parse_args(argv)
    cfg = load_config(args.config)
    fc = cfg["frames"]
    per_video = args.per_video or fc["per_video"]

    videos = sorted(v for v in p(cfg["paths"]["videos_dir"]).iterdir() if v.suffix.lower() in VIDEO_EXTS)
    if not videos:
        raise SystemExit(f"нет видео в {p(cfg['paths']['videos_dir'])}")
    out = p(cfg["paths"]["frames_dir"])
    (out / "images").mkdir(parents=True, exist_ok=True)

    rows = []
    for vi, video in enumerate(videos, 1):
        cap = cv2.VideoCapture(str(video))
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 25
        # последние пару кадров контейнер часто не отдаёт — берём с запасом
        last = max(n - 3, 1)
        wanted = np.unique(np.linspace(0, last, min(per_video, last + 1)).round().astype(int))
        wanted_set, bounds = set(wanted.tolist()), stage_bounds(cfg, video.name)
        idx = 0
        for target in tqdm(wanted, desc=video.name, unit="кадр"):
            while idx < target:              # пропуск без декодирования
                cap.grab()
                idx += 1
            ok, frame = cap.read()
            idx += 1
            if not ok:
                break
            h, w = frame.shape[:2]
            if w > fc["max_width"]:
                frame = cv2.resize(frame, (fc["max_width"], round(h * fc["max_width"] / w)),
                                   interpolation=cv2.INTER_AREA)
            name = f"v{vi}_{target:06d}.jpg"
            cv2.imwrite(str(out / "images" / name), frame, [cv2.IMWRITE_JPEG_QUALITY, fc["jpeg_quality"]])
            progress = target / last
            rows.append({"image": name, "video": video.name, "video_id": vi, "frame": target,
                         "time_s": round(target / fps, 3), "t": round(progress, 6),
                         "progress": round(progress, 6), "stage": stage_of(progress, bounds)})
        cap.release()
        assert len(wanted_set) >= 2, f"слишком короткое видео {video}"

    with (out / "frames.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    print(f"\nкадров: {len(rows)} из {len(videos)} видео → {out}")
    names = [s["name"] for s in cfg["stages"]]
    counts = np.bincount([r["stage"] for r in rows], minlength=len(names))
    for n, c in zip(names, counts):
        print(f"  {n:<30} {c:>6}")


if __name__ == "__main__":
    sys.exit(main())
