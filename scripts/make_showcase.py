#!/usr/bin/env python3
"""Визуальные материалы для README по результатам инференса.

Берёт outputs/inference/<видео>/detections.csv + summary.json, заново рисует боксы
поверх исходного видео (ищется по имени, по умолчанию в data/videos) и сохраняет в reports/showcase/:
    <видео>.jpg           — самый информативный кадр (больше всего классов и боксов)
    <видео>.gif           — короткий фрагмент с детекциями
    gallery.jpg           — лучшие кадры всех видео в один ряд

    python scripts/run_inference.py            # сначала инференс
    python scripts/make_showcase.py
    python scripts/make_showcase.py --gif-seconds 6 --gif-width 720
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import cv2
import numpy as np
import pandas as pd
import supervision as sv
from PIL import Image

from construction_monitor.config import resolve_path
from construction_monitor.results import Annotator
from construction_monitor.video import find_videos


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Картинки и GIF для README")
    p.add_argument("--results", type=Path, default=Path("outputs/inference"))
    p.add_argument("--out", type=Path, default=Path("reports/showcase"))
    p.add_argument("--gif-seconds", type=float, default=6.0)
    p.add_argument("--gif-width", type=int, default=560)
    p.add_argument("--frame-width", type=int, default=1280)
    return p.parse_args(argv)


def class_names_from(df: pd.DataFrame) -> list[str]:
    """Список классов по class_id из CSV (для отрисовки нужны только встретившиеся)."""
    if df.empty:
        return []
    ids = df.drop_duplicates("class_id").set_index("class_id")["class_name"].to_dict()
    return [ids.get(i, str(i)) for i in range(int(max(ids)) + 1)]


def to_detections(rows: pd.DataFrame) -> sv.Detections:
    if rows is None or rows.empty:
        return sv.Detections.empty()
    return sv.Detections(xyxy=rows[["x1", "y1", "x2", "y2"]].to_numpy(dtype=np.float32),
                         confidence=rows["confidence"].to_numpy(dtype=np.float32),
                         class_id=rows["class_id"].to_numpy(dtype=int))


def best_frame(df: pd.DataFrame) -> int:
    """Кадр с наибольшим числом разных классов, затем боксов, затем средней уверенностью."""
    g = df.groupby("frame").agg(classes=("class_id", "nunique"), boxes=("class_id", "size"),
                                conf=("confidence", "mean"))
    return int(g.sort_values(["classes", "boxes", "conf"], ascending=False).index[0])


def read_frames(path: Path, wanted: set[int]) -> dict[int, np.ndarray]:
    cap = cv2.VideoCapture(str(path))
    out, idx, last = {}, 0, max(wanted)
    while idx <= last:
        if idx in wanted:
            ok, f = cap.read()
            if not ok:
                break
            out[idx] = f
        elif not cap.grab():
            break
        idx += 1
    cap.release()
    return out


def resize_w(img: np.ndarray, width: int) -> np.ndarray:
    h, w = img.shape[:2]
    return cv2.resize(img, (width, round(h * width / w)), interpolation=cv2.INTER_AREA)


def save_gif(frames_bgr: list[np.ndarray], path: Path, fps: float):
    imgs = [Image.fromarray(cv2.cvtColor(f, cv2.COLOR_BGR2RGB)) for f in frames_bgr]
    # общая палитра на весь ролик — без мерцания цветов между кадрами
    pal = imgs[len(imgs) // 2].quantize(colors=128, method=Image.Quantize.MEDIANCUT)
    q = [im.quantize(palette=pal, dither=Image.Dither.NONE) for im in imgs]
    q[0].save(path, save_all=True, append_images=q[1:], duration=round(1000 / fps), loop=0,
              optimize=True)


def make_gallery(images: list[np.ndarray], path: Path, height=450):
    """Лучшие кадры всех видео встык в один ряд, одной высоты."""
    row = [cv2.resize(img, (round(img.shape[1] * height / img.shape[0]), height),
                      interpolation=cv2.INTER_AREA) for img in images]
    cv2.imwrite(str(path), np.hstack(row), [cv2.IMWRITE_JPEG_QUALITY, 88])


def main(argv=None):
    args = parse_args(argv)
    results, out = resolve_path(args.results), resolve_path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    dirs = sorted(d for d in results.iterdir() if (d / "summary.json").exists()) if results.is_dir() else []
    if not dirs:
        raise SystemExit(f"нет результатов в {results} — сначала запустите scripts/run_inference.py")

    gallery = []
    for d in dirs:
        summary = json.loads((d / "summary.json").read_text(encoding="utf-8"))
        df = pd.read_csv(d / "detections.csv")
        src = find_videos([summary["video"]])
        if df.empty or not src:
            print(f"! {d.name}: {'нет детекций' if df.empty else 'не найдено исходное видео'}, пропущено")
            continue

        stride, fps = summary["frame_stride"], summary["fps"]
        total = summary["frames_processed"] * stride
        best = best_frame(df)
        # фрагмент для GIF вокруг лучшего кадра, только обработанные кадры
        n = max(1, round(args.gif_seconds * fps / stride))
        start = max(0, min(best - (n // 2) * stride, total - n * stride))
        clip = [f for f in range(start, start + n * stride, stride) if f < total]
        frames = read_frames(src[0], set(clip) | {best})

        annotate = Annotator(class_names_from(df))
        by_frame = dict(tuple(df.groupby("frame")))
        draw = lambda f: annotate(frames[f], to_detections(by_frame.get(f)))

        best_img = draw(best)
        cv2.imwrite(str(out / f"{d.name}.jpg"), resize_w(best_img, args.frame_width),
                    [cv2.IMWRITE_JPEG_QUALITY, 88])
        save_gif([resize_w(draw(f), args.gif_width) for f in clip if f in frames],
                 out / f"{d.name}.gif", fps / stride)
        gallery.append(best_img)
        print(f"{d.name}: кадр {best}, GIF {len(clip)} кадров")

    if gallery:
        make_gallery(gallery, out / "gallery.jpg")
    for p in sorted(out.iterdir()):
        print(f"  {p.name:<45} {p.stat().st_size / 1024:>8.0f} КБ")


if __name__ == "__main__":
    sys.exit(main())
