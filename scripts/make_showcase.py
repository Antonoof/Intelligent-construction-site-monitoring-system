#!/usr/bin/env python3
"""Визуальные материалы для README по результатам инференса.

Берёт outputs/inference/<видео>/detections.csv + summary.json, заново рисует боксы
поверх исходного видео (ищется по имени, по умолчанию в data/videos) и сохраняет в reports/showcase/:
    <видео>.jpg           — самый информативный кадр (больше всего классов и боксов)
    <видео>.gif           — короткий фрагмент с детекциями
    <видео>_timeline.png  — когда какая техника была в кадре
    gallery.jpg           — лучшие кадры всех видео одной картинкой

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
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import supervision as sv
from matplotlib import font_manager
from PIL import Image, ImageDraw, ImageFont

from construction_monitor.config import resolve_path
from construction_monitor.results import Annotator, class_color
from construction_monitor.video import find_videos

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
GRID = "#e6e5e1"


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Картинки, GIF и таймлайны для README")
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


def presence_segments(frames: list[int], present: set[int]) -> list[tuple[int, int]]:
    """Отрезки подряд идущих обработанных кадров, где класс виден: [(первый, последний)]."""
    segs, start, prev = [], None, None
    for f in frames:
        if f in present:
            if start is None:
                start = f
            prev = f
        elif start is not None:
            segs.append((start, prev))
            start = None
    if start is not None:
        segs.append((start, prev))
    return segs


def plot_timeline(df: pd.DataFrame, summary: dict, path: Path):
    fps, stride = summary["fps"], summary["frame_stride"]
    step = stride / fps
    processed = list(range(0, summary["frames_processed"] * stride, stride))
    duration = len(processed) * step
    classes = sorted(summary["classes"], key=lambda n: -summary["classes"][n]["presence_ratio"])

    fig, ax = plt.subplots(figsize=(10, 1.0 + 0.55 * max(len(classes), 1)), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    for lane, name in enumerate(classes):
        sub = df[df["class_name"] == name]
        present = set(sub["frame"])
        multi = set(sub.groupby("frame").size().loc[lambda s: s > 1].index)
        y = len(classes) - 1 - lane
        segs = [(a / fps, (b - a) / fps + step) for a, b in presence_segments(processed, present)]
        ax.broken_barh(segs, (y - 0.3, 0.6), facecolors=class_color(name), edgecolor=SURFACE,
                       linewidth=1)
        # несколько единиц одновременно — светлая полоса внутри бара
        msegs = [(a / fps, (b - a) / fps + step) for a, b in presence_segments(processed, multi)]
        if msegs:
            ax.broken_barh(msegs, (y - 0.06, 0.12), facecolors=SURFACE, alpha=0.85)
        c = summary["classes"][name]
        note = f"{c['presence_ratio']:.0%} времени"
        if c["max_simultaneous"] > 1:
            note += f" · до {c['max_simultaneous']} шт."
        ax.text(duration * 1.01, y, note, va="center", fontsize=9, color=INK_2)

    ax.set_yticks(range(len(classes)), list(reversed(classes)))
    ax.tick_params(axis="y", length=0, labelcolor=INK, labelsize=10)
    ax.tick_params(axis="x", length=0, colors=INK_2)
    ax.set_xlim(0, duration)
    ax.set_ylim(-0.6, len(classes) - 0.4)
    ax.set_xlabel("время, с", color=INK_2)
    ax.grid(axis="x", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.set_title(f"Техника в кадре · {summary['video']}", loc="left", fontsize=12,
                 fontweight="semibold", color=INK, pad=24)
    ax.text(0, 1.02, "бар — техника видна в кадре; светлая полоса внутри — несколько единиц одновременно",
            transform=ax.transAxes, fontsize=8.5, color=INK_2, va="bottom")
    fig.tight_layout()
    fig.subplots_adjust(right=0.8)
    fig.savefig(path, dpi=160, facecolor=SURFACE)
    plt.close(fig)


def font(size: int, bold=False) -> ImageFont.FreeTypeFont:
    name = "DejaVu Sans:bold" if bold else "DejaVu Sans"
    return ImageFont.truetype(font_manager.findfont(name), size)


def make_gallery(items: list[tuple[np.ndarray, dict]], path: Path, tile_w=800, cols=3):
    """Сетка лучших кадров, под каждым — название видео и найденная техника с цветными метками."""
    pad, cap_h = 16, 78
    tiles = [(resize_w(img, tile_w), s) for img, s in items]
    tile_h = max(t.shape[0] for t, _ in tiles)
    cols = min(cols, len(tiles))
    rows = (len(tiles) + cols - 1) // cols
    W = cols * tile_w + (cols + 1) * pad
    H = rows * (tile_h + cap_h) + (rows + 1) * pad
    canvas = Image.new("RGB", (W, H), SURFACE)
    draw = ImageDraw.Draw(canvas)
    f_title, f_item = font(20, bold=True), font(17)
    for i, (img, s) in enumerate(tiles):
        r, c = divmod(i, cols)
        x = pad + c * (tile_w + pad)
        y = pad + r * (tile_h + cap_h + pad)
        canvas.paste(Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)), (x, y))
        ty = y + img.shape[0] + 10
        draw.text((x, ty), Path(s["video"]).stem, font=f_title, fill=INK)
        cx = x
        for name, v in sorted(s["classes"].items(), key=lambda kv: -kv[1]["presence_ratio"]):
            label = f"{name} {v['presence_ratio']:.0%}"
            draw.rounded_rectangle((cx, ty + 36, cx + 14, ty + 50), radius=3, fill=class_color(name))
            draw.text((cx + 20, ty + 32), label, font=f_item, fill=INK_2)
            cx += 20 + draw.textlength(label, font=f_item) + 22
    canvas.save(path, quality=88, optimize=True)


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
        plot_timeline(df, summary, out / f"{d.name}_timeline.png")
        gallery.append((best_img, summary))
        print(f"{d.name}: кадр {best}, GIF {len(clip)} кадров")

    if gallery:
        make_gallery(gallery, out / "gallery.jpg")
    for p in sorted(out.iterdir()):
        print(f"  {p.name:<45} {p.stat().st_size / 1024:>8.0f} КБ")


if __name__ == "__main__":
    sys.exit(main())
