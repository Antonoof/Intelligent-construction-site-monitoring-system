#!/usr/bin/env python3
"""Графики обучения RF-DETR из metrics.csv (лог Lightning) -> PNG.

    python scripts/plot_train_metrics.py
    python scripts/plot_train_metrics.py --csv runs/rfdetr/metrics.csv --out reports/plots

Сохраняет в reports/plots/:
    00_overview.png         — сводка: mAP, P/R/F1, лосс, AP по классам
    01_map.png              — mAP50 / mAP50-95 / mAP75 по эпохам
    02_precision_recall.png — precision / recall / F1 по эпохам
    03_loss.png             — лосс обучения и его составляющие
    04_class_ap_best.png    — AP50-95 по классам на лучшей эпохе
    05_class_ap_heatmap.png — AP50-95 по классам и эпохам
    best_epoch.json         — метрики лучшей эпохи
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

ROOT = Path(__file__).resolve().parents[1]

# палитра (проверена на различимость при дальтонизме); текст — только нейтральными цветами
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
INK_MUTED = "#8a8984"
GRID = "#e6e5e1"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]          # синий, оранжевый, бирюзовый
BLUE_RAMP = ["#eef5fd", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
MARKERS = ["o", "s", "D"]

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "font.family": "DejaVu Sans", "font.size": 10,
    "axes.edgecolor": GRID, "axes.labelcolor": INK_2, "axes.titlecolor": INK,
    "axes.titlesize": 12, "axes.titleweight": "semibold", "axes.titlelocation": "left",
    "axes.titlepad": 12, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "axes.grid.axis": "y", "grid.color": GRID, "grid.linewidth": 0.8,
    "xtick.color": INK_2, "ytick.color": INK_2, "xtick.major.size": 0, "ytick.major.size": 0,
    "legend.frameon": False, "legend.labelcolor": INK_2,
})


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Графики метрик обучения RF-DETR")
    p.add_argument("--csv", type=Path, default=ROOT / "reports" / "rfdetr_large_train_metrics.csv")
    p.add_argument("--out", type=Path, default=ROOT / "reports" / "plots")
    p.add_argument("--weights", type=Path, default=ROOT / "weights" / "rfdetr_large_best_ema.pth",
                   help="чекпоинт — из него берётся полный список классов модели")
    p.add_argument("--dpi", type=int, default=200)
    return p.parse_args(argv)


def load(csv: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Lightning пишет train- и val-метрики разными строками — сводим по эпохам."""
    d = pd.read_csv(csv)
    val = d[d["val/mAP_50_95"].notna()].groupby("epoch").last()
    train = d[d["train/loss"].notna()].groupby("epoch").last()
    return train, val


def checkpoint_classes(weights: Path) -> list[str]:
    """Все классы модели: в CSV есть только те, что встретились в валидации."""
    if not weights.exists():
        return []
    import torch
    ckpt = torch.load(weights, map_location="cpu", weights_only=False)
    return list((ckpt.get("args") or {}).get("class_names") or [])


def class_columns(val: pd.DataFrame) -> dict[str, str]:
    """AP по классам у EMA-модели: именно её веса сохраняются в checkpoint_best_ema.pth,
    и среднее по этим колонкам совпадает с val/mAP_50_95."""
    prefix = "val/ema_AP/" if any(c.startswith("val/ema_AP/") for c in val.columns) else "val/AP/"
    return {c[len(prefix):]: c for c in val.columns if c.startswith(prefix)}


def mark_best(ax, best: int, label=True):
    ax.axvline(best, color=INK_MUTED, lw=1, ls=(0, (3, 3)), zorder=1)
    if label:
        ax.annotate(f"лучшая эпоха {best}", xy=(best, 1), xycoords=("data", "axes fraction"),
                    xytext=(4, -2), textcoords="offset points", va="top", fontsize=8.5, color=INK_2)


def line_panel(ax, val, series, best, title, ylabel=None):
    """Несколько метрик на одной оси + подписи на концах линий."""
    x = val.index.to_numpy()
    ends = []
    for i, (col, name) in enumerate(series):
        y = val[col].to_numpy()
        ax.plot(x, y, color=SERIES[i], lw=2, marker=MARKERS[i], ms=5,
                markeredgecolor=SURFACE, markeredgewidth=1.2, label=name, zorder=3)
        ends.append([y[-1], name])
    # подписи справа, раздвигаем, чтобы не наезжали
    lo, hi = ax.get_ylim()
    gap = (hi - lo) * 0.07
    ends.sort()
    for k in range(1, len(ends)):
        ends[k][0] = max(ends[k][0], ends[k - 1][0] + gap)
    for yv, name in ends:
        ax.annotate(name, xy=(x[-1], yv), xytext=(8, 0), textcoords="offset points",
                    va="center", fontsize=9, color=INK_2)
    mark_best(ax, best)
    ax.set_title(title)
    ax.set_xlabel("эпоха")
    if ylabel:
        ax.set_ylabel(ylabel)
    ax.set_xticks(x)
    ax.set_xlim(x[0] - 0.4, x[-1] + 2.2)
    ax.legend(loc="lower right", ncols=len(series), fontsize=9, handlelength=1.6)


def plot_map(ax, val, best):
    line_panel(ax, val, [("val/mAP_50", "mAP50"), ("val/mAP_50_95", "mAP50-95"),
                         ("val/mAP_75", "mAP75")], best, "mAP на валидации")
    ax.set_ylim(0.6, 0.95)


def plot_prf(ax, val, best):
    line_panel(ax, val, [("val/precision", "Precision"), ("val/recall", "Recall"),
                         ("val/F1", "F1")], best, "Precision / Recall / F1")
    ax.set_ylim(0.65, 1.0)


def plot_total_loss(ax, train, best):
    x = train.index.to_numpy()
    ax.plot(x, train["train/loss"], color=SERIES[0], lw=2, marker="o", ms=5,
            markeredgecolor=SURFACE, markeredgewidth=1.2, zorder=3)
    mark_best(ax, best)
    ax.set_title("Лосс обучения (train/loss)")
    ax.set_xlabel("эпоха")
    ax.set_xticks(x)


def plot_class_bars(ax, val, best, title=None):
    ap = {name: val.loc[best, col] for name, col in class_columns(val).items()}
    names = sorted(ap, key=ap.get)
    vals = [ap[n] for n in names]
    mean = float(np.mean(vals))
    bars = ax.barh(names, vals, height=0.62, color=SERIES[0], edgecolor=SURFACE, linewidth=2)
    for b, v in zip(bars, vals):
        ax.text(v + 0.01, b.get_y() + b.get_height() / 2, f"{v:.2f}", va="center",
                fontsize=9, color=INK)
    ax.axvline(mean, color=INK_MUTED, lw=1, ls=(0, (3, 3)))
    ax.annotate(f"среднее {mean:.2f}", xy=(mean, 1), xycoords=("data", "axes fraction"),
                xytext=(4, 2), textcoords="offset points", fontsize=8.5, color=INK_2)
    ax.set_xlim(0, 1.05)
    ax.grid(axis="y", visible=False)
    ax.grid(axis="x", visible=True)
    ax.tick_params(axis="y", labelcolor=INK)
    ax.set_title(title or f"AP50-95 по классам, эпоха {best}")


def fig_loss(train, best, out, dpi):
    parts = [("train/loss", "Суммарный лосс"), ("train/loss_ce", "Классификация (CE)"),
             ("train/loss_bbox", "Координаты бокса (L1)"), ("train/loss_giou", "Перекрытие бокса (GIoU)")]
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), sharex=True)
    x = train.index.to_numpy()
    for ax, (col, title) in zip(axes.flat, parts):
        y = train[col].to_numpy()
        ax.plot(x, y, color=SERIES[0], lw=2, marker="o", ms=5,
                markeredgecolor=SURFACE, markeredgewidth=1.2, zorder=3)
        ax.annotate(f"{y[-1]:.3f}", xy=(x[-1], y[-1]), xytext=(6, 0), textcoords="offset points",
                    va="center", fontsize=9, color=INK_2)
        drop = (1 - y[-1] / y[0]) * 100
        ax.set_title(f"{title}   ", loc="left")
        ax.text(1, 1.02, f"−{drop:.0f}% за обучение", transform=ax.transAxes, ha="right",
                va="bottom", fontsize=9, color=INK_2)
        mark_best(ax, best, label=False)
        ax.set_xticks(x)
        ax.set_xlim(x[0] - 0.4, x[-1] + 1.4)
    for ax in axes[1]:
        ax.set_xlabel("эпоха")
    fig.suptitle("Лосс обучения по эпохам", x=0.01, ha="left", fontsize=14, fontweight="bold", color=INK)
    fig.text(0.01, 0.935, f"пунктир — лучшая по валидации эпоха {best}; у каждого графика своя шкала",
             fontsize=9.5, color=INK_2)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(out / "03_loss.png", dpi=dpi)
    plt.close(fig)


def fig_heatmap(val, best, out, dpi):
    cols = class_columns(val)
    order = sorted(cols, key=lambda n: -val.loc[best, cols[n]])
    m = np.array([[val.loc[e, cols[n]] for e in val.index] for n in order])
    cmap = LinearSegmentedColormap.from_list("blue", BLUE_RAMP)
    fig, ax = plt.subplots(figsize=(12, 4.8))
    im = ax.imshow(m, cmap=cmap, vmin=0.3, vmax=1.0, aspect="auto")
    for i in range(m.shape[0]):
        for j in range(m.shape[1]):
            v = m[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=8.5,
                    color="#ffffff" if v > 0.72 else INK)
    ax.set_xticks(range(len(val.index)), [str(e) for e in val.index])
    ax.set_yticks(range(len(order)), order)
    ax.tick_params(axis="y", labelcolor=INK)
    ax.set_xticks(np.arange(-0.5, m.shape[1]), minor=True)
    ax.set_yticks(np.arange(-0.5, m.shape[0]), minor=True)
    ax.grid(which="minor", color=SURFACE, linewidth=2)
    ax.grid(which="major", visible=False)
    ax.tick_params(which="minor", length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    b = list(val.index).index(best)
    ax.add_patch(plt.Rectangle((b - 0.5, -0.5), 1, m.shape[0], fill=False, ec="#eb6834", lw=3,
                               zorder=10, clip_on=False))
    ax.set_xlabel("эпоха")
    ax.set_title(f"AP50-95 по классам и эпохам (рамка — лучшая эпоха {best})")
    cb = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    cb.outline.set_visible(False)
    cb.ax.tick_params(colors=INK_2, length=0)
    fig.tight_layout()
    fig.savefig(out / "05_class_ap_heatmap.png", dpi=dpi)
    plt.close(fig)


def fig_overview(train, val, best, out, dpi, missing):
    r = val.loc[best]
    fig = plt.figure(figsize=(15, 10.5))
    gs = fig.add_gridspec(3, 2, height_ratios=[0.5, 3, 3], hspace=0.45, wspace=0.18)

    # строка с ключевыми цифрами
    kpis = [("mAP50", r["val/mAP_50"]), ("mAP50-95", r["val/mAP_50_95"]),
            ("Precision", r["val/precision"]), ("Recall", r["val/recall"]), ("F1", r["val/F1"])]
    kax = fig.add_subplot(gs[0, :])
    kax.axis("off")
    for i, (name, v) in enumerate(kpis):
        xx = i / len(kpis)
        kax.text(xx, 0.95, name, fontsize=10, color=INK_2, va="top", transform=kax.transAxes)
        kax.text(xx, 0.45, f"{v:.3f}", fontsize=22, fontweight="bold", color=INK,
                 va="top", transform=kax.transAxes)

    plot_map(fig.add_subplot(gs[1, 0]), val, best)
    plot_prf(fig.add_subplot(gs[1, 1]), val, best)
    plot_total_loss(fig.add_subplot(gs[2, 0]), train, best)
    plot_class_bars(fig.add_subplot(gs[2, 1]), val, best)

    fig.suptitle("RF-DETR Large · детектор строительной техники", x=0.07, y=0.985, ha="left",
                 fontsize=16, fontweight="bold", color=INK)
    sub = f"валидация, лучшая эпоха {best} из {int(val.index.max())}, разрешение 736 px"
    if missing:
        sub += f" · нет в валидации: {', '.join(missing)}"
    fig.text(0.07, 0.945, sub, fontsize=10, color=INK_2)
    fig.subplots_adjust(left=0.07, right=0.97, top=0.9, bottom=0.06)
    fig.savefig(out / "00_overview.png", dpi=dpi)
    plt.close(fig)


def single(draw, name, out, dpi, size=(10, 5.5)):
    fig, ax = plt.subplots(figsize=size)
    draw(ax)
    fig.tight_layout()
    fig.savefig(out / name, dpi=dpi)
    plt.close(fig)


def main(argv=None):
    args = parse_args(argv)
    if not args.csv.exists():
        raise SystemExit(f"не найден {args.csv}")
    args.out.mkdir(parents=True, exist_ok=True)
    train, val = load(args.csv)
    best = int(val["val/mAP_50_95"].idxmax())

    all_classes = checkpoint_classes(args.weights)
    missing = [c for c in all_classes if c not in class_columns(val)]

    fig_overview(train, val, best, args.out, args.dpi, missing)
    single(lambda ax: plot_map(ax, val, best), "01_map.png", args.out, args.dpi)
    single(lambda ax: plot_prf(ax, val, best), "02_precision_recall.png", args.out, args.dpi)
    fig_loss(train, best, args.out, args.dpi)
    single(lambda ax: plot_class_bars(ax, val, best), "04_class_ap_best.png", args.out, args.dpi,
           size=(9, 5))
    fig_heatmap(val, best, args.out, args.dpi)

    r = val.loc[best]
    summary = {
        "best_epoch": best, "selected_by": "val/mAP_50_95",
        "metrics": {k.split("/", 1)[1]: round(float(r[k]), 4) for k in
                    ("val/mAP_50", "val/mAP_50_95", "val/mAP_75", "val/mAR",
                     "val/precision", "val/recall", "val/F1")},
        "class_ap_50_95": {n: round(float(r[c]), 4) for n, c in class_columns(val).items()},
        "classes_not_in_val": missing,
    }
    (args.out / "best_epoch.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2),
                                              encoding="utf-8")
    for p in sorted(args.out.iterdir()):
        print(p)


if __name__ == "__main__":
    sys.exit(main())
