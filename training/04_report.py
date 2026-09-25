#!/usr/bin/env python3
"""Шаг 4. Итоговый результат для README: графики, итоговые панели примеров (модель + проверка), «до/после» детекции.

Берёт последний запуск из training/runs/, прогон предсказаний и ручную проверку кадров docs/claude_review.json.
Результат — docs/assets/.

    python training/04_report.py
    python training/04_report.py --predictions predictions/20260925_212142
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from PIL import Image, ImageDraw, ImageFont

from common import CLASS_RU, ROOT, load_config, monotone, p, plan_status
from render import render_panel, signed

REPO = ROOT
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e0"
BLUE, PLAN = "#2a78d6", "#8a8983"
GOOD, BAD, WARN, LIGHT = "#1a7f45", "#c23b33", "#a86b00", "#9ec5f4"
# примеры для README по порядку: exampleN.jpg, exampleN_heatmap.jpg, exampleN_review.jpg
SHOW = ["example1", "example2", "example3", "example4", "example5"]
HEATMAPS = {"example2", "example3"}   # карты внимания, которые показывает README

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.spines.top": False,
    "axes.spines.right": False, "font.size": 11, "axes.titlesize": 13, "axes.titleweight": "bold",
    "axes.titlelocation": "left", "text.color": INK,
})


def training_curves(h: pd.DataFrame, out: Path):
    fig, axes = plt.subplots(1, 3, figsize=(15, 3.8))
    for ax, (col, title, fmt) in zip(axes, [("loss", "Лосс", "{:.3f}"), ("mae_pp", "Ошибка готовности, п.п.", "{:.1f}"),
                                            ("stage_acc", "Точность стадии", "{:.1%}")]):
        ax.plot(h["epoch"], h[col], color=BLUE, lw=2)
        last = h[col].iloc[-1]
        ax.plot(h["epoch"].iloc[-1], last, "o", ms=7, color=BLUE, mec=SURFACE, mew=2)
        ax.annotate(fmt.format(last), (h["epoch"].iloc[-1], last), xytext=(-8, 10), textcoords="offset points",
                    ha="right", color=INK, fontweight="bold")
        ax.set_title(title)
        ax.set_xlabel("эпоха")
    axes[2].yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def smooth(pred: pd.DataFrame) -> pd.DataFrame:
    """Итоговая обработка: прогнозы каждого видео по времени → неубывающая кривая готовности."""
    parts = []
    for _, d in pred.groupby("video"):
        d = d.sort_values("t").copy()
        d["final_progress"] = monotone(d["pred_progress"].values)
        parts.append(d)
    out = pd.concat(parts)
    out["final_err_pp"] = (out["final_progress"] - out["progress"]).abs() * 100
    return out


def readiness_final(pred: pd.DataFrame, marks: list[dict], out: Path):
    """Сверху — готовность по всему видео (план, прогноз по кадрам, итог после обработки), снизу — отклонение от плана."""
    videos = sorted(pred["video"].unique())
    fig, axes = plt.subplots(2, len(videos), figsize=(3.7 * len(videos), 6.4), sharey="row",
                             gridspec_kw={"height_ratios": [2.2, 1]})
    dev_max = max(2.0, np.ceil((pred["final_progress"] - pred["progress"]).abs().max() * 100))
    for k, v in enumerate(videos):
        d = pred[pred["video"] == v]
        top, bot = axes[0, k], axes[1, k]
        top.plot(d["t"], d["progress"], color=PLAN, lw=1.6, ls=(0, (5, 4)), label="план")
        top.plot(d["t"], d["pred_progress"], color=LIGHT, lw=0.9, label="прогноз по кадрам")
        top.plot(d["t"], d["final_progress"], color=BLUE, lw=2.2, label="итог после обработки")
        for m in (m for m in marks if m["video"] == v):
            i = (d["t"] - m["t"]).abs().idxmin()
            top.plot(d.loc[i, "t"], d.loc[i, "final_progress"], "o", ms=15, mew=2, mec=SURFACE, color=INK, zorder=5)
            top.text(d.loc[i, "t"], d.loc[i, "final_progress"], str(m["n"]), color=SURFACE, fontsize=9,
                     fontweight="bold", ha="center", va="center", zorder=6)
        top.set_title(f"{v.removesuffix('.mp4')}\n", fontsize=13)
        top.text(0, 1.02, f"ошибка {d['final_err_pp'].mean():.1f} п.п. · было {d['abs_err_pp'].mean():.1f}",
                 transform=top.transAxes, color=INK2, fontsize=10.5)
        top.set_ylim(0, 1.02)
        dev = (d["final_progress"] - d["progress"]).values * 100
        bot.axhline(0, color=INK2, lw=0.9)
        bot.fill_between(d["t"], 0, dev, where=dev >= 0, color=BLUE, alpha=0.25, lw=0, interpolate=True)
        bot.fill_between(d["t"], 0, dev, where=dev < 0, color=BAD, alpha=0.25, lw=0, interpolate=True)
        bot.plot(d["t"], dev, color=INK, lw=1.1)
        bot.set_ylim(-dev_max, dev_max)
        bot.set_xlabel("время съёмки")
        for ax in (top, bot):
            ax.set_xlim(0, 1)
            ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
        top.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
    axes[0, 0].set_ylabel("готовность")
    axes[1, 0].set_ylabel("от плана, п.п.")
    axes[1, 0].text(0.02, 0.93, "опережение", transform=axes[1, 0].transAxes, color=BLUE, fontsize=9.5, va="top")
    axes[1, 0].text(0.02, 0.07, "отставание", transform=axes[1, 0].transAxes, color=BAD, fontsize=9.5)
    handles = axes[0, 0].get_legend_handles_labels()[0] + [
        plt.Line2D([], [], marker="o", ls="", color=INK, mec=SURFACE, ms=10, label="пример из README (номер)")]
    fig.legend(handles=handles, loc="upper center", ncol=4, frameon=False, fontsize=10.5, bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(out, dpi=150)
    plt.close(fig)


def stage_confusion(pred: pd.DataFrame, stages: list[dict], out: Path):
    n = len(stages)
    m = np.zeros((n, n))
    for a, b in zip(pred["stage"], pred["pred_stage"]):
        m[a, b] += 1
    share = m / m.sum(1, keepdims=True)
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list(
        "blue", ["#f4f8fd", "#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"])
    fig, ax = plt.subplots(figsize=(8.6, 5.6))
    ax.imshow(share, cmap=cmap, vmin=0, vmax=1)
    ax.grid(False)
    for i in range(n):
        for j in range(n):
            if m[i, j]:
                ax.text(j, i, f"{share[i, j]:.1%}\n{int(m[i, j])}", ha="center", va="center", fontsize=10,
                        color=SURFACE if share[i, j] > 0.5 else INK)
    names = [f"{s['id']} {s['name']}" for s in stages]
    ax.set_xticks(range(n), [s["id"] for s in stages])
    ax.set_yticks(range(n), names)
    ax.set_xlabel("предсказанная стадия")
    ax.set_title("Стадия: разметка → прогноз", pad=12)
    fig.savefig(out, dpi=150, bbox_inches="tight", pad_inches=0.25)
    plt.close(fig)


RU2CLASS = {ru: en for en, ru in CLASS_RU.items()}


def final_result(pj: dict, rv: dict, pred: pd.DataFrame, cfg: dict) -> tuple[dict, dict]:
    """Итог по примеру: рамки модели с поправками проверки, готовность со сглаженной кривой видео, вердикт."""
    r = dict(pj)
    find = lambda b, key: next((x for x in rv.get(key, []) if np.allclose(b[:4], x[:4], atol=40)), None)
    eq, other = [], []
    for b in pj["equipment_boxes"]:
        if find(b, "remove"):
            continue
        fix = find(b, "relabel")
        (other if fix else eq).append([*b[:4], fix[5], None] if fix else b)
    other += [b for b in pj["other_boxes"] if not find(b, "remove")]
    for a in rv.get("add", []):                      # объекты, которые отметила проверка (без уверенности модели)
        cls = RU2CLASS.get(a[4])
        (eq if cls else other).append([*a[:4], cls or a[4], None])
    counts, conf, other_counts = {}, {}, {}
    for *_, cls, s in eq:
        counts[cls] = counts.get(cls, 0) + 1
        conf[cls] = max(conf.get(cls) or 0, s) if s is not None else conf.get(cls)
    for *_, lab, _ in other:
        other_counts[lab] = other_counts.get(lab, 0) + 1

    d = pred[pred["video"] == pj["video"]].sort_values("t")
    final = float(d.loc[d["image"] == pj["train_image"], "final_progress"].iloc[0])
    delta = final - pj["expected"]
    status, status_ru = plan_status(cfg, delta)
    si = rv.get("stage", pj["stage_idx"])
    step = max(1, len(d) // 120)
    parts = [f"Стадия «{pj['stage_names'][si]}». Готовность ≈ {final:.0%} при плане {pj['expected']:.0%}: "
             f"{status_ru} ({signed(delta * 100)} п.п., ≈ {signed(delta * pj['plan_days'], 0)} дн.)."]
    if counts:
        parts.append("Техника: " + ", ".join(f"{CLASS_RU.get(k, k)} ×{v}" for k, v in counts.items()) + ".")
    if other_counts:
        parts.append("Объекты: " + ", ".join(f"{k} ×{v}" for k, v in other_counts.items()) + ".")
    r.update({"equipment_boxes": eq, "other_boxes": other, "equipment_counts": counts, "equipment_conf": conf,
              "other_counts": other_counts, "pred_progress": final, "delta_pp": delta * 100,
              "delta_days": delta * pj["plan_days"], "status": status, "status_ru": status_ru, "stage_idx": si,
              "stage_name": pj["stage_names"][si], "stage_prob": pj["stage_probs"][si], "summary": " ".join(parts),
              "trajectory": {"t": d["t"].iloc[::step].tolist(), "pred": d["final_progress"].iloc[::step].tolist()}})
    vlm = pj.get("vlm_stage")
    verdict = (f"стадия {si + 1} подтверждена ✓" if si == pj["stage_idx"] else f"стадия исправлена на {si + 1}") + \
        ("" if vlm is None else f" · Qwen3-VL: стадия {vlm + 1} " + ("✓" if vlm == si else "✗"))
    return r, {"verdict": verdict, "comment": rv.get("comment", "")}


def resize(src: Path, dst: Path, width: int):
    im = Image.open(src).convert("RGB")
    if im.width > width:
        im = im.resize((width, round(im.height * width / im.width)), Image.LANCZOS)
    im.save(dst, quality=86, optimize=True, progressive=True)


def _font(size: int, bold: bool = False):
    path = font_manager.findfont(font_manager.FontProperties(family="DejaVu Sans", weight="bold" if bold else "normal"))
    return ImageFont.truetype(path, size)


def _dashed_rect(d: ImageDraw.ImageDraw, box, color, width=3, dash=12):
    x1, y1, x2, y2 = box
    for a, b in (((x1, y1), (x2, y1)), ((x2, y1), (x2, y2)), ((x2, y2), (x1, y2)), ((x1, y2), (x1, y1))):
        length = max(abs(b[0] - a[0]), abs(b[1] - a[1]))
        for s in range(0, int(length), dash * 2):
            e = min(s + dash, length)
            d.line([(a[0] + (b[0] - a[0]) * s / length, a[1] + (b[1] - a[1]) * s / length),
                    (a[0] + (b[0] - a[0]) * e / length, a[1] + (b[1] - a[1]) * e / length)], fill=color, width=width)


def _labels(d: ImageDraw.ImageDraw, items, font, width: int, height: int):
    """Подписи рамок без наложений: над рамкой, внутри сверху, внутри снизу или под ней — первое свободное место."""
    placed, th = [], 24
    free = lambda r: all(r[2] <= q[0] or q[2] <= r[0] or r[3] <= q[1] or q[3] <= r[1] for q in placed)
    for (x1, y1, x2, y2), text, color in items:
        tw = d.textlength(text, font=font) + 12
        x = min(max(0, x1), width - tw)
        spots = [y1 - th, y1, y2 - th, y2] + [y1 + k * th for k in range(1, 20)]
        y = next((y for y in spots if 0 <= y <= height - th and free((x, y, x + tw, y + th))), max(0, y1 - th))
        placed.append((x, y, x + tw, y + th))
        d.rectangle([x, y, x + tw, y + th], fill=color)
        d.text((x + 6, y + 3), text, fill="#ffffff", font=font)


def review_image(frame: Path, pj: dict, rv: dict, out: Path, tiles: int = 1):
    """Одна картинка: слева рамки моделей, справа — после проверки (подтверждено, исправлено, добавлено)."""
    base = Image.open(frame).convert("RGB")
    w, h = base.size
    sx, sy = w / 1280, h / 720
    sc = lambda b: [b[0] * sx, b[1] * sy, b[2] * sx, b[3] * sy]
    small = _font(15, bold=True)
    find = lambda b, key: next((r for r in rv.get(key, []) if np.allclose(b[:4], r[:4], atol=40)), None)

    left, right = base.copy(), base.copy()
    dl, dr = ImageDraw.Draw(left), ImageDraw.Draw(right)
    lab_l, lab_r = [], []
    # рамки, снятые проверкой с итоговой панели (review_removed_*), слева показываются как вывод модели
    model_eq = pj.get("equipment_boxes", []) + pj.get("review_removed_equipment", [])
    model_other = pj.get("other_boxes", []) + pj.get("review_removed_other", [])
    boxes = [(b, CLASS_RU.get(b[4], b[4]), False) for b in model_eq] + [(b, b[4], True) for b in model_other]
    for b, name, dashed in boxes:
        wrong, relabel = find(b, "remove"), find(b, "relabel")
        color = BAD if wrong else WARN if relabel else "#3a3a38" if dashed else BLUE
        if dashed:
            _dashed_rect(dl, sc(b[:4]), "#ffffff" if color == "#3a3a38" else color)
        else:
            dl.rectangle(sc(b[:4]), outline=color, width=3)
        lab_l.append((sc(b[:4]), f"{name} {b[5]:.2f}", color))
        if wrong:
            _dashed_rect(dr, sc(b[:4]), BAD)
            lab_r.append((sc(b[:4]), f"✗ {name} — {wrong[5]}", BAD))
        elif relabel:
            dr.rectangle(sc(b[:4]), outline=WARN, width=3)
            lab_r.append((sc(b[:4]), f"~ {relabel[5]}", WARN))
        else:
            dr.rectangle(sc(b[:4]), outline=GOOD, width=3)
            lab_r.append((sc(b[:4]), f"✓ {name}", GOOD))
    for a in rv.get("add", []):
        color = GOOD if a[5] == "confirmed" else BLUE
        dr.rectangle(sc(a[:4]), outline=color, width=3)
        lab_r.append((sc(a[:4]), ("✓ " if a[5] == "confirmed" else "+ ") + a[4], color))
    _labels(dl, lab_l, small, w, h)
    _labels(dr, lab_r, small, w, h)

    head, gap = 56, 16
    canvas = Image.new("RGB", (2 * w + gap, h + head), SURFACE)
    canvas.paste(left, (0, head))
    canvas.paste(right, (w + gap, head))
    d = ImageDraw.Draw(canvas)
    big = _font(24, bold=True)
    d.text((4, 14), "Детекция моделей" + (f" · RF-DETR по фрагментам {tiles}×{tiles}" if tiles > 1 else ""),
           fill=INK, font=big)
    d.text((w + gap + 4, 14), "После проверки Claude", fill=INK, font=big)
    canvas = canvas.resize((1800, round(canvas.height * 1800 / canvas.width)), Image.LANCZOS)
    canvas.save(out, quality=88, optimize=True, progressive=True)


def main(argv=None):
    ap = argparse.ArgumentParser(description="Иллюстрации для README")
    ap.add_argument("--config", default=None)
    ap.add_argument("--run", default=None, help="папка запуска в training/runs (по умолчанию — последний)")
    ap.add_argument("--predictions", default=None, help="папка прогона в predictions (по умолчанию — последняя)")
    ap.add_argument("--out", default=str(REPO / "docs" / "assets"))
    args = ap.parse_args(argv)
    cfg = load_config(args.config)

    runs = p(cfg["paths"]["runs_dir"])
    run = runs / (args.run or (runs / "latest.txt").read_text(encoding="utf-8").strip())
    preds = p(args.predictions) if args.predictions else sorted(p(cfg["paths"]["predictions_dir"]).iterdir())[-1]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    hist = pd.read_csv(run / "history.csv")
    pred = pd.read_csv(run / "train_predictions.csv")
    training_curves(hist, out / "training_curves.png")
    stage_confusion(pred, cfg["stages"], out / "stage_confusion.png")
    pred = smooth(pred)

    review = json.loads((REPO / "docs" / "claude_review.json").read_text(encoding="utf-8"))["frames"]
    marks = []
    for n, name in enumerate(SHOW, 1):
        if name in HEATMAPS:
            resize(preds / "heatmaps" / f"{name}.jpg", out / f"example{n}_heatmap.jpg", 1100)
        pj = json.loads((preds / "json" / f"{name}.json").read_text(encoding="utf-8"))
        rv = review.get(name, {})
        fr, verdict = final_result(pj, rv, pred, cfg)
        render_panel(fr, Image.open(preds / "frames" / f"{name}.jpg").convert("RGB"), out / "_panel.png", review=verdict)
        resize(out / "_panel.png", out / f"example{n}.jpg", 1800)
        (out / "_panel.png").unlink()
        tiles = int((cfg["videos"].get(pj["video"]) or {}).get("tiles", cfg["detector"].get("tiles", 1)))
        review_image(preds / "frames" / f"{name}.jpg", pj, rv, out / f"example{n}_review.jpg", tiles)
        marks.append({"video": pj["video"], "t": pj["t"], "n": n})
    readiness_final(pred, marks, out / "readiness_final.png")

    by_video = pred.groupby("video").apply(lambda d: pd.Series({
        "frames": len(d), "mae_pp": d["abs_err_pp"].mean(), "mae_pp_final": d["final_err_pp"].mean(),
        "stage_acc": (d["pred_stage"] == d["stage"]).mean()}), include_groups=False)
    by_video.loc["все"] = [len(pred), pred["abs_err_pp"].mean(), pred["final_err_pp"].mean(),
                          (pred["pred_stage"] == pred["stage"]).mean()]
    by_video.round(4).to_csv(REPO / "docs" / "metrics_by_video.csv")
    print(by_video.round(3).to_string())
    print(f"готово: {out}")


if __name__ == "__main__":
    sys.exit(main())
