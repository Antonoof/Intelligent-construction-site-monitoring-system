"""Отрисовка результатов: панель (сверху легенда, слева фото, справа текст) и отдельная карта внимания."""
from __future__ import annotations

from datetime import date
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from common import CLASS_COLORS, CLASS_RU

W, H = 3500, 1450
PHOTO = (20, 124, 2322, 1306)                 # x, y, w, h — 16:9
SIDE_X, SIDE_W = 2382, 1098
BG, INK, INK2, INK3, LINE, SUNK = "#f4f5f2", "#1b1f1c", "#4d544f", "#7a817c", "#d6dad5", "#e4e7e2"
STATUS = {"ahead": ("#1d5fae", "ОПЕРЕЖАЕМ"), "on_track": ("#1a7f45", "УСПЕВАЕМ"),
          "risk": ("#a86b00", "РИСК ОТСТАВАНИЯ"), "late": ("#c23b33", "ОТСТАЁМ"), "none": ("#7a817c", "НЕТ ПЛАНА")}
OTHER = "#f4f5f2"
MINUS = "−"


def signed(x: float, digits: int = 1) -> str:
    """+3.4 / −3.4 с типографским минусом."""
    return ("+" if x >= 0 else MINUS) + f"{abs(x):.{digits}f}"


def ru_date(iso: str | None) -> str:
    return date.fromisoformat(iso[:10]).strftime("%d.%m.%Y") if iso else ""


def _font_path(bold=False):
    from matplotlib import font_manager
    return font_manager.findfont(font_manager.FontProperties(family="DejaVu Sans", weight="bold" if bold else "normal"))


_FONTS: dict = {}


def font(size, bold=False):
    key = (size, bold)
    if key not in _FONTS:
        _FONTS[key] = ImageFont.truetype(_font_path(bold), size)
    return _FONTS[key]


def text_on(hex_color):
    r, g, b = (int(hex_color[i:i + 2], 16) / 255 for i in (1, 3, 5))
    return "#0b0b0b" if 0.2126 * r ** 2.2 + 0.7152 * g ** 2.2 + 0.0722 * b ** 2.2 > 0.3 else "#ffffff"


def wrap(draw, text, fnt, width):
    lines, line = [], ""
    for word in text.split():
        cand = f"{line} {word}".strip()
        if draw.textlength(cand, font=fnt) <= width:
            line = cand
        else:
            if line:
                lines.append(line)
            line = word
    if line:
        lines.append(line)
    return lines


def clip_lines(lines, n):
    if len(lines) > n:
        lines = lines[:max(n, 0)]
        if lines:
            lines[-1] = lines[-1].rstrip(".,;:»") + "…"
    return lines


def dashed_rect(draw, box, color, width=3, dash=14):
    x1, y1, x2, y2 = box
    for (ax, ay, bx, by) in ((x1, y1, x2, y1), (x2, y1, x2, y2), (x2, y2, x1, y2), (x1, y2, x1, y1)):
        length = max(abs(bx - ax), abs(by - ay))
        steps = max(int(length // dash), 1)
        for i in range(0, steps, 2):
            s0, s1 = i / steps, min((i + 1) / steps, 1)
            draw.line((ax + (bx - ax) * s0, ay + (by - ay) * s0, ax + (bx - ax) * s1, ay + (by - ay) * s1),
                      fill=color, width=width)


def place_label(box, tw, th, placed, bounds):
    """Место для подписи рамки: над ней, внутри сверху, под ней, внутри снизу — первое свободное."""
    x1, y1, x2, y2 = box
    bx1, by1, bx2, by2 = bounds
    lx = min(max(x1, bx1), bx2 - tw)
    free = lambda a: not any(a[0] < b[2] and b[0] < a[2] and a[1] < b[3] and b[1] < a[3] for b in placed)
    for cx, cy in [(lx, y1 - th), (lx, y1), (lx, y2), (lx, y2 - th)]:
        if cy >= by1 and cy + th <= by2 and free((cx, cy, cx + tw, cy + th)):
            return cx, cy
    cy = max(by1, y1 - th)
    while cy + th <= by2 and not free((lx, cy, lx + tw, cy + th)):
        cy += th
    return lx, min(cy, by2 - th)


def section(draw, y, title):
    draw.text((SIDE_X, y), title, font=font(22, True), fill=INK3)
    return y + 36


def legend(d, counts, y=76, x=20):
    f, fb = font(23), font(23, True)
    for cls, color in CLASS_COLORS.items():
        n = counts.get(cls, 0)
        label = f"{CLASS_RU[cls]} {n}" if n else CLASS_RU[cls]
        if n:
            d.rounded_rectangle((x, y + 3, x + 24, y + 27), 5, fill=color)
        else:
            d.rounded_rectangle((x, y + 3, x + 24, y + 27), 5, outline=color, width=3)
        d.text((x + 32, y), label, font=fb if n else f, fill=INK if n else INK3)
        x += 32 + d.textlength(label, font=fb if n else f) + 30
    dashed_rect(d, (x, y + 3, x + 24, y + 27), INK2, 2, 5)
    d.text((x + 32, y), "прочие объекты (Grounding DINO)", font=f, fill=INK2)


# ---------------- панель ----------------

def render_panel(r: dict, image: Image.Image, out_path: Path, boxes: bool = True):
    """r — словарь результата по кадру (см. 03_predict.py).

    boxes=False — фото без рамок и без легенды классов: легенда нужна только вместе с рамками.
    """
    top = PHOTO[1] if boxes else 72
    height = top + PHOTO[3] + 20
    canvas = Image.new("RGB", (W, height), BG)
    d = ImageDraw.Draw(canvas)
    d.text((20, 18), r["title"], font=font(36, True), fill=INK)
    counts = r["equipment_counts"]
    if boxes:
        legend(d, counts)

    # слева: фото
    px, py, pw, ph = PHOTO[0], top, PHOTO[2], PHOTO[3]
    sx, sy = pw / image.width, ph / image.height
    canvas.paste(image.resize((pw, ph), Image.LANCZOS), (px, py))
    if boxes:
        placed, labels = [], []
        for (x1, y1, x2, y2, label, score) in r["other_boxes"]:
            box = (px + x1 * sx, py + y1 * sy, px + x2 * sx, py + y2 * sy)
            dashed_rect(d, box, "#0b0b0b", 6)
            dashed_rect(d, box, OTHER, 3)
            labels.append((box, f"{label} {score:.2f}", font(22), "#2a2e2b", OTHER))
        for (x1, y1, x2, y2, cls, score) in r["equipment_boxes"]:
            color = CLASS_COLORS.get(cls, "#52514e")
            box = (px + x1 * sx, py + y1 * sy, px + x2 * sx, py + y2 * sy)
            d.rectangle(box, outline=color, width=6)
            labels.append((box, f"{CLASS_RU.get(cls, cls)} {score:.2f}", font(24, True), color, text_on(color)))
        for box, t, fnt, bg, fg in labels:
            tw, th = d.textlength(t, font=fnt) + 14, fnt.size + 14
            lx, ly = place_label(box, tw, th, placed, (px, py, px + pw, py + ph))
            placed.append((lx, ly, lx + tw, ly + th))
            d.rectangle((lx, ly, lx + tw, ly + th), fill=bg)
            d.text((lx + 7, ly + 6), t, font=fnt, fill=fg)

    # справа: текст
    bottom = height - 22
    y = section(d, top - 4, "СТАДИЯ")
    for line in wrap(d, r["stage_name"], font(48, True), SIDE_W)[:2]:
        d.text((SIDE_X, y), line, font=font(48, True), fill=INK)
        y += 58
    d.text((SIDE_X, y), f"уверенность {r['stage_prob']:.0%}", font=font(26), fill=INK2)
    y += 44
    for i, (name, prob) in enumerate(zip(r["stage_names"], r["stage_probs"])):
        mark = i == r["stage_idx"]
        d.text((SIDE_X, y), f"{i + 1}. {name}", font=font(24, mark), fill=INK if mark else INK2)
        bx = SIDE_X + 620
        d.rounded_rectangle((bx, y + 6, bx + 360, y + 24), 4, fill=SUNK)
        d.rounded_rectangle((bx, y + 6, bx + max(5, 360 * prob), y + 24), 4, fill="#1d5fae" if mark else "#9ab1cc")
        d.text((bx + 372, y), f"{prob:.0%}", font=font(24), fill=INK2)
        if r.get("stage_label") == i:
            d.text((bx - 30, y), "◆", font=font(24), fill="#a86b00")
        y += 34
    if r.get("stage_label") is not None:
        d.text((SIDE_X, y + 2), "◆ — стадия по разметке обучающего видео", font=font(20), fill=INK3)
        y += 30

    y = section(d, y + 20, "ГОТОВНОСТЬ И ГРАФИК")
    d.text((SIDE_X, y - 8), f"{r['pred_progress']:.0%}", font=font(96, True), fill=INK)
    color, label = STATUS[r["status"]]
    tx = SIDE_X + 250
    if r.get("expected") is not None:
        plan_lbl = f"план на {ru_date(r['frame_date'])}" if r.get("plan_basis") == "date" else "по плану"
        d.text((tx, y + 4), f"{plan_lbl}: {r['expected']:.0%}", font=font(32), fill=INK2)
        d.text((tx, y + 50), f"{signed(r['delta_pp'])} п.п. · ≈ {signed(r['delta_days'], 0)} дн.",
               font=font(32, True), fill=color)
    y += 122
    d.rounded_rectangle((SIDE_X, y, SIDE_X + SIDE_W, y + 26), 7, fill=SUNK)
    d.rounded_rectangle((SIDE_X, y, SIDE_X + max(10, SIDE_W * r["pred_progress"]), y + 26), 7, fill=color)
    if r.get("expected") is not None:
        mx = SIDE_X + SIDE_W * r["expected"]
        d.rectangle((mx - 3, y - 8, mx + 3, y + 34), fill=INK)
    y += 42
    pw_ = d.textlength(label, font=font(28, True)) + 52
    d.rounded_rectangle((SIDE_X, y, SIDE_X + pw_, y + 48), 24, fill=color)
    d.text((SIDE_X + 26, y + 8), label, font=font(28, True), fill="#ffffff")
    y += 62
    if r.get("frame_date"):
        when = ru_date(r["frame_date"]) + (f" {r['frame_time']}" if r.get("frame_time") else "")
        d.text((SIDE_X, y), f"Дата на кадре: {when} — распознана OCR", font=font(24, True), fill=INK)
        y += 34
        if r.get("video_span"):
            sp = r["video_span"]
            d.text((SIDE_X, y), f"съёмка объекта {ru_date(sp['start'])} – {ru_date(sp['end'])}; план — по календарю",
                   font=font(21), fill=INK2)
            y += 32
    elif r.get("expected") is not None:
        d.text((SIDE_X, y), "Даты на кадре нет — план по положению кадра в видео", font=font(22), fill=INK2)
        y += 32

    y = section(d, y + 16, "ТЕХНИКА В КАДРЕ")
    if counts:
        for cls, n in sorted(counts.items(), key=lambda kv: -kv[1]):
            d.rounded_rectangle((SIDE_X, y + 5, SIDE_X + 22, y + 27), 4, fill=CLASS_COLORS.get(cls, INK2))
            d.text((SIDE_X + 34, y), CLASS_RU.get(cls, cls), font=font(28), fill=INK)
            d.text((SIDE_X + 470, y + 2), f"{n} шт. · уверенность до {r['equipment_conf'][cls]:.2f}",
                   font=font(25), fill=INK2)
            y += 38
    else:
        d.text((SIDE_X, y), "техника не обнаружена", font=font(28), fill=INK2)
        y += 38
    if r["other_counts"]:
        others = ", ".join(f"{k} — {v}" for k, v in sorted(r["other_counts"].items(), key=lambda kv: -kv[1]))
        for line in wrap(d, "Прочие объекты: " + others, font(24), SIDE_W)[:2]:
            d.text((SIDE_X, y + 4), line, font=font(24), fill=INK2)
            y += 32

    vlm = r.get("vlm_text")
    y = section(d, y + 18, "ВЫВОД (собран из чисел выше)")
    for line in clip_lines(wrap(d, r["summary"], font(26), SIDE_W), 4 if vlm else 6):
        d.text((SIDE_X, y), line, font=font(26), fill=INK)
        y += 36

    # траектория занимает то, что осталось, с запасом под блок VLM
    reserve = 250 if vlm else 0
    room = bottom - reserve - y - 18 - 36 - 40
    if r.get("trajectory") and room >= 110:
        y = section(d, y + 18, "ТРАЕКТОРИЯ ОБЪЕКТА · прогноз по всему видео")
        y = trajectory_chart(d, y, r["trajectory"], r, min(room, 300))

    if vlm:
        y = section(d, y + 16, "НЕЗАВИСИМАЯ ПРОВЕРКА · " + r["vlm_model"])
        agree = r.get("vlm_agree")
        if agree is not None:
            vc, vt = ("#1a7f45", f"стадия {r['vlm_stage'] + 1} — совпадает с моделью ✓") if agree else \
                     ("#c23b33", f"стадия {r['vlm_stage'] + 1} — не совпадает с моделью ✗")
            d.text((SIDE_X, y), vt, font=font(26, True), fill=vc)
            y += 38
        for line in clip_lines(wrap(d, "«" + vlm + "»", font(23), SIDE_W), max(0, (bottom - y) // 32)):
            d.text((SIDE_X, y), line, font=font(23), fill=INK2)
            y += 32

    canvas.save(out_path)


def trajectory_chart(d, y, traj, r, h):
    """Прогноз готовности по кадрам видео (линия), план (пунктир), текущий кадр (точка)."""
    x0, x1 = SIDE_X + 62, SIDE_X + SIDE_W - 8
    y1 = y + 6 + h
    for v in (0, 0.25, 0.5, 0.75, 1):
        yy = y1 - v * h
        d.line((x0, yy, x1, yy), fill=LINE, width=1)
        d.text((SIDE_X, yy - 12), f"{v:.0%}", font=font(19), fill=INK3)
    X = lambda t: x0 + t * (x1 - x0)
    Y = lambda v: y1 - v * h
    finish = r.get("plan_finish") or 1.0
    plan = [(X(t), Y(min(t / finish, 1))) for t in np.linspace(0, 1, 80)]
    for i in range(0, len(plan) - 1, 2):
        d.line((*plan[i], *plan[i + 1]), fill=INK2, width=3)
    d.line([(X(t), Y(v)) for t, v in zip(traj["t"], traj["pred"])], fill="#1d5fae", width=5, joint="curve")
    if r.get("t") is not None:
        cx, cy = X(r["t"]), Y(r["pred_progress"])
        d.ellipse((cx - 12, cy - 12, cx + 12, cy + 12), fill=STATUS[r["status"]][0], outline="#ffffff", width=4)
    d.text((x0, y1 + 8), "начало съёмки", font=font(19), fill=INK3)
    d.text((x1 - d.textlength("конец", font=font(19)), y1 + 8), "конец", font=font(19), fill=INK3)
    lx = x0 + 230
    d.line((lx, y1 + 20, lx + 36, y1 + 20), fill="#1d5fae", width=5)
    d.text((lx + 46, y1 + 8), "прогноз модели", font=font(19), fill=INK2)
    lx += 250
    for i in range(0, 36, 12):
        d.line((lx + i, y1 + 20, lx + i + 6, y1 + 20), fill=INK2, width=3)
    d.text((lx + 46, y1 + 8), "план", font=font(19), fill=INK2)
    lx += 140
    d.ellipse((lx, y1 + 11, lx + 18, y1 + 29), fill=STATUS[r["status"]][0])
    d.text((lx + 28, y1 + 8), "этот кадр", font=font(19), fill=INK2)
    return y1 + 40


# ---------------- отдельная карта внимания ----------------

def render_heatmap(r: dict, image: Image.Image, out_path: Path):
    """Кадр с наложенной картой внимания головы: где модель искала признаки стадии и готовности."""
    import matplotlib
    grid = np.asarray(r["attention_grid"], np.float32)
    a = (grid - grid.min()) / (grid.max() - grid.min() + 1e-8)
    iw, ih = image.size
    hm = Image.fromarray((a * 255).astype(np.uint8)).resize((iw, ih), Image.BICUBIC)
    cmap = matplotlib.colormaps["inferno"]
    rgb = (cmap(np.asarray(hm) / 255.0)[..., :3] * 255).astype(np.uint8)
    overlay = Image.blend(image.convert("RGB"), Image.fromarray(rgb), 0.55)

    top, bottom = 96, 76
    canvas = Image.new("RGB", (iw + 40, ih + top + bottom), BG)
    d = ImageDraw.Draw(canvas)
    d.text((20, 14), f"Куда смотрела модель · {r['title']}", font=font(30, True), fill=INK)
    d.text((20, 56), f"Веса внимания головы по {grid.shape[0]}×{grid.shape[1]} участкам кадра. "
                     f"Прогноз: {r['stage_name']}, готовность {r['pred_progress']:.0%}.", font=font(21), fill=INK2)
    canvas.paste(overlay, (20, top))
    bx, by, bw = 20, top + ih + 22, 420
    grad = (cmap(np.linspace(0, 1, bw))[:, :3] * 255).astype(np.uint8)
    canvas.paste(Image.fromarray(np.repeat(grad[None], 18, 0)), (bx, by))
    d.text((bx, by + 22), "меньше влияния", font=font(16), fill=INK3)
    d.text((bx + bw - d.textlength("больше", font=font(16)), by + 22), "больше", font=font(16), fill=INK3)
    note = ("Хорошо, когда подсвечено само здание или котлован. Если подсвечены дата, подпись или небо — "
            "модель опирается на случайную подсказку.")
    for i, line in enumerate(wrap(d, note, font(18), iw - bw - 30)[:2]):
        d.text((bx + bw + 30, by - 4 + i * 26), line, font=font(18), fill=INK2)
    canvas.save(out_path, quality=92)
