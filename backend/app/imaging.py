"""Работа со снимками: чтение, время съёмки, контроль качества, превью и разметка.

Контроль качества стоит до детекции: тёмный, пересвеченный, размытый или «туманный» снимок
не участвует в правилах отсутствия техники и даёт статус «нет данных».
"""
from __future__ import annotations

import datetime as dt
import hashlib
import io
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
from PIL import ExifTags, Image, ImageDraw, ImageFont, ImageOps

EXIF_DT_TAGS = {k for k, v in ExifTags.TAGS.items() if v in ("DateTimeOriginal", "DateTimeDigitized", "DateTime")}
# «cam01_2026-09-24_10-30.jpg», «20260924T103000», «2026.09.24 10:30:00», «CAM-01 24.09.2026 10-30»
# дата и, если есть, время; за датой не может сразу идти вторая дата — «2025-03-01_2025-04-30» это период
# съёмки, а не 20:25:04
_FN_ISO = re.compile(r"(?<![0-3]\d[._-][01]\d[._-])(20\d{2})[-_.]?([01]\d)[-_.]?([0-3]\d)"
                     r"(?:[T _\-]*(?!20\d{2}[-_.]?[01]\d[-_.]?[0-3]\d)([0-2]\d)[-_:.h]?([0-5]\d)(?:[-_:.m]?([0-5]\d))?)?")
_FN_RU = re.compile(r"([0-3]\d)[._-]([01]\d)[._-](20\d{2})[T _\-]*([0-2]\d)[-_:.h]([0-5]\d)(?:[-_:.]([0-5]\d))?")


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def load_image(data: bytes) -> Image.Image:
    img = Image.open(io.BytesIO(data))
    img = ImageOps.exif_transpose(img)
    return img.convert("RGB")


def exif_datetime(data: bytes) -> dt.datetime | None:
    try:
        img = Image.open(io.BytesIO(data))
        exif = img.getexif()
        values = [exif.get(t) for t in EXIF_DT_TAGS]
        try:
            sub = exif.get_ifd(0x8769)          # Exif IFD: DateTimeOriginal живёт здесь
            values = [sub.get(t) for t in EXIF_DT_TAGS] + values
        except Exception:
            pass
        for v in values:
            if v:
                return dt.datetime.strptime(str(v).strip()[:19], "%Y:%m:%d %H:%M:%S")
    except Exception:
        return None
    return None


def filename_datetime(name: str) -> dt.datetime | None:
    stem = Path(name).stem
    m = _FN_ISO.search(stem)
    try:
        if m:
            y, mo, d, h, mi, s = m.groups()
            if h is None:                 # в имени только дата — полдень: середина рабочей смены
                return dt.datetime(int(y), int(mo), int(d), 12, 0)
            return dt.datetime(int(y), int(mo), int(d), int(h), int(mi), int(s or 0))
        m = _FN_RU.search(stem)
        if m:
            d, mo, y, h, mi, s = m.groups()
            return dt.datetime(int(y), int(mo), int(d), int(h), int(mi), int(s or 0))
    except ValueError:
        return None
    return None


def resolve_time(data: bytes, filename: str, explicit: dt.datetime | None) -> tuple[dt.datetime, str]:
    """Время съёмки: указано вручную → EXIF → имя файла → время загрузки.

    Дату, впечатанную камерой в кадр, читает OCR ML-части (EasyOCR); в прототипе он не подключён —
    это запасной источник для промышленной схемы, если у камеры нет EXIF и имя файла без времени.
    """
    if explicit:
        return explicit.replace(tzinfo=None), "form"
    t = exif_datetime(data)
    if t:
        return t, "exif"
    t = filename_datetime(filename)
    if t:
        return t, "filename"
    return dt.datetime.now().replace(microsecond=0), "upload"


@dataclass
class Quality:
    brightness: float
    contrast: float
    sharpness: float
    ok: bool
    reason: str


def assess_quality(img: Image.Image, cfg: dict) -> Quality:
    g = img.convert("L")
    if g.width > 640:
        g = g.resize((640, max(1, round(g.height * 640 / g.width))), Image.BILINEAR)
    a = np.asarray(g, dtype=np.float32)
    brightness = float(a.mean() / 255.0)
    contrast = float(a.std() / 255.0)
    lap = a[1:-1, 2:] + a[1:-1, :-2] + a[2:, 1:-1] + a[:-2, 1:-1] - 4.0 * a[1:-1, 1:-1]
    sharpness = float(lap.var())
    reasons = []
    if brightness < cfg.get("min_brightness", 0.12):
        reasons.append("темно")
    elif brightness > cfg.get("max_brightness", 0.93):
        reasons.append("засветка")
    if contrast < cfg.get("min_contrast", 0.035):
        reasons.append("нет контраста: туман, засветка или закрыт объектив")
    elif sharpness < cfg.get("min_sharpness", 25.0):
        reasons.append("размыто")
    return Quality(round(brightness, 3), round(contrast, 3), round(sharpness, 1), not reasons, ", ".join(reasons))


def thumbnail(img: Image.Image, width: int = 480) -> bytes:
    t = img.copy()
    t.thumbnail((width, width * 2))
    buf = io.BytesIO()
    t.save(buf, "JPEG", quality=82)
    return buf.getvalue()


@lru_cache(maxsize=4)
def _font(size: int):
    for name in ("DejaVuSans-Bold.ttf", "DejaVuSans.ttf", "LiberationSans-Bold.ttf", "Arial.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    for p in Path("/usr/share/fonts").rglob("DejaVuSans*.ttf"):
        return ImageFont.truetype(str(p), size)
    return ImageFont.load_default()


def _hex(c: str) -> tuple[int, int, int]:
    c = c.lstrip("#")
    return tuple(int(c[i:i + 2], 16) for i in (0, 2, 4))


def render_annotated(img: Image.Image, boxes: list[dict], zones: list[dict] | None = None,
                     highlight: set[int] | None = None, width: int | None = None) -> bytes:
    """Снимок с зонами и рамками техники (для отчётов и доказательств).

    boxes: [{id, label, color, x1, y1, x2, y2, conf, dashed}]; zones: [{name, color, polygon (0..1)}].
    """
    base = img.copy().convert("RGBA")
    W, H = base.size
    overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(overlay)
    lw = max(2, W // 400)
    font = _font(max(12, W // 70))
    for z in zones or []:
        pts = [(x * W, y * H) for x, y in z["polygon"]]
        r, g, b = _hex(z.get("color", "#2A78D6"))
        d.polygon(pts, fill=(r, g, b, 40), outline=(r, g, b, 200))
        d.line(pts + [pts[0]], fill=(r, g, b, 220), width=lw)
        tx, ty = min(p[0] for p in pts) + 6, min(p[1] for p in pts) + 4
        tw = d.textlength(z["name"], font=font)
        d.rectangle([tx - 3, ty - 2, tx + tw + 3, ty + font.size + 3], fill=(r, g, b, 210))
        d.text((tx, ty), z["name"], fill=(255, 255, 255, 255), font=font)
    for bx in boxes:
        r, g, b = _hex(bx.get("color", "#E1A21C"))
        hl = highlight and bx.get("id") in highlight
        width_px = lw * (2 if hl else 1)
        x1, y1, x2, y2 = bx["x1"], bx["y1"], bx["x2"], bx["y2"]
        if bx.get("dashed"):
            step = max(8, W // 120)
            for x in np.arange(x1, x2, step * 2):
                d.line([(x, y1), (min(x + step, x2), y1)], fill=(r, g, b, 255), width=width_px)
                d.line([(x, y2), (min(x + step, x2), y2)], fill=(r, g, b, 255), width=width_px)
            for y in np.arange(y1, y2, step * 2):
                d.line([(x1, y), (x1, min(y + step, y2))], fill=(r, g, b, 255), width=width_px)
                d.line([(x2, y), (x2, min(y + step, y2))], fill=(r, g, b, 255), width=width_px)
        else:
            d.rectangle([x1, y1, x2, y2], outline=(r, g, b, 255), width=width_px)
        label = f'{bx["label"]} {bx["conf"]:.2f}'
        tw = d.textlength(label, font=font)
        ty = max(0, y1 - font.size - 6)
        d.rectangle([x1, ty, x1 + tw + 8, ty + font.size + 5], fill=(r, g, b, 235))
        d.text((x1 + 4, ty + 1), label, fill=(255, 255, 255, 255), font=font)
    out = Image.alpha_composite(base, overlay).convert("RGB")
    if width and out.width > width:
        out.thumbnail((width, width * 2))
    buf = io.BytesIO()
    out.save(buf, "JPEG", quality=88)
    return buf.getvalue()
