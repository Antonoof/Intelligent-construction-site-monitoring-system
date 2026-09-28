"""Детекция по фрагментам кадра для общих планов с высоты (как в ML-части).

Кадр целиком плюс сетка N×N перекрывающихся фрагментов. С фрагментов берутся только мелкие
рамки (меньше 2 % площади кадра) — крупные объекты надёжнее находит проход по целому кадру.
Всё сливается через NMS по классам.
"""
from __future__ import annotations

from typing import Callable

import numpy as np
from PIL import Image

from .base import RawDet


def tiles_grid(width: int, height: int, n: int, overlap: float = 0.2) -> list[tuple[int, int, int, int]]:
    tw, th = width / (n - (n - 1) * overlap), height / (n - (n - 1) * overlap)
    sx, sy = tw * (1 - overlap), th * (1 - overlap)
    out = []
    for i in range(n):
        for j in range(n):
            x1, y1 = int(round(j * sx)), int(round(i * sy))
            out.append((x1, y1, min(width, int(round(x1 + tw))), min(height, int(round(y1 + th)))))
    return out


def iou(a, b) -> float:
    ix1, iy1, ix2, iy2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def nms(dets: list[RawDet], iou_thr: float = 0.5) -> list[RawDet]:
    keep: list[RawDet] = []
    for d in sorted(dets, key=lambda x: -x.conf):
        if all(k.cls != d.cls or iou(k.box, d.box) < iou_thr for k in keep):
            keep.append(d)
    return keep


def detect_tiled(img: Image.Image, detect_fn: Callable[[Image.Image], list[RawDet]], n: int = 3,
                 small_area: float = 0.02, iou_thr: float = 0.5) -> list[RawDet]:
    W, H = img.size
    dets = list(detect_fn(img))
    frame_area = float(W * H)
    for (x1, y1, x2, y2) in tiles_grid(W, H, n):
        crop = img.crop((x1, y1, x2, y2))
        for d in detect_fn(crop):
            bx = (d.box[0] + x1, d.box[1] + y1, d.box[2] + x1, d.box[3] + y1)
            if (bx[2] - bx[0]) * (bx[3] - bx[1]) / frame_area < small_area:
                dets.append(RawDet(d.cls, d.conf, bx, d.source))
    return nms(dets, iou_thr)


def to_array(img: Image.Image) -> np.ndarray:
    return np.asarray(img.convert("RGB"))
