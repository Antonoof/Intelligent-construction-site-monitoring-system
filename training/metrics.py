"""Метрики детектора без внешних зависимостей: AP50 по классам (интерполяция по всем точкам, как в VOC/COCO),
точность и полнота при рабочем пороге уверенности."""
from __future__ import annotations

from collections import defaultdict

import numpy as np


def iou(a, b) -> float:
    ix1, iy1, ix2, iy2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def average_precision(recall: np.ndarray, precision: np.ndarray) -> float:
    r = np.concatenate([[0.0], recall, [1.0]])
    p = np.concatenate([[0.0], precision, [0.0]])
    for i in range(len(p) - 2, -1, -1):          # огибающая: точность не растёт с полнотой
        p[i] = max(p[i], p[i + 1])
    idx = np.where(r[1:] != r[:-1])[0]
    return float(np.sum((r[idx + 1] - r[idx]) * p[idx + 1]))


def evaluate(gt: dict, pred: dict, classes: list[str], iou_thr: float = 0.5, conf_thr: float = 0.5) -> dict:
    """gt/pred: {image_id: [(cls, x1, y1, x2, y2[, conf])]} → {cls: {ap50, precision, recall, n_gt}}."""
    out = {}
    for c in classes:
        gts = {k: [b for b in v if b[0] == c] for k, v in gt.items()}
        n_gt = sum(len(v) for v in gts.values())
        dets = sorted(((k, b) for k, v in pred.items() for b in v if b[0] == c), key=lambda x: -x[1][5])
        used = defaultdict(set)
        tp = np.zeros(len(dets))
        for i, (k, b) in enumerate(dets):
            best, best_j = 0.0, -1
            for j, g in enumerate(gts.get(k, [])):
                if j in used[k]:
                    continue
                o = iou(b[1:5], g[1:5])
                if o > best:
                    best, best_j = o, j
            if best >= iou_thr:
                tp[i] = 1
                used[k].add(best_j)
        if n_gt == 0:
            out[c] = {"ap50": None, "precision": None, "recall": None, "n_gt": 0}
            continue
        ctp = np.cumsum(tp)
        cfp = np.cumsum(1 - tp)
        rec = ctp / n_gt
        prec = ctp / np.maximum(ctp + cfp, 1e-9)
        confs = np.array([b[5] for _, b in dets])
        sel = confs >= conf_thr
        tp_at = tp[sel].sum()
        out[c] = {"ap50": round(average_precision(rec, prec), 4) if len(dets) else 0.0,
                  "precision": round(float(tp_at / max(sel.sum(), 1)), 4),
                  "recall": round(float(tp_at / n_gt), 4), "n_gt": n_gt}
    vals = [v["ap50"] for v in out.values() if v["ap50"] is not None]
    out["__mean__"] = {"ap50": round(float(np.mean(vals)), 4) if vals else None}
    return out
