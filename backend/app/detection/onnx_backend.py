"""RF-DETR, экспортированный в ONNX, — детектор для ноутбуков без GPU (onnxruntime, без PyTorch).

Экспорт: python training/08_export_cpu.py → weights/rfdetr_v2.onnx + weights/rfdetr_v2.classes.json.
Предобработка и декодирование повторяют RFDETR.predict(): билинейное масштабирование с центрами
пикселей (без сглаживания), нормализация ImageNet, NCHW; выходы dets (cxcywh, 0..1) и labels (логиты,
последний слот — фон), top-k по всем парам «запрос × класс».
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image

from ..methodology import get_methodology
from .base import DetectResult, Detector, RawDet
from .tiling import detect_tiled

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def resize_half_pixel(a: np.ndarray, out_h: int, out_w: int) -> np.ndarray:
    """Билинейное масштабирование HWC float32 с центрами пикселей и без сглаживания (как torchvision, antialias=False)."""
    in_h, in_w = a.shape[:2]
    ys = (np.arange(out_h, dtype=np.float32) + 0.5) * (in_h / out_h) - 0.5
    xs = (np.arange(out_w, dtype=np.float32) + 0.5) * (in_w / out_w) - 0.5
    ys, xs = np.clip(ys, 0, in_h - 1), np.clip(xs, 0, in_w - 1)
    y0, x0 = np.floor(ys).astype(int), np.floor(xs).astype(int)
    y1, x1 = np.minimum(y0 + 1, in_h - 1), np.minimum(x0 + 1, in_w - 1)
    wy, wx = (ys - y0)[:, None, None], (xs - x0)[None, :, None]
    top = a[y0][:, x0] * (1 - wx) + a[y0][:, x1] * wx
    bot = a[y1][:, x0] * (1 - wx) + a[y1][:, x1] * wx
    return top * (1 - wy) + bot * wy


def preprocess(img: Image.Image, h: int, w: int) -> np.ndarray:
    a = np.asarray(img.convert("RGB"), dtype=np.float32) / 255.0
    a = resize_half_pixel(a, h, w)
    a = (a - MEAN) / STD
    return a.transpose(2, 0, 1)[None].astype(np.float32)


def decode(dets: np.ndarray, logits: np.ndarray, width: int, height: int, threshold: float,
           num_select: int | None = None) -> list[tuple[int, float, tuple[float, float, float, float]]]:
    """Сырые выходы → [(индекс класса, уверенность, xyxy в пикселях)]; последний слот логитов — фон."""
    prob = 1.0 / (1.0 + np.exp(-logits[0][:, :-1]))            # (Q, C-1)
    q, c = prob.shape
    flat = prob.reshape(-1)
    k = min(num_select or q, flat.size)
    idx = np.argpartition(-flat, k - 1)[:k]
    idx = idx[np.argsort(-flat[idx])]
    out = []
    for i in idx:
        score = float(flat[i])
        if score < threshold:
            break
        qi, ci = divmod(int(i), c)
        cx, cy, bw, bh = dets[0][qi]
        bw, bh = max(0.0, float(bw)), max(0.0, float(bh))
        box = ((cx - bw / 2) * width, (cy - bh / 2) * height, (cx + bw / 2) * width, (cy + bh / 2) * height)
        box = (max(0.0, box[0]), max(0.0, box[1]), min(float(width), box[2]), min(float(height), box[3]))
        out.append((ci, score, box))
    return out


class OnnxDetector(Detector):
    def __init__(self, model_path: Path, classes_path: Path | None = None, threshold: float = 0.3):
        import onnxruntime as ort
        self.model_path = Path(model_path)
        opts = ort.SessionOptions()
        opts.add_session_config_entry("session.intra_op.allow_spinning", "0")
        self.sess = ort.InferenceSession(str(self.model_path), sess_options=opts, providers=["CPUExecutionProvider"])
        inp = self.sess.get_inputs()[0]
        self.input_name = inp.name
        self.h, self.w = int(inp.shape[2]), int(inp.shape[3])
        names = [o.name for o in self.sess.get_outputs()]
        self.i_dets = next(i for i, n in enumerate(names) if "dets" in n)
        self.i_logits = next(i for i, n in enumerate(names) if "labels" in n)
        cp = Path(classes_path) if classes_path else self.model_path.with_suffix(".classes.json")
        class_names = json.loads(cp.read_text(encoding="utf-8"))
        m = get_methodology()
        self.index_to_class = {i: m.normalize_class(n) for i, n in enumerate(class_names)}
        self.classes = frozenset(c for c in self.index_to_class.values() if c)
        self.threshold = threshold
        self.name = f"onnx:{self.model_path.stem}"

    def _predict(self, img: Image.Image) -> list[RawDet]:
        x = preprocess(img, self.h, self.w)
        res = self.sess.run(None, {self.input_name: x})
        out = []
        for ci, score, box in decode(res[self.i_dets], res[self.i_logits], img.width, img.height, self.threshold):
            cls = self.index_to_class.get(ci)
            if cls:
                out.append(RawDet(cls, score, box, "model"))
        return out

    def detect(self, img: Image.Image, *, tiles: int = 0, sha256: str | None = None) -> DetectResult:
        dets = detect_tiled(img, self._predict, n=tiles) if tiles and tiles > 1 else self._predict(img)
        return DetectResult(dets, self.name)
