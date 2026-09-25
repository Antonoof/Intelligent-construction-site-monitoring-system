"""Общий код обучения: конфиг, метки, признаки DINOv2 и RF-DETR, голова."""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import yaml

TRAIN_DIR = Path(__file__).resolve().parent
ROOT = TRAIN_DIR.parent

# Цвета классов — те же, что на видео и в отчётах.
CLASS_COLORS = {
    "Excavator": "#2a78d6", "Dump truck": "#eb6834", "Bucket loader": "#1baf7a",
    "Crane manipulator": "#eda100", "Autocran": "#4a3aa7", "Mixer": "#e87ba4",
    "Bulldozer": "#e34948", "Motor grader": "#008300", "Tower crane": "#9085e9",
    "Drilling rig": "#8a5a2b",
}
CLASS_RU = {
    "Excavator": "экскаватор", "Dump truck": "самосвал", "Bucket loader": "погрузчик",
    "Crane manipulator": "кран-манипулятор", "Autocran": "автокран", "Mixer": "бетоносмеситель",
    "Bulldozer": "бульдозер", "Motor grader": "грейдер", "Tower crane": "башенный кран",
    "Drilling rig": "буровая установка",
}
DET_FEATS = 4          # на класс: log(1+число), макс. уверенность, доля площади кадра, средний центр по вертикали
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], np.float32)


def load_config(path: str | Path | None = None) -> dict:
    path = Path(path) if path else TRAIN_DIR / "config.yaml"
    cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
    cfg["videos"] = cfg.get("videos") or {}
    return cfg


def p(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


# ---------------- метки ----------------

def stage_bounds(cfg: dict, video: str) -> list[float]:
    own = (cfg["videos"].get(video) or {}).get("stage_until")
    return list(own) if own else [s["until"] for s in cfg["stages"]]


def stage_of(progress: float, bounds: list[float]) -> int:
    for i, b in enumerate(bounds):
        if progress <= b + 1e-9:
            return i
    return len(bounds) - 1


def plan_expected(cfg: dict, video: str | None, t: float | None) -> float | None:
    """Плановая готовность в момент t (доля времени видео). None — плана нет."""
    if video is None or t is None:
        return None
    finish = float((cfg["videos"].get(video) or {}).get("plan_finish", 1.0))
    return min(t / finish, 1.0)


def plan_status(cfg: dict, delta: float | None) -> tuple[str, str]:
    """(ключ, подпись) по отклонению прогресса от плана."""
    if delta is None:
        return "none", "нет плана"
    pl = cfg["plan"]
    if delta >= pl["ahead"]:
        return "ahead", "опережаем"
    if delta >= pl["risk"]:
        return "on_track", "успеваем"
    if delta >= pl["late"]:
        return "risk", "риск отставания"
    return "late", "отстаём"


def total_days(cfg: dict, video: str | None) -> float:
    return float((cfg["videos"].get(video) or {}).get("total_days", cfg["plan"]["total_days"]))


def monotone(values) -> np.ndarray:
    """Изотоническая регрессия (PAVA): готовность стройки со временем не убывает, поэтому прогнозы по кадрам
    одного видео, упорядоченным по времени, сглаживаются до ближайшей неубывающей последовательности."""
    vals, sizes = [], []
    for v in np.asarray(values, float):
        vals.append(v)
        sizes.append(1)
        while len(vals) > 1 and vals[-2] > vals[-1]:
            v2, s2 = vals.pop(), sizes.pop()
            vals[-1] = (vals[-1] * sizes[-1] + v2 * s2) / (sizes[-1] + s2)
            sizes[-1] += s2
    return np.repeat(vals, sizes)


# ---------------- признаки RF-DETR ----------------

def load_detector(cfg: dict, device: str):
    import logging
    import warnings

    from rfdetr import RFDETR
    logging.getLogger("rf-detr").setLevel(logging.ERROR)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        model = RFDETR.from_checkpoint(str(p(cfg["paths"]["rfdetr_weights"])), device=device,
                                       trust_checkpoint=True)
    if device.startswith("cuda"):
        import torch
        model.inference(compile=False, dtype=torch.float16)
    return model


def detect(model, rgb_images: list[np.ndarray], threshold: float):
    out = model.predict(rgb_images, threshold=threshold, include_source_image=False)
    return out if isinstance(out, list) else [out]


def detect_tiled(model, rgb: np.ndarray, threshold: float, grid: int = 3, overlap: float = 0.25, nms_iou: float = 0.5,
                 max_area: float = 0.02):
    """Кадр целиком + сетка grid×grid перекрывающихся фрагментов: мелкая техника на общих планах видна детектору
    крупнее. С фрагментов берутся только мелкие рамки (меньше max_area площади кадра) — крупные объекты находит
    проход по целому кадру. Рамки переводятся в координаты кадра и сливаются через NMS по классам."""
    import supervision as sv
    h, w = rgb.shape[:2]
    tw, th = int(w / grid * (1 + overlap)), int(h / grid * (1 + overlap))
    tiles = [(x, y) for y in np.linspace(0, h - th, grid).round().astype(int)
             for x in np.linspace(0, w - tw, grid).round().astype(int)]
    dets = detect(model, [rgb] + [np.ascontiguousarray(rgb[y:y + th, x:x + tw]) for x, y in tiles], threshold)
    parts = [dets[0]]
    for (x, y), d in zip(tiles, dets[1:]):
        d = d[d.area / (w * h) < max_area] if len(d) else d
        if len(d):
            d.xyxy = d.xyxy + np.array([x, y, x, y], dtype=d.xyxy.dtype)
            parts.append(d)
    return sv.Detections.merge(parts).with_nms(threshold=nms_iou, class_agnostic=False)


def det_features(det, n_classes: int, width: int, height: int) -> np.ndarray:
    f = np.zeros((n_classes, DET_FEATS), np.float32)
    for (x1, y1, x2, y2), c, s in zip(det.xyxy, det.class_id, det.confidence):
        c = int(c)
        f[c, 0] += 1
        f[c, 1] = max(f[c, 1], float(s))
        f[c, 2] += (x2 - x1) * (y2 - y1) / (width * height)
        f[c, 3] += (y1 + y2) / 2 / height
    n = f[:, 0].copy()
    f[:, 3] = np.where(n > 0, f[:, 3] / np.maximum(n, 1), 0)
    f[:, 0] = np.log1p(n)
    return f.ravel()


# ---------------- дата на кадре (OCR) ----------------

def load_ocr(device: str):
    import easyocr
    return easyocr.Reader(["en"], gpu=device.startswith("cuda"), verbose=False)


def _valid_date(y, m, d):
    from datetime import date
    try:
        y, m, d = int(y), int(m), int(d)
        return date(y, m, d) if 2000 <= y <= 2040 else None
    except ValueError:
        return None


def parse_date(texts: list[str]):
    """Дата из распознанных строк: 26-01-2022, 24.11.2025, 2022/01/26, а также «2601-2022» (OCR потерял знак)."""
    import re
    for t in texts:
        m = re.search(r"(\d{1,2})[.\-/ ](\d{1,2})[.\-/ ](\d{4})", t)
        if m and (dt := _valid_date(m[3], m[2], m[1])):
            return dt
        m = re.search(r"(\d{4})[.\-/ ](\d{1,2})[.\-/ ](\d{1,2})", t)
        if m and (dt := _valid_date(m[1], m[2], m[3])):
            return dt
        digits = re.sub(r"\D", "", t)
        for i in range(max(0, len(digits) - 7)):
            ch = digits[i:i + 8]
            if dt := (_valid_date(ch[4:], ch[2:4], ch[:2]) or _valid_date(ch[:4], ch[4:6], ch[6:])):
                return dt
    return None


def parse_time(texts: list[str]):
    import re
    for t in texts:
        m = re.search(r"\b([01]?\d|2[0-3]):([0-5]\d)(?::([0-5]\d))?\b", t)
        if m:
            return f"{int(m[1]):02d}:{m[2]}"
    return None


def read_frame_date(reader, rgb) -> dict:
    """Экранная дата камеры: ищется в верхней и нижней полосе кадра, где её обычно печатает камера."""
    h = rgb.shape[0]
    texts = []
    for strip in (rgb[: int(h * 0.2)], rgb[int(h * 0.75):]):
        texts += [t for _, t, conf in reader.readtext(strip, allowlist="0123456789.-:/ ", detail=1) if conf > 0.3]
    dt = parse_date(texts)
    return {"frame_date": dt.isoformat() if dt else None, "frame_time": parse_time(texts) if dt else None,
            "ocr_text": " | ".join(texts)}


# ---------------- DINOv2 ----------------

class Backbone:
    """Замороженный ViT: CLS-токен и патч-токены, усреднённые до сетки grid."""

    def __init__(self, cfg: dict, device: str):
        import torch
        from transformers import AutoModel

        b = cfg["backbone"]
        dtype = getattr(torch, b.get("dtype", "bfloat16")) if device.startswith("cuda") else torch.float32
        self.model = AutoModel.from_pretrained(b["name"], dtype=dtype).to(device).eval()
        mc = self.model.config
        self.patch = int(getattr(mc, "patch_size", 16))
        self.n_reg = int(getattr(mc, "num_register_tokens", 0) or 0)
        h, w = b["image_size"]
        self.hw = (h - h % self.patch, w - w % self.patch)
        self.grid = tuple(b["grid"])
        self.dim = int(mc.hidden_size)
        self.device, self.dtype = device, dtype

    def preprocess(self, rgb_images: list[np.ndarray]):
        import cv2
        import torch
        h, w = self.hw
        x = np.stack([cv2.resize(im, (w, h), interpolation=cv2.INTER_AREA) for im in rgb_images])
        x = (x.astype(np.float32) / 255 - IMAGENET_MEAN) / IMAGENET_STD
        return torch.from_numpy(x).permute(0, 3, 1, 2)

    def __call__(self, rgb_images: list[np.ndarray]):
        """-> cls [B, D], patches [B, gh*gw, D] (float32, numpy)."""
        import torch
        import torch.nn.functional as F
        x = self.preprocess(rgb_images).to(self.device, self.dtype)
        with torch.inference_mode():
            hs = self.model(pixel_values=x).last_hidden_state
            gh, gw = self.hw[0] // self.patch, self.hw[1] // self.patch
            patches = hs[:, 1 + self.n_reg:1 + self.n_reg + gh * gw]
            patches = patches.reshape(len(rgb_images), gh, gw, -1).permute(0, 3, 1, 2).float()
            pooled = F.adaptive_avg_pool2d(patches, self.grid).flatten(2).transpose(1, 2)
            return hs[:, 0].float().cpu().numpy(), pooled.cpu().numpy()


# ---------------- голова ----------------

def build_head(dim: int, n_det: int, n_stages: int, hidden: int = 512):
    import torch
    from torch import nn

    class StageProgressHead(nn.Module):
        """Внимание по патчам DINOv2 + CLS + признаки техники → прогресс 0–1 и стадия.

        Веса внимания — это и есть карта «куда смотрела модель» для объяснения вывода.
        """

        def __init__(self):
            super().__init__()
            self.norm_p, self.proj_p = nn.LayerNorm(dim), nn.Linear(dim, hidden)
            self.norm_c, self.proj_c = nn.LayerNorm(dim), nn.Linear(dim, hidden)
            self.key = nn.Linear(hidden, hidden)
            self.query = nn.Parameter(torch.randn(hidden) * 0.02)
            self.det = nn.Sequential(nn.Linear(n_det, 128), nn.GELU(), nn.Linear(128, 128))
            self.mlp = nn.Sequential(nn.Linear(2 * hidden + 128, hidden), nn.GELU(), nn.Dropout(0.1),
                                     nn.Linear(hidden, 256), nn.GELU())
            self.progress = nn.Linear(256, 1)
            self.stage = nn.Linear(256, n_stages)

        def forward(self, patches, cls, det, token_dropout: float = 0.0):
            tok = self.proj_p(self.norm_p(patches))
            scores = self.key(tok) @ self.query / math.sqrt(tok.shape[-1])
            if self.training and token_dropout > 0:
                drop = torch.rand_like(scores) < token_dropout
                scores = scores.masked_fill(drop, float("-inf"))
            attn = scores.softmax(-1)
            pooled = (attn.unsqueeze(-1) * tok).sum(1)
            z = self.mlp(torch.cat([pooled, self.proj_c(self.norm_c(cls)), self.det(det)], -1))
            return self.progress(z).squeeze(-1).sigmoid(), self.stage(z), attn, z

    return StageProgressHead()


def pick_device() -> str:
    import torch
    return "cuda" if torch.cuda.is_available() else "cpu"
