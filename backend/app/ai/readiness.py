"""Модель готовности из ML-части — только инференс: стадия S1–S5 и готовность объекта 0–100 % по кадру.

Обучение (training/ ML-части: 01_build_features → 02_train_head) здесь не нужно — берётся результат
training/runs/<дата>/, скопированный в weights/readiness/<дата>/:

    head.pt            голова: внимание по патчам DINOv2 + CLS + признаки техники → готовность и стадия
    train_embed.npy    эмбеддинги головы для обучающих кадров (float16, 256) — похожие кадры обучения
    train_meta.csv     видео, готовность и стадия обучающих кадров (в том же порядке)

Конвейер кадра повторяет training/03_predict.py (stage_perception):
    кадр → DINOv2 ViT-g (заморожен, facebook/dinov2-giant) → CLS + патчи, усреднённые до сетки 9×16
         + рамки RF-DETR этого же снимка (из БД, порог 0.3) → 4 признака на класс: log(1+n), макс. уверенность,
           доля площади кадра, средний центр по вертикали
         → голова → готовность (sigmoid), стадия (softmax), карта внимания 9×16

Плановая готовность берётся из календарного графика проекта: линейно от начала первого до окончания
последнего этапа; статус — по порогам из training/config.yaml (plan: ahead / risk / late).

OKO_READINESS: auto (weights/readiness/latest.txt или самый свежий прогон) | путь к прогону или head.pt | off.
DINOv2 ViT-g в float32 — ≈4.6 ГБ ОЗУ и 20–40 с на кадр на 4 vCPU; OKO_READINESS_DTYPE=bfloat16 — вдвое меньше памяти.
"""
from __future__ import annotations

import csv
import datetime as dt
import logging
import math
import threading
import time
from pathlib import Path

import numpy as np
from PIL import Image

from ..config import REPO_DIR, settings
from .vlm import resolve_device

log = logging.getLogger("oko.readiness")

DET_FEATS = 4
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], np.float32)
PLAN = {"ahead": 0.03, "risk": -0.03, "late": -0.08}           # training/config.yaml → plan
STATUS_RU = {"ahead": "опережаем", "on_track": "успеваем", "risk": "риск отставания", "late": "отстаём"}
BACKBONE_PARAMS_B = {"facebook/dinov2-giant": 1.14, "facebook/dinov2-large": 0.30, "facebook/dinov2-base": 0.09}


# ------------------------------------------------------------------ чистые функции (без torch)

def det_features(dets: list[tuple[str, float, tuple]], class_keys: list[str | None], width: int, height: int
                 ) -> np.ndarray:
    """Признаки техники как в common.det_features; class_keys — ключи классов в порядке чекпойнта головы."""
    idx = {k: i for i, k in enumerate(class_keys) if k}
    f = np.zeros((len(class_keys), DET_FEATS), np.float32)
    for cls, conf, (x1, y1, x2, y2) in dets:
        c = idx.get(cls)
        if c is None:
            continue
        f[c, 0] += 1
        f[c, 1] = max(f[c, 1], float(conf))
        f[c, 2] += (x2 - x1) * (y2 - y1) / (width * height)
        f[c, 3] += (y1 + y2) / 2 / height
    n = f[:, 0].copy()
    f[:, 3] = np.where(n > 0, f[:, 3] / np.maximum(n, 1), 0)
    f[:, 0] = np.log1p(n)
    return f.ravel()


def plan_status(delta: float | None) -> tuple[str, str]:
    if delta is None:
        return "none", "нет плана"
    key = "ahead" if delta >= PLAN["ahead"] else "on_track" if delta >= PLAN["risk"] else \
        "risk" if delta >= PLAN["late"] else "late"
    return key, STATUS_RU[key]


def plan_progress(tasks, day: dt.date) -> dict | None:
    """Плановая готовность на дату по графику: доля срока от начала первого до окончания последнего этапа."""
    real = [t for t in tasks if not t.is_summary]
    if not real:
        return None
    start, end = min(t.start for t in real), max(t.end for t in real)
    days = max((end - start).days, 1)
    return {"expected": min(max((day - start).days / days, 0.0), 1.0), "days": days,
            "start": start.isoformat(), "end": end.isoformat()}


def find_run(setting: str) -> Path | None:
    """Каталог прогона с head.pt: явный путь или weights/readiness/latest.txt, иначе самый свежий."""
    if setting.lower() in ("", "off", "0", "false", "no"):
        return None
    root = REPO_DIR / "weights" / "readiness"
    if setting.lower() != "auto":
        p = Path(setting).expanduser()
        p = p.parent if p.suffix == ".pt" else p
        if (p / "head.pt").exists():
            return p
        root = p                                   # каталог с прогонами (как weights/readiness)
    latest = root / "latest.txt"
    if latest.exists():
        p = root / latest.read_text(encoding="utf-8").strip()
        if (p / "head.pt").exists():
            return p
    runs = sorted(d for d in root.glob("*") if (d / "head.pt").exists()) if root.exists() else []
    return runs[-1] if runs else None


# ------------------------------------------------------------------ голова (как в training/common.py)

def build_head(dim: int, n_det: int, n_stages: int, hidden: int = 512):
    import torch
    from torch import nn

    class StageProgressHead(nn.Module):
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

        def forward(self, patches, cls, det):
            tok = self.proj_p(self.norm_p(patches))
            scores = self.key(tok) @ self.query / math.sqrt(tok.shape[-1])
            attn = scores.softmax(-1)
            pooled = (attn.unsqueeze(-1) * tok).sum(1)
            z = self.mlp(torch.cat([pooled, self.proj_c(self.norm_c(cls)), self.det(det)], -1))
            return self.progress(z).squeeze(-1).sigmoid(), self.stage(z), attn, z

    return StageProgressHead()


class ReadinessModel:
    def __init__(self, setting: str, device_pref: str = "auto", dtype: str = "float32", reserve_gb: float = 1.0):
        import importlib.util
        self.run = find_run(setting)
        self.setting = setting
        self.device_pref, self.dtype_name, self.reserve_gb = device_pref, dtype, reserve_gb
        self._lock = threading.Lock()
        self._ready = False
        self.reason = "" if self.run else ("выключена" if setting.lower() in ("", "off", "0", "false", "no")
                                           else "нет head.pt в weights/readiness/")
        if self.run and not all(importlib.util.find_spec(x) for x in ("torch", "transformers")):
            self.run, self.reason = None, "нет PyTorch и transformers (образ Dockerfile.ai)"

    @property
    def enabled(self) -> bool:
        return self.run is not None

    def pending_gb(self) -> float:
        """Сколько памяти займёт модель, когда загрузится (VLM выбирается с учётом этого запаса)."""
        if not self.enabled or self._ready:
            return 0.0
        return BACKBONE_PARAMS_B["facebook/dinov2-giant"] * (4 if self.dtype_name == "float32" else 2) * 1.15 + 0.5

    def plan(self) -> dict:
        if not self.enabled:
            return {"enabled": False, "reason": self.reason}
        return {"enabled": True, "run": self.run.name, "loaded": self._ready,
                "backbone": getattr(self, "backbone_name", None), "device": getattr(self, "device", None)}

    def _load(self) -> None:
        import torch
        from transformers import AutoModel
        try:
            ck = torch.load(self.run / "head.pt", map_location="cpu", weights_only=True)
        except Exception:                          # старый формат чекпойнта — доверяем своему файлу
            ck = torch.load(self.run / "head.pt", map_location="cpu", weights_only=False)
        device, avail = resolve_device(self.device_pref)
        dtype = getattr(torch, self.dtype_name, torch.float32)
        name = ck["backbone"]
        need = BACKBONE_PARAMS_B.get(name, 1.14) * (2 if dtype != torch.float32 else 4) * 1.15 + 0.5
        if avail - self.reserve_gb < need:
            raise RuntimeError(f"модели готовности нужно ≈{need:.1f} ГБ, доступно {avail:.1f} ГБ "
                               f"(OKO_READINESS_DTYPE=bfloat16 — вдвое меньше)")
        t0 = time.perf_counter()
        head = build_head(ck["dim"], ck["n_det"], len(ck["stages"]), ck["hidden"])
        head.load_state_dict(ck["state_dict"])
        self.head = head.to(device).eval()
        try:
            self.backbone = AutoModel.from_pretrained(name, dtype=dtype)
        except TypeError:                          # transformers < 4.56
            self.backbone = AutoModel.from_pretrained(name, torch_dtype=dtype)
        self.backbone = self.backbone.to(device).eval()
        mc = self.backbone.config
        self.patch = int(getattr(mc, "patch_size", 14))
        self.n_reg = int(getattr(mc, "num_register_tokens", 0) or 0)
        h, w = ck["image_size"]
        self.hw = (h - h % self.patch, w - w % self.patch)
        self.grid = tuple(ck["grid"])
        self.stages = ck["stages"]
        self.class_names = list(ck["class_names"])
        self.device, self.dtype, self.backbone_name = device, dtype, name
        self.train_z = self.train_meta = None
        emb = self.run / "train_embed.npy"
        if emb.exists():
            z = np.load(emb).astype(np.float32)
            self.train_z = torch.from_numpy(z / np.maximum(np.linalg.norm(z, axis=1, keepdims=True), 1e-6)).to(device)
            meta = self.run / "train_meta.csv"
            if meta.exists():
                with open(meta, encoding="utf-8") as fh:
                    rows = list(csv.DictReader(fh))
                if len(rows) == len(z):
                    self.train_meta = rows
        self._ready = True
        log.info("модель готовности %s (%s) загружена на %s за %.0f с", self.run.name, name, device,
                 time.perf_counter() - t0)

    def _preprocess(self, img: Image.Image):
        import torch
        h, w = self.hw
        rgb = np.asarray(img.convert("RGB"))
        try:
            import cv2                               # как при обучении: INTER_AREA
            x = cv2.resize(rgb, (w, h), interpolation=cv2.INTER_AREA)
        except ImportError:
            x = np.asarray(img.convert("RGB").resize((w, h), Image.BOX))
        x = (x.astype(np.float32) / 255 - IMAGENET_MEAN) / IMAGENET_STD
        return torch.from_numpy(x).permute(2, 0, 1)[None]

    def predict(self, img: Image.Image, dets: list[tuple[str, float, tuple]], class_key) -> dict:
        """dets — (ключ класса, уверенность, рамка в пикселях снимка); class_key — имя класса чекпойнта → ключ."""
        import torch
        import torch.nn.functional as F
        with self._lock:
            if not self._ready:
                self._load()
            t0 = time.perf_counter()
            x = self._preprocess(img).to(self.device, self.dtype)
            with torch.inference_mode():
                hs = self.backbone(pixel_values=x).last_hidden_state
                gh, gw = self.hw[0] // self.patch, self.hw[1] // self.patch
                patches = hs[:, 1 + self.n_reg:1 + self.n_reg + gh * gw]
                patches = patches.reshape(1, gh, gw, -1).permute(0, 3, 1, 2).float()
                pooled = F.adaptive_avg_pool2d(patches, self.grid).flatten(2).transpose(1, 2)
                cls = hs[:, 0].float()
                keys = [class_key(n) for n in self.class_names]
                dfeat = torch.from_numpy(det_features(dets, keys, img.width, img.height)[None]).to(self.device)
                prog, logits, attn, z = self.head(pooled, cls, dfeat)
                probs = logits.softmax(-1)[0].float().cpu().numpy()
                zn = F.normalize(z.float(), dim=-1)[0]
                sims = (self.train_z @ zn).cpu().numpy() if self.train_z is not None else None
        i = int(probs.argmax())
        a = attn[0].float().cpu().numpy().reshape(self.grid)
        out = {"stage": self.stages[i]["id"], "stage_name": self.stages[i]["name"],
               "stage_prob": round(float(probs[i]), 3),
               "stage_probs": {s["id"]: round(float(p), 3) for s, p in zip(self.stages, probs)},
               "readiness": round(float(prog[0]) * 100, 1),
               "attention": np.round(a / max(a.max(), 1e-9), 3).tolist(),
               "model": f"{self.backbone_name.split('/')[-1]} + голова {self.run.name}",
               "seconds": round(time.perf_counter() - t0, 1)}
        if sims is not None:
            # похожие обучающие кадры — как neighbors в 03_predict: лучший кадр каждого из трёх других видео
            if self.train_meta:
                nbs, seen = [], set()
                for j in np.argsort(-sims):
                    row = self.train_meta[int(j)]
                    if row["video"] in seen:
                        continue
                    seen.add(row["video"])
                    nbs.append(float(row["progress"]))
                    if len(nbs) == 3:
                        break
                out["neighbors_readiness"] = round(float(np.mean(nbs)) * 100, 1)
        return out


_model: ReadinessModel | None = None


def get_readiness() -> ReadinessModel:
    global _model
    if _model is None:
        _model = ReadinessModel(settings.readiness, settings.device, settings.readiness_dtype)
    return _model


def set_readiness(m) -> None:
    """Подмена модели (тесты)."""
    global _model
    _model = m


def assess_snapshot(s, snap) -> dict | None:
    """Оценить снимок моделью готовности, сравнить с планом по графику, сохранить в snapshots.assessment
    и пересчитать отклонения дня (правило STAGE_MISMATCH по кадрам общего плана)."""
    from .. import models as M
    from .. import services as S
    from ..imaging import load_image
    from ..methodology import get_methodology

    model = get_readiness()
    if not model.enabled:
        return None
    p = s.get(M.Project, snap.project_id)
    if p.is_demo and snap.assessment:          # демо-сцены синтетические: оставляем оценку из разметки
        return snap.assessment
    m = get_methodology()
    dets = [(d.equipment_class, d.confidence, (d.x1, d.y1, d.x2, d.y2)) for d in snap.detections
            if not d.is_rejected and d.source != "llm" and d.confidence >= 0.3]
    img = load_image(Path(snap.file_path).read_bytes())
    out = model.predict(img, dets, m.normalize_class)
    cam = s.get(M.Camera, snap.camera_id)
    out["overview"] = bool(cam.is_overview)
    if not cam.is_overview:
        out["note"] = "кадр зоны: модель обучена на общих планах площадки, оценка ориентировочная"
    plan = plan_progress(S.task_infos(s, p, m), snap.taken_at.date())
    if plan:
        delta = out["readiness"] / 100 - plan["expected"]
        key, ru = plan_status(delta)
        out.update({"expected": round(plan["expected"] * 100, 1), "delta_pp": round(delta * 100, 1),
                    "delta_days": round(delta * plan["days"]), "status": key, "status_ru": ru,
                    "plan_basis": f"график {plan['start']} – {plan['end']}, линейно по сроку"})
    snap.assessment = out
    s.flush()
    S.recompute_day(s, p, snap.taken_at.date())
    return out
