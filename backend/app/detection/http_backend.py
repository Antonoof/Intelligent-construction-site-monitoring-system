"""Удалённый AI-сервис на GPU (контракт POST /internal/detect из архитектуры, backend/readme.md).

Основной сервис (API, правила, БД) работает на CPU, а тяжёлые модели — отдельным сервисом на GPU.
Тот же код прототипа запускается как AI-сервис: OKO_DETECTOR=rfdetr, эндпоинт /internal/detect.
"""
from __future__ import annotations

import io

import httpx
from PIL import Image

from ..methodology import get_methodology
from .base import DetectResult, Detector, RawDet


class HttpDetector(Detector):
    def __init__(self, base_url: str, timeout: float = 30.0):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        r = httpx.get(f"{self.base_url}/internal/info", timeout=5.0)
        r.raise_for_status()
        info = r.json()
        self.name = f"http:{info['name']}"
        self.classes = frozenset(info["classes"])

    def detect(self, img: Image.Image, *, tiles: int = 0, sha256: str | None = None) -> DetectResult:
        buf = io.BytesIO()
        img.save(buf, "JPEG", quality=92)
        r = httpx.post(f"{self.base_url}/internal/detect", params={"tiles": tiles, "sha256": sha256 or ""},
                       files={"file": ("frame.jpg", buf.getvalue(), "image/jpeg")}, timeout=self.timeout)
        r.raise_for_status()
        data = r.json()
        m = get_methodology()
        dets = []
        for b in data["boxes"]:
            cls = m.normalize_class(b["cls"])
            if cls:
                dets.append(RawDet(cls, float(b["conf"]), tuple(map(float, b["xyxy"])), "model"))
        return DetectResult(dets, data.get("model_version", self.name), assessment=data.get("assessment"),
                            note=data.get("note", ""), recognized=bool(data.get("recognized", True)))
