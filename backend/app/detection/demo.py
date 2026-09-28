"""Демо-детектор: разметка из JSON рядом со снимком.

Нужен, чтобы прототип работал на любом ноутбуке без весов модели: демо-снимки лежат в
data/demo/snapshots вместе с разметкой (<имя>.json). Снимок находится по SHA-256 содержимого,
поэтому его можно загрузить через интерфейс под любым именем. Чужой снимок демо-детектор не знает:
он возвращает recognized=False, и снимок получает статус «нет данных» с подсказкой подключить веса RF-DETR.
"""
from __future__ import annotations

import hashlib
import json
from functools import cached_property
from pathlib import Path

from PIL import Image

from ..methodology import get_methodology
from .base import DetectResult, Detector, RawDet


class DemoDetector(Detector):
    def __init__(self, demo_dir: Path):
        self.demo_dir = Path(demo_dir)
        m = get_methodology()
        # демо-разметка покрывает все 12 классов модели v2
        self.classes = frozenset(k for k, v in m.classes.items() if "v2" in v.get("models", []))
        self.name = "demo-labels"

    @cached_property
    def index(self) -> dict[str, dict]:
        idx: dict[str, dict] = {}
        for js in sorted(self.demo_dir.rglob("*.json")):
            try:
                meta = json.loads(js.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if not isinstance(meta, dict) or "detections" not in meta:
                continue
            sha = meta.get("sha256")
            if not sha:
                img = js.with_suffix(".jpg")
                if not img.exists():
                    continue
                sha = hashlib.sha256(img.read_bytes()).hexdigest()
            idx[sha] = meta
        return idx

    def detect(self, img: Image.Image, *, tiles: int = 0, sha256: str | None = None) -> DetectResult:
        m = get_methodology()
        meta = self.index.get(sha256 or "")
        if meta is None:
            return DetectResult([], self.name, recognized=False,
                                note="Снимка нет в демо-разметке: для своих снимков подключите веса RF-DETR "
                                     "(OKO_DETECTOR=rfdetr или onnx) — см. weights/README.md.")
        dets = []
        for d in meta["detections"]:
            cls = m.normalize_class(d["cls"])
            if cls:
                dets.append(RawDet(cls, float(d["conf"]), tuple(map(float, d["xyxy"])), "demo"))
        return DetectResult(dets, self.name, assessment=meta.get("assessment"), note=meta.get("note", ""))
