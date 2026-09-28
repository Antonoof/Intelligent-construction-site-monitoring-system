"""Детектор RF-DETR — модель из ML-части решения (training/README.md).

Веса: weights/rfdetr_large_best_ema.pth (RF-DETR Large, 10 классов, mAP50 0.907) или новая
версия с катком и грузовиком (training/06_train_detector.py). На GPU — ~0.1–0.2 с на кадр,
на CPU ноутбука — секунды на кадр; для CPU можно экспортировать модель в ONNX/OpenVINO
(training/08_export_cpu.py).

Имена классов берутся из чекпойнта и приводятся к ключам методики через синонимы
(methodology/equipment.yaml). Если в чекпойнте имён нет — задайте порядок классов в OKO_RFDETR_CLASSES.
"""
from __future__ import annotations

import logging
from pathlib import Path

from PIL import Image

from ..methodology import get_methodology
from .base import DetectResult, Detector, RawDet
from .tiling import detect_tiled

log = logging.getLogger("oko.rfdetr")

# порядок классов модели v1 из ML-части, если чекпойнт не хранит имена
V1_CLASSES = ["excavator", "dump_truck", "loader", "crane_manipulator", "mobile_crane", "concrete_mixer",
              "bulldozer", "grader", "tower_crane", "drilling_rig"]


def _device(pref: str) -> str:
    if pref != "auto":
        return pref
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return "mps"
    except ImportError:
        pass
    return "cpu"


class RFDETRDetector(Detector):
    def __init__(self, weights: Path, variant: str = "large", device: str = "auto", threshold: float = 0.3,
                 class_names: str = "", trust: bool = True):
        import rfdetr  # noqa: F401 — понятная ошибка, если пакет не установлен

        self.weights = Path(weights)
        if not self.weights.exists():
            raise FileNotFoundError(f"Нет весов RF-DETR: {self.weights}")
        self.device = _device(device)
        self.threshold = threshold
        self.trust = trust
        self.model = self._load(variant)
        m = get_methodology()
        names = [s.strip() for s in class_names.split(",") if s.strip()]
        if not names:
            try:
                names = list(self.model.class_names)
            except Exception:
                names = []
        if not names or len(names) > 40:          # нет имён или COCO — берём порядок ML-части
            names = V1_CLASSES
        self.index_to_class = {i: m.normalize_class(n) for i, n in enumerate(names)}
        self.classes = frozenset(c for c in self.index_to_class.values() if c)
        unknown = [n for i, n in enumerate(names) if not self.index_to_class[i]]
        if unknown:
            log.warning("классы чекпойнта без соответствия в методике (игнорируются): %s", unknown)
        self.name = f"rfdetr-{variant}-{len(self.classes)}cls:{self.weights.stem}"
        log.info("RF-DETR загружен: %s на %s, классы: %s", self.weights, self.device, sorted(self.classes))

    def _load(self, variant: str):
        from rfdetr import RFDETR
        try:
            # современные чекпойнты хранят архитектуру и имена классов
            return RFDETR.from_checkpoint(str(self.weights), device=self.device, trust_checkpoint=self.trust)
        except Exception as e:  # старый формат — собираем модель по варианту
            log.info("from_checkpoint не сработал (%s), загружаем как RFDETR%s", e, variant.capitalize())
            import rfdetr
            cls = {"nano": "RFDETRNano", "small": "RFDETRSmall", "medium": "RFDETRMedium",
                   "base": "RFDETRBase", "large": "RFDETRLarge"}[variant.lower()]
            return getattr(rfdetr, cls)(pretrain_weights=str(self.weights), device=self.device,
                                        trust_checkpoint=self.trust)

    def _predict(self, img: Image.Image) -> list[RawDet]:
        det = self.model.predict(img, threshold=self.threshold)
        out: list[RawDet] = []
        names = det.data.get("class_name") if hasattr(det, "data") and det.data else None
        m = get_methodology()
        for k in range(len(det.xyxy)):
            cls = None
            if names is not None and len(names) > k:
                cls = m.normalize_class(str(names[k]))
            if cls is None:
                cls = self.index_to_class.get(int(det.class_id[k]))
            if cls:
                x1, y1, x2, y2 = (float(v) for v in det.xyxy[k])
                out.append(RawDet(cls, float(det.confidence[k]), (x1, y1, x2, y2), "model"))
        return out

    def detect(self, img: Image.Image, *, tiles: int = 0, sha256: str | None = None) -> DetectResult:
        dets = detect_tiled(img, self._predict, n=tiles) if tiles and tiles > 1 else self._predict(img)
        return DetectResult(dets, self.name)
