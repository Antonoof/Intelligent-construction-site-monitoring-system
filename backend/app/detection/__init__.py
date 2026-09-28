"""Выбор детектора по настройке OKO_DETECTOR.

auto   — RF-DETR (PyTorch), если есть пакет rfdetr и веса; иначе ONNX-модель, если есть; иначе демо-разметка;
rfdetr — только RF-DETR на PyTorch (ошибка, если нет весов);
onnx   — RF-DETR, экспортированный в ONNX: CPU-ноутбук без PyTorch (training/08_export_cpu.py);
http   — удалённый AI-сервис на GPU (OKO_AI_URL);
demo   — демо-разметка из data/demo.
"""
from __future__ import annotations

import logging
import threading

from ..config import settings
from .base import DetectResult, Detector, RawDet

log = logging.getLogger("oko.detector")
_lock = threading.Lock()
_detector: Detector | None = None


def _build() -> Detector:
    from .demo import DemoDetector
    mode = settings.detector.lower()
    if mode in ("rfdetr", "auto"):
        try:
            from .rfdetr_backend import RFDETRDetector
            return RFDETRDetector(settings.rfdetr_weights, settings.rfdetr_variant, settings.device,
                                  settings.detector_threshold, settings.rfdetr_class_names, settings.rfdetr_trust)
        except Exception as e:
            if mode == "rfdetr":
                raise
            log.warning("RF-DETR недоступен (%s) — пробуем ONNX-модель", e)
    if mode in ("onnx", "auto"):
        try:
            from .onnx_backend import OnnxDetector
            return OnnxDetector(settings.onnx_model, threshold=settings.detector_threshold)
        except Exception as e:
            if mode == "onnx":
                raise
            log.warning("ONNX-модель недоступна (%s) — работаем на демо-разметке", e)
    if mode == "http":
        from .http_backend import HttpDetector
        return HttpDetector(settings.ai_url)
    return DemoDetector(settings.demo_dir)


def get_detector() -> Detector:
    global _detector
    with _lock:
        if _detector is None:
            _detector = _build()
        return _detector


def set_detector(d: Detector | None) -> None:
    """Подмена детектора (тесты, CLI)."""
    global _detector
    with _lock:
        _detector = d


__all__ = ["Detector", "DetectResult", "RawDet", "get_detector", "set_detector"]
