"""Мониторинг строительной площадки: детекция техники на видео и сопоставление с графиком работ."""

from construction_monitor.config import PROJECT_ROOT, load_config
from construction_monitor.detector import Detector

__all__ = ["PROJECT_ROOT", "Detector", "load_config"]
