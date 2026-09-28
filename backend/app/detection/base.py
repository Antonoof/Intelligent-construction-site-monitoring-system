"""Общий интерфейс детекторов техники."""
from __future__ import annotations

from dataclasses import dataclass, field

from PIL import Image


@dataclass(frozen=True)
class RawDet:
    cls: str                      # ключ класса из methodology/equipment.yaml
    conf: float
    box: tuple[float, float, float, float]   # x1, y1, x2, y2 в пикселях снимка
    source: str = "model"         # model | demo | manual


@dataclass
class DetectResult:
    dets: list[RawDet]
    model: str
    assessment: dict | None = None   # стадия и готовность по кадру общего плана (ML-часть), если есть
    note: str = ""
    extra: dict = field(default_factory=dict)
    # False — детектор не смог обработать снимок (демо-детектор не знает чужой снимок): это «нет данных»,
    # а не «техники нет», иначе без весов модели любой свой снимок давал бы ложное «работы не ведутся»
    recognized: bool = True


class Detector:
    """Детектор: снимок → рамки техники канонических классов.

    name     — версия модели, пишется в каждый снимок (snapshots.detector);
    classes  — классы, которые модель распознаёт. Правила «техники нет» проверяют только их:
               если модель не знает каток, отсутствие катка не считается отклонением.
    """
    name: str = "base"
    classes: frozenset = frozenset()

    def detect(self, img: Image.Image, *, tiles: int = 0, sha256: str | None = None) -> DetectResult:
        raise NotImplementedError

    def info(self) -> dict:
        return {"name": self.name, "classes": sorted(self.classes)}
