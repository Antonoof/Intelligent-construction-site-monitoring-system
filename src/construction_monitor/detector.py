"""Обёртка над RF-DETR: загрузка чекпоинта и предсказание на кадрах OpenCV."""
from __future__ import annotations

import logging
import warnings
from pathlib import Path

import cv2
import numpy as np
import supervision as sv
import torch


def pick_device(device: str = "auto") -> str:
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if device.startswith("cuda") and not torch.cuda.is_available():
        print(f"! CUDA недоступна, вместо {device} используется cpu")
        return "cpu"
    return device


class Detector:
    """Детектор строительной техники.

    Классы и разрешение берутся из самого чекпоинта, поэтому при переобучении
    достаточно подменить файл весов.
    """

    def __init__(self, weights: str | Path, device: str = "auto", threshold: float = 0.5,
                 optimize: bool = False, half: bool = False, batch_size: int = 1):
        from rfdetr import RFDETR

        # при загрузке rfdetr пишет, что не грузит веса DINOv2, — для дообученной модели это норма
        logging.getLogger("rf-detr").setLevel(logging.ERROR)

        weights = Path(weights)
        if not weights.exists():
            raise SystemExit(f"не найдены веса {weights}")

        self.device = pick_device(device)
        self.threshold = threshold
        if half and not self.device.startswith("cuda"):
            print("! fp16 поддерживается только на CUDA, используется fp32")
            half = False
        self.half = half

        # device передаём явно: в чекпоинте сохранён device='cuda' с машины, где шло обучение.
        # trust_checkpoint: чекпоинт наш, в нём лежат объекты Lightning, безопасная загрузка их не читает.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FutureWarning)  # про устаревшие варианты моделей
            self.model = RFDETR.from_checkpoint(str(weights), device=self.device,
                                                trust_checkpoint=True)
        self.class_names: list[str] = list(self.model.class_names)
        self.resolution: int = self.model.model_config.resolution

        # при JIT-трассировке граф фиксируется под batch_size — неполный батч добиваем повтором кадра
        self.traced_batch = batch_size if optimize else None
        if optimize or half:
            self.model.inference(compile=optimize, batch_size=batch_size,
                                 dtype=torch.float16 if half else torch.float32)

    def predict(self, frames_bgr: list[np.ndarray]) -> list[sv.Detections]:
        """Кадры в формате OpenCV (BGR) -> список sv.Detections (xyxy в пикселях исходного кадра)."""
        if not frames_bgr:
            return []
        rgb = [cv2.cvtColor(f, cv2.COLOR_BGR2RGB) for f in frames_bgr]
        n = len(rgb)
        if self.traced_batch and n < self.traced_batch:
            rgb += [rgb[-1]] * (self.traced_batch - n)
        out = self.model.predict(rgb, threshold=self.threshold, include_source_image=False)
        if isinstance(out, sv.Detections):
            out = [out]
        return list(out[:n])

    def name(self, class_id: int) -> str:
        return self.class_names[class_id] if 0 <= class_id < len(self.class_names) else str(class_id)

    def describe(self) -> str:
        gpu = f" ({torch.cuda.get_device_name(self.device)})" if self.device.startswith("cuda") else ""
        mode = ", ".join(m for m, on in (("jit", bool(self.traced_batch)), ("fp16", self.half)) if on)
        return (f"RF-DETR {type(self.model).__name__} @ {self.resolution}px на {self.device}{gpu}"
                + (f" [{mode}]" if mode else "")
                + f", порог {self.threshold}, классов {len(self.class_names)}")
