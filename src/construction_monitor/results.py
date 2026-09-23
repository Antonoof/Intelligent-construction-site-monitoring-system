"""Отрисовка детекций и сохранение результатов.

detections.csv — покадровый лог (что, где и когда видно), из него дальше строится
сопоставление с календарным планом. summary.json — сводка присутствия техники по видео.
"""
from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import supervision as sv

CSV_FIELDS = ["video", "frame", "time_s", "class_id", "class_name", "confidence",
              "x1", "y1", "x2", "y2"]


# Постоянный цвет у каждого класса — одинаковый на видео, таймлайнах и графиках.
# Классы, которые часто видны вместе (экскаватор, самосвал, погрузчик), разнесены по цветам
# так, чтобы их различали и люди с нарушением цветовосприятия; к тому же у каждого бокса есть подпись.
CLASS_COLORS = {
    "Excavator": "#2a78d6",
    "Dump truck": "#eb6834",
    "Bucket loader": "#1baf7a",
    "Crane manipulator": "#eda100",
    "Autocran": "#4a3aa7",
    "Mixer": "#e87ba4",
    "Bulldozer": "#e34948",
    "Motor grader": "#008300",
    "Tower crane": "#9085e9",
    "Drilling rig": "#8a5a2b",
}
FALLBACK_COLORS = ["#52514e", "#8a8984"]


def class_color(name: str, idx: int = 0) -> str:
    return CLASS_COLORS.get(name, FALLBACK_COLORS[idx % len(FALLBACK_COLORS)])


def _text_on(hex_color: str) -> str:
    """Чёрный текст на светлой плашке, белый — на тёмной."""
    r, g, b = (int(hex_color[i:i + 2], 16) / 255 for i in (1, 3, 5))
    lum = 0.2126 * r ** 2.2 + 0.7152 * g ** 2.2 + 0.0722 * b ** 2.2
    return "#0b0b0b" if lum > 0.3 else "#ffffff"


class Annotator:
    """Рамки и подписи; толщина и размер шрифта подстраиваются под разрешение кадра."""

    def __init__(self, class_names: list[str]):
        self.class_names = class_names
        colors = [class_color(n, i) for i, n in enumerate(class_names)]
        self.palette = sv.ColorPalette.from_hex(colors)
        self.text_palette = sv.ColorPalette.from_hex([_text_on(c) for c in colors])
        self._by_width: dict[int, tuple[sv.BoxAnnotator, sv.LabelAnnotator]] = {}

    def _annotators(self, width: int):
        if width not in self._by_width:
            k = width / 1280
            self._by_width[width] = (
                sv.BoxAnnotator(color=self.palette, thickness=max(2, round(3 * k))),
                sv.LabelAnnotator(color=self.palette, text_color=self.text_palette,
                                  text_scale=0.6 * k, text_thickness=max(1, round(1.5 * k)),
                                  text_padding=max(4, round(7 * k))),
            )
        return self._by_width[width]

    def __call__(self, frame: np.ndarray, det: sv.Detections) -> np.ndarray:
        box, label = self._annotators(frame.shape[1])
        labels = [f"{self.class_names[c] if c < len(self.class_names) else c} {s:.2f}"
                  for c, s in zip(det.class_id, det.confidence)]
        out = box.annotate(frame.copy(), det)
        return label.annotate(out, det, labels=labels)


class VideoReport:
    """Копит детекции одного видео: пишет CSV построчно и считает сводку по классам."""

    def __init__(self, video_name: str, fps: float, stride: int, class_names: list[str],
                 csv_path: Path | None):
        self.video_name = video_name
        self.fps = fps
        self.stride = stride
        self.class_names = class_names
        self.frames_processed = 0
        self.frames_with = defaultdict(int)      # в скольких обработанных кадрах класс есть
        self.max_count = defaultdict(int)        # максимум единиц класса в одном кадре
        self.conf_sum = defaultdict(float)
        self.boxes = defaultdict(int)
        self.first_s: dict[int, float] = {}
        self.last_s: dict[int, float] = {}

        self._file = self._csv = None
        if csv_path:
            csv_path.parent.mkdir(parents=True, exist_ok=True)
            self._file = csv_path.open("w", newline="", encoding="utf-8")
            self._csv = csv.writer(self._file)
            self._csv.writerow(CSV_FIELDS)

    def add(self, frame_idx: int, det: sv.Detections) -> None:
        self.frames_processed += 1
        t = frame_idx / self.fps if self.fps else 0.0
        counts = defaultdict(int)
        for (x1, y1, x2, y2), c, s in zip(det.xyxy, det.class_id, det.confidence):
            c = int(c)
            counts[c] += 1
            self.conf_sum[c] += float(s)
            self.boxes[c] += 1
            if self._csv:
                self._csv.writerow([self.video_name, frame_idx, f"{t:.3f}", c, self._name(c),
                                    f"{s:.4f}", f"{x1:.1f}", f"{y1:.1f}", f"{x2:.1f}", f"{y2:.1f}"])
        for c, n in counts.items():
            self.frames_with[c] += 1
            self.max_count[c] = max(self.max_count[c], n)
            self.first_s.setdefault(c, t)
            self.last_s[c] = t

    def summary(self) -> dict:
        step_s = self.stride / self.fps if self.fps else 0.0
        per_class = {}
        for c in sorted(self.frames_with):
            per_class[self._name(c)] = {
                "frames_present": self.frames_with[c],
                "presence_ratio": round(self.frames_with[c] / max(self.frames_processed, 1), 4),
                "seconds_present": round(self.frames_with[c] * step_s, 2),
                "first_seen_s": round(self.first_s[c], 2),
                "last_seen_s": round(self.last_s[c], 2),
                "max_simultaneous": self.max_count[c],
                "mean_confidence": round(self.conf_sum[c] / self.boxes[c], 4),
            }
        return {"video": self.video_name, "frames_processed": self.frames_processed,
                "frame_stride": self.stride, "fps": self.fps, "classes": per_class}

    def close(self, summary_path: Path | None = None) -> dict:
        if self._file:
            self._file.close()
        s = self.summary()
        if summary_path:
            summary_path.parent.mkdir(parents=True, exist_ok=True)
            summary_path.write_text(json.dumps(s, ensure_ascii=False, indent=2), encoding="utf-8")
        return s

    def _name(self, c: int) -> str:
        return self.class_names[c] if 0 <= c < len(self.class_names) else str(c)


def unique_dir(root: Path, name: str, used: set[str]) -> Path:
    """Папка результатов для видео; одинаковые имена из разных подпапок не перетирают друг друга."""
    candidate, i = name, 1
    while candidate in used:
        candidate = f"{name}_{i}"
        i += 1
    used.add(candidate)
    return root / candidate


def write_summary_table(out_root: Path) -> Path | None:
    """summary.csv по всем видео в папке результатов, а не только по текущему запуску."""
    rows = []
    for f in sorted(out_root.glob("*/summary.json")):
        s = json.loads(f.read_text(encoding="utf-8"))
        for name, c in s["classes"].items():
            rows.append({"video": s["video"], "class_name": name, **c})
    if not rows:
        return None
    path = out_root / "summary.csv"
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    return path
