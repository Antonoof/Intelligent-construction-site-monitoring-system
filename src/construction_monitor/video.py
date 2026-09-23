"""Поиск видео по glob-шаблонам, чтение кадров и запись результата."""
from __future__ import annotations

import glob
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import cv2
import numpy as np

from construction_monitor.config import resolve_path

DEFAULT_VIDEO_DIR = "data/videos"
VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".mpg", ".mpeg", ".wmv", ".m4v", ".webm"}


def _expand(source: str) -> list[str]:
    """Файл, папка или glob-шаблон. Путь ищется от текущей папки, затем от корня репозитория,
    затем в data/videos (так можно передать просто имя файла)."""
    for base in (Path(source).expanduser(), resolve_path(source),
                 resolve_path(DEFAULT_VIDEO_DIR) / "**" / source):
        if base.is_dir():
            return glob.glob(str(base / "**" / "*"), recursive=True)
        hits = glob.glob(str(base), recursive=True)
        if hits:
            return hits
    return []


def find_videos(sources: list[str]) -> list[Path]:
    """Все видео по файлам, папкам и glob-шаблонам, без дублей, в стабильном порядке."""
    found: dict[Path, None] = {}
    for source in sources:
        for p in sorted(_expand(source)):
            path = Path(p).resolve()
            if path.is_file() and path.suffix.lower() in VIDEO_EXTS:
                found.setdefault(path, None)
    return list(found)


@dataclass
class VideoInfo:
    path: Path
    fps: float
    width: int
    height: int
    frames: int

    @property
    def duration_s(self) -> float:
        return self.frames / self.fps if self.fps else 0.0


def open_video(path: Path) -> tuple[cv2.VideoCapture, VideoInfo]:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"не удалось открыть видео {path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    info = VideoInfo(path, fps,
                     int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                     int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                     int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
    return cap, info


def iter_frames(cap: cv2.VideoCapture, stride: int = 1) -> Iterator[tuple[int, np.ndarray]]:
    """(номер кадра, кадр BGR). Пропущенные кадры не декодируются (grab без retrieve)."""
    idx = 0
    while True:
        if idx % stride:
            if not cap.grab():
                return
        else:
            ok, frame = cap.read()
            if not ok:
                return
            yield idx, frame
        idx += 1


def batched(it: Iterator, n: int) -> Iterator[list]:
    batch = []
    for x in it:
        batch.append(x)
        if len(batch) == n:
            yield batch
            batch = []
    if batch:
        yield batch


def make_writer(path: Path, fps: float, width: int, height: int) -> cv2.VideoWriter:
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"не удалось создать видео {path}")
    return writer
