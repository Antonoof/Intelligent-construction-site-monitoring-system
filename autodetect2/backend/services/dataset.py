from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable, Iterator, Sequence

import numpy as np

from ..config import IMAGE_SUFFIXES, LABEL_SUFFIX

ProgressFn = Callable[[str, float], None]

# Label files are tiny but numerous; on a cold cache the wall time is all
# I/O latency, so reads are issued in parallel rather than one at a time.
READ_WORKERS = 24


@dataclass(slots=True)
class Sample:
    """One image and the YOLO label file that belongs to it."""

    name: str
    image: str
    label: str | None
    split: str
    source: int
    boxes: int = 0
    classes: tuple[int, ...] = ()

    def as_dict(self) -> dict:
        return {
            "name": self.name,
            "image": self.image,
            "label": self.label,
            "split": self.split,
            "source": self.source,
            "boxes": self.boxes,
        }


@dataclass
class ScanResult:
    samples: list[Sample] = field(default_factory=list)
    sources: list[dict] = field(default_factory=list)
    class_ids: set[int] = field(default_factory=set)

    @property
    def boxes(self) -> int:
        return sum(s.boxes for s in self.samples)


def iter_images(root: Path) -> Iterator[Path]:
    """os.walk keeps this instant even on datasets with tens of thousands of files."""
    for current, dirs, files in os.walk(root):
        dirs.sort()
        for name in sorted(files):
            if os.path.splitext(name)[1].lower() in IMAGE_SUFFIXES:
                yield Path(current) / name


def label_dirs(image: Path) -> list[Path]:
    """Places a YOLO label can live, most conventional first."""
    parts = list(image.parts)
    candidates: list[Path] = []

    for i in range(len(parts) - 1, -1, -1):
        if parts[i].lower() == "images":
            mirrored = list(parts)
            mirrored[i] = "labels"
            candidates.append(Path(*mirrored).parent)
            break

    parent = image.parent
    return candidates + [parent / "labels", parent.parent / "labels", parent]


def resolve_label(image: Path) -> Path | None:
    stem = image.stem + LABEL_SUFFIX
    for directory in label_dirs(image):
        candidate = directory / stem
        if candidate.is_file():
            return candidate
    return None


class LabelIndex:
    """Directory listings cached by folder — one scandir instead of a stat per image."""

    def __init__(self) -> None:
        self._dirs: dict[str, dict[str, str]] = {}

    def _stems(self, directory: Path) -> dict[str, str]:
        key = str(directory)
        table = self._dirs.get(key)
        if table is None:
            table = {}
            try:
                with os.scandir(directory) as entries:
                    for entry in entries:
                        name = entry.name
                        if name.lower().endswith(LABEL_SUFFIX):
                            table[os.path.splitext(name)[0]] = entry.path
            except (OSError, PermissionError):
                table = {}
            self._dirs[key] = table
        return table

    def resolve(self, image: Path) -> str | None:
        stem = image.stem
        for directory in label_dirs(image):
            hit = self._stems(directory).get(stem)
            if hit:
                return hit
        return None


def read_label(path: Path | str) -> np.ndarray:
    """Returns an (N, 5) array of `cls xc yc w h`; malformed rows are dropped."""
    try:
        with open(path, "rb") as handle:
            raw = handle.read()
    except OSError:
        return np.empty((0, 5), dtype=np.float32)

    rows = []
    for line in raw.splitlines():
        parts = line.split()
        if len(parts) < 5:
            continue
        rows.append(parts[:5])
    if not rows:
        return np.empty((0, 5), dtype=np.float32)

    try:
        return np.array(rows, dtype=np.float32)
    except ValueError:
        return np.empty((0, 5), dtype=np.float32)


def label_summary(path: str | None) -> tuple[int, tuple[int, ...]]:
    """Box count and class ids — everything the index needs, without keeping arrays."""
    if not path:
        return 0, ()
    data = read_label(path)
    if not len(data):
        return 0, ()
    return len(data), tuple(sorted({int(v) for v in data[:, 0]}))


def scan_sources(
    sources: Sequence[dict],
    progress: ProgressFn | None = None,
) -> ScanResult:
    """Walks every configured folder, pairs images with labels and collects classes."""
    result = ScanResult()
    seen: set[str] = set()

    for index, source in enumerate(sources):
        root = Path(source["path"])
        split = source.get("split", "train")
        if not root.is_dir():
            raise FileNotFoundError(f"Папка не найдена: {root}")

        if progress:
            progress(f"{root.name}: поиск изображений", index / max(len(sources), 1))

        images = list(iter_images(root))
        label_index = LabelIndex()
        pending: list[Sample] = []
        label_paths: list[str | None] = []

        for image in images:
            name = image.name
            if name in seen:
                name = f"{image.parent.name}__{image.name}"
                if name in seen:
                    continue
            seen.add(name)

            label = label_index.resolve(image)
            pending.append(
                Sample(
                    name=name,
                    image=str(image),
                    label=label,
                    split=split,
                    source=index,
                )
            )
            label_paths.append(label)

        matched = sum(1 for path in label_paths if path)
        boxes = 0
        total = max(len(pending), 1)

        with ThreadPoolExecutor(max_workers=READ_WORKERS) as pool:
            for pos, (sample, (count, classes)) in enumerate(
                zip(pending, pool.map(label_summary, label_paths, chunksize=32))
            ):
                sample.boxes = count
                sample.classes = classes
                result.class_ids.update(classes)
                boxes += count

                if progress and pos % 500 == 0:
                    share = (index + (pos + 1) / total) / max(len(sources), 1)
                    progress(f"{root.name}: {pos + 1}/{total}", share)

        result.samples.extend(pending)
        result.sources.append(
            {
                "path": str(root),
                "split": split,
                "images": len(images),
                "labels": matched,
                "boxes": boxes,
            }
        )

    if progress:
        progress("Готово", 1.0)
    return result


def class_names(class_ids: Iterable[int], provided: Sequence[str] | None = None) -> list[str]:
    """Список классов проекта.

    Заданный список имён — это истина; выйти за его пределы разметка не может.
    Единственный id 99 в битом файле не должен превращать десятиклассовый проект
    в стоклассовый — такие строки ловит проверка разметки как `bad_class`.
    """
    if provided:
        return [str(name).strip() or f"class_{i}" for i, name in enumerate(provided)]

    ids = sorted(class_ids)
    return [f"class_{i}" for i in range((max(ids) + 1) if ids else 0)]


def xywh_to_xyxy(boxes: np.ndarray) -> np.ndarray:
    if not len(boxes):
        return np.empty((0, 4), dtype=np.float32)
    xc, yc, w, h = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    return np.stack([xc - w / 2, yc - h / 2, xc + w / 2, yc + h / 2], axis=1)


def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Pairwise IoU; normalized and absolute coordinates give identical values."""
    if not len(a) or not len(b):
        return np.zeros((len(a), len(b)), dtype=np.float32)

    x1 = np.maximum(a[:, None, 0], b[None, :, 0])
    y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2])
    y2 = np.minimum(a[:, None, 3], b[None, :, 3])

    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area_a = np.clip(a[:, 2] - a[:, 0], 0, None) * np.clip(a[:, 3] - a[:, 1], 0, None)
    area_b = np.clip(b[:, 2] - b[:, 0], 0, None) * np.clip(b[:, 3] - b[:, 1], 0, None)
    union = area_a[:, None] + area_b[None, :] - inter
    return np.where(union > 0, inter / np.maximum(union, 1e-9), 0.0).astype(np.float32)
