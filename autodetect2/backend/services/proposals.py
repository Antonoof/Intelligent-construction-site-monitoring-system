"""Предразметка: модель расставляет боксы, человек их правит.

Отличается от `submission.py` назначением. Там — результат обученной модели,
который оценивают метриками. Здесь — черновик, который через секунду станет
разметкой: его можно принять целиком по одной кнопке, а плохие кадры дорисовать
руками. Поэтому предложения живут отдельным файлом и не портят статистику.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence

import numpy as np

from .project import Project, utc_now
from . import workflow
from ..config import STATUS_DONE

ProgressFn = Callable[[str, float], None]
CACHE_NAME = "proposals.npz"


@dataclass
class Proposals:
    names: list[str]
    offsets: np.ndarray
    boxes: np.ndarray  # (M, 4) нормированные xyxy
    scores: np.ndarray
    classes: np.ndarray
    model: str
    created_at: str

    def index(self) -> dict[str, int]:
        return {name: i for i, name in enumerate(self.names)}

    def for_image(self, name: str, conf: float = 0.0) -> list[dict]:
        position = self.index().get(name)
        if position is None:
            return []
        start, end = int(self.offsets[position]), int(self.offsets[position + 1])
        out = []
        for i in range(start, end):
            score = float(self.scores[i])
            if score < conf:
                continue
            x1, y1, x2, y2 = (float(v) for v in self.boxes[i])
            out.append(
                {
                    "cls": int(self.classes[i]),
                    "conf": round(score, 4),
                    "x": (x1 + x2) / 2,
                    "y": (y1 + y2) / 2,
                    "w": x2 - x1,
                    "h": y2 - y1,
                    "src": "model",
                }
            )
        return sorted(out, key=lambda item: -item["conf"])


def cache_path(project: Project) -> Path:
    return project.cache_dir / CACHE_NAME


def load(project: Project) -> Proposals | None:
    path = cache_path(project)
    if not path.exists():
        return None
    try:
        data = np.load(path, allow_pickle=True)
        return Proposals(
            names=[str(n) for n in data["names"]],
            offsets=data["offsets"].astype(np.int64),
            boxes=data["boxes"].astype(np.float32),
            scores=data["scores"].astype(np.float32),
            classes=data["classes"].astype(np.int32),
            model=str(data["model"]) if "model" in data.files else "",
            created_at=str(data["created_at"]) if "created_at" in data.files else "",
        )
    except (OSError, KeyError, ValueError):
        return None


def forget(project: Project) -> None:
    cache_path(project).unlink(missing_ok=True)
    project.meta.pop("proposals", None)
    project.save()


def _store(project: Project, rows: dict[str, list[tuple[int, float, float, float, float, float]]], model: str) -> Proposals:
    names = sorted(rows)
    counts = [len(rows[name]) for name in names]
    offsets = np.zeros(len(names) + 1, dtype=np.int64)
    np.cumsum(counts, out=offsets[1:])

    total = int(offsets[-1])
    boxes = np.zeros((total, 4), dtype=np.float32)
    scores = np.zeros(total, dtype=np.float32)
    classes = np.zeros(total, dtype=np.int32)

    cursor = 0
    for name in names:
        for cls, score, x1, y1, x2, y2 in rows[name]:
            boxes[cursor] = (x1, y1, x2, y2)
            scores[cursor] = score
            classes[cursor] = cls
            cursor += 1

    created = utc_now()
    project.cache_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        cache_path(project),
        names=np.array(names),
        offsets=offsets,
        boxes=boxes,
        scores=scores,
        classes=classes,
        model=model,
        created_at=created,
    )
    project.meta["proposals"] = {
        "model": model,
        "at": created,
        "images": len(names),
        "boxes": total,
    }
    project.save()
    return Proposals(names, offsets, boxes, scores, classes, model, created)


def run(
    project: Project,
    model_path: Path,
    images: Sequence[dict],
    conf: float = 0.25,
    iou: float = 0.6,
    imgsz: int = 960,
    device: str = "auto",
    progress: ProgressFn | None = None,
) -> Proposals:
    """Прогоняет модель по кадрам и складывает предложения в кэш проекта."""
    from ultralytics import YOLO

    if not images:
        raise ValueError("Нет изображений для предразметки")

    if progress:
        progress(f"Загрузка {model_path.name}", 0.05)
    model = YOLO(str(model_path))
    kwargs = {"conf": conf, "iou": iou, "imgsz": imgsz, "verbose": False}
    if device != "auto":
        kwargs["device"] = device

    rows: dict[str, list[tuple[int, float, float, float, float, float]]] = {}
    batch = 16
    total = len(images)

    for start in range(0, total, batch):
        chunk = images[start : start + batch]
        results = model.predict([s["image"] for s in chunk], **kwargs)
        for sample, result in zip(chunk, results):
            found = []
            boxes = getattr(result, "boxes", None)
            if boxes is not None and len(boxes):
                height, width = result.orig_shape
                xyxy = boxes.xyxy.cpu().numpy()
                confs = boxes.conf.cpu().numpy()
                classes = boxes.cls.cpu().numpy().astype(int)
                for (x1, y1, x2, y2), score, cls in zip(xyxy, confs, classes):
                    found.append(
                        (
                            int(cls),
                            float(score),
                            float(np.clip(x1 / width, 0.0, 1.0)),
                            float(np.clip(y1 / height, 0.0, 1.0)),
                            float(np.clip(x2 / width, 0.0, 1.0)),
                            float(np.clip(y2 / height, 0.0, 1.0)),
                        )
                    )
            rows[sample["name"]] = found

        if progress:
            done = min(start + batch, total)
            progress(f"Предразметка: {done}/{total}", 0.05 + 0.9 * done / total)

    result = _store(project, rows, model_path.name)
    if progress:
        progress(f"Готово: {len(result.boxes)} боксов на {len(result.names)} кадрах", 1.0)
    return result


def apply(
    project: Project,
    names: Sequence[str] | None = None,
    conf: float = 0.5,
    only_unlabeled: bool = True,
    mark_done: bool = False,
) -> dict:
    """Превращает предложения в настоящую разметку.

    По умолчанию трогает только кадры без разметки: перезаписать ручную работу
    автоматическим черновиком — самый дорогой способ потерять день.
    """
    store = load(project)
    if store is None:
        raise ValueError("Предразметка ещё не выполнялась")

    known = project.by_name()
    wanted = list(names) if names else list(store.names)
    project.annotations_dir.mkdir(parents=True, exist_ok=True)

    written, boxes, skipped = 0, 0, 0
    touched: list[str] = []

    for name in wanted:
        sample = known.get(name)
        if sample is None:
            continue
        if only_unlabeled and (sample.get("boxes") or project.label_path(sample) is not None):
            skipped += 1
            continue

        found = store.for_image(name, conf)
        lines = [
            f"{item['cls']} {item['x']:.6f} {item['y']:.6f} {item['w']:.6f} {item['h']:.6f}"
            for item in found
            if item["w"] > 0 and item["h"] > 0
        ]
        body = "\n".join(lines)
        (project.annotations_dir / f"{Path(name).stem}.txt").write_text(
            body + ("\n" if body else ""), encoding="utf-8"
        )
        written += 1
        boxes += len(lines)
        touched.append(name)

    if mark_done and touched:
        with workflow.transaction(project) as state:
            state.set(touched, STATUS_DONE, source="prelabel")

    for cache in ("dashboard.json", "heatmap.json", "quality.json", "ground_truth.npz"):
        (project.cache_dir / cache).unlink(missing_ok=True)

    return {"images": written, "boxes": boxes, "skipped": skipped, "conf": conf}
