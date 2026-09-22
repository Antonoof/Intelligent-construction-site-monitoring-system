from __future__ import annotations

import csv
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .dataset import xywh_to_xyxy
from .project import Project, utc_now

log = logging.getLogger(__name__)

IMAGE_KEYS = ("image_id", "image", "image_name", "img", "name", "file", "filename", "id")
CLASS_KEYS = ("class_id", "class", "cls", "category_id", "category", "label")
SCORE_KEYS = ("confidence", "conf", "score", "prob", "probability")
XYWH_KEYS = (("x_center", "y_center", "width", "height"), ("xc", "yc", "w", "h"), ("x", "y", "w", "h"))
XYXY_KEYS = (
    ("x_min", "y_min", "x_max", "y_max"),
    ("xmin", "ymin", "xmax", "ymax"),
    ("x1", "y1", "x2", "y2"),
    ("left", "top", "right", "bottom"),
)
BLOB_KEYS = ("prediction", "predictions", "boxes", "bboxes", "annotation", "labels", "pred")
SIZE_KEYS = (("img_w", "img_h"), ("image_width", "image_height"), ("img_width", "img_height"))

# Exports often carry ground truth and predictions in one file, told apart by a
# marker column. Ground truth rows must never be scored as if the model made them.
KIND_KEYS = ("source", "kind", "type", "row_type", "origin")
PRED_VALUES = {"pred", "preds", "prediction", "predictions", "predict", "model", "dt", "detection", "det"}
GT_VALUES = {"gt", "ground_truth", "groundtruth", "label", "labels", "true", "target", "annotation"}

CACHE_VERSION = 2


@dataclass
class ImagePrediction:
    boxes: np.ndarray = field(default_factory=lambda: np.empty((0, 4), dtype=np.float32))
    scores: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.float32))
    classes: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.int32))


@dataclass
class Submission:
    by_image: dict[str, ImagePrediction]
    layout: str
    rows: int
    matched: int
    unmatched: list[str]
    skipped: dict[str, int] = field(default_factory=dict)

    @property
    def images(self) -> int:
        return len(self.by_image)

    @property
    def boxes(self) -> int:
        return int(sum(len(p.boxes) for p in self.by_image.values()))


def _index(header: dict[str, int], keys: tuple[str, ...]) -> int | None:
    for key in keys:
        if key in header:
            return header[key]
    return None


def _index_group(header: dict[str, int], groups: tuple[tuple[str, ...], ...]) -> list[int] | None:
    for group in groups:
        if all(key in header for key in group):
            return [header[key] for key in group]
    return None


def _parse_blob(text: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Kaggle-style packed predictions: `cls conf xc yc w h ...` or `conf xc yc w h ...`."""
    parts = text.replace(",", " ").split()
    if not parts:
        return np.empty((0, 4), np.float32), np.empty(0, np.float32), np.empty(0, np.int32)

    stride = 6 if len(parts) % 6 == 0 else 5
    values = np.asarray(parts[: len(parts) // stride * stride], dtype=np.float32).reshape(-1, stride)
    if stride == 6:
        classes = values[:, 0].astype(np.int32)
        scores = values[:, 1]
        xywh = values[:, 2:6]
    else:
        classes = np.zeros(len(values), dtype=np.int32)
        scores = values[:, 0]
        xywh = values[:, 1:5]
    return xywh_to_xyxy(xywh), scores, classes


@dataclass
class ParseResult:
    predictions: dict[str, ImagePrediction]
    layout: str
    rows: int
    skipped: dict[str, int]
    sizes: dict[str, tuple[float, float]]


def parse_csv(path: Path) -> ParseResult:
    """Reads a submission file into per-image arrays.

    Rows are streamed with `csv.reader` instead of `DictReader`: a full-dataset
    export runs to millions of lines and building a dict per row dominates the cost.
    """
    skipped = {"ground_truth": 0, "malformed": 0, "no_detection": 0}
    buckets: dict[str, list[tuple[list[float], float, int]]] = {}
    sizes: dict[str, tuple[float, float]] = {}
    rows = 0

    with open(path, "r", encoding="utf-8-sig", newline="") as handle:
        sample = handle.read(8192)
        handle.seek(0)
        try:
            dialect = csv.Sniffer().sniff(sample, delimiters=",;\t")
        except csv.Error:
            dialect = csv.excel

        reader = csv.reader(handle, dialect)
        try:
            raw_header = next(reader)
        except StopIteration as exc:
            raise ValueError("Пустой CSV") from exc

        header = {name.strip().lower(): i for i, name in enumerate(raw_header) if name.strip()}
        if not header:
            raise ValueError("В CSV нет заголовка")

        image_idx = _index(header, IMAGE_KEYS) or 0
        class_idx = _index(header, CLASS_KEYS)
        score_idx = _index(header, SCORE_KEYS)
        kind_idx = _index(header, KIND_KEYS)
        blob_idx = _index(header, BLOB_KEYS)
        size_idx = _index_group(header, SIZE_KEYS)
        xywh_idx = _index_group(header, XYWH_KEYS)
        xyxy_idx = _index_group(header, XYXY_KEYS)

        if xywh_idx:
            layout, box_idx = "xywh", xywh_idx
        elif xyxy_idx:
            layout, box_idx = "xyxy", xyxy_idx
        elif blob_idx is not None:
            layout, box_idx = "packed", None
        else:
            raise ValueError(
                "Не удалось распознать колонки с боксами. Ожидаются "
                "x_center/y_center/width/height, x_min/y_min/x_max/y_max "
                "или упакованная колонка prediction"
            )

        columns = len(raw_header)
        for row in reader:
            if len(row) < columns:
                skipped["malformed"] += 1
                continue

            if kind_idx is not None:
                kind = row[kind_idx].strip().lower()
                if kind in GT_VALUES:
                    skipped["ground_truth"] += 1
                    continue
                if kind and kind not in PRED_VALUES:
                    skipped["malformed"] += 1
                    continue

            key = row[image_idx].strip()
            if not key:
                skipped["malformed"] += 1
                continue

            bucket = buckets.setdefault(key, [])
            if size_idx and key not in sizes:
                try:
                    sizes[key] = (float(row[size_idx[0]]), float(row[size_idx[1]]))
                except ValueError:
                    pass

            if layout == "packed":
                boxes, scores, classes = _parse_blob(row[blob_idx])
                for box, score, cls in zip(boxes, scores, classes):
                    bucket.append((box.tolist(), float(score), int(cls)))
                rows += 1
                continue

            try:
                cls = int(float(row[class_idx])) if class_idx is not None and row[class_idx] else 0
                values = [float(row[i]) for i in box_idx]
            except (TypeError, ValueError):
                # A row with no coordinates marks "nothing found here": keep the
                # image in the submission, but do not invent a box for it.
                skipped["no_detection"] += 1
                continue

            if cls < 0 or (layout == "xywh" and (values[2] <= 0 or values[3] <= 0)):
                skipped["no_detection"] += 1
                continue

            try:
                score = float(row[score_idx]) if score_idx is not None and row[score_idx] else 1.0
            except ValueError:
                score = 1.0

            box = xywh_to_xyxy(np.asarray([values], dtype=np.float32))[0].tolist() if layout == "xywh" else values
            bucket.append((box, score, cls))
            rows += 1

    predictions: dict[str, ImagePrediction] = {}
    for key, items in buckets.items():
        if not items:
            predictions[key] = ImagePrediction()
            continue
        predictions[key] = ImagePrediction(
            boxes=np.asarray([i[0] for i in items], dtype=np.float32),
            scores=np.asarray([i[1] for i in items], dtype=np.float32),
            classes=np.asarray([i[2] for i in items], dtype=np.int32),
        )

    filled = [p.boxes for p in predictions.values() if len(p.boxes)]
    if filled and float(np.nanmax(np.concatenate(filled))) > 1.5:
        layout += "-absolute"

    return ParseResult(predictions, layout, rows, skipped, sizes)


def _align_names(project: Project, raw: dict[str, ImagePrediction]) -> tuple[dict, list[str]]:
    names = [s["name"] for s in project.samples()]
    by_exact = {n: n for n in names}
    by_stem = {Path(n).stem.lower(): n for n in names}

    aligned: dict[str, ImagePrediction] = {}
    unmatched: list[str] = []
    for key, prediction in raw.items():
        resolved = by_exact.get(key) or by_stem.get(Path(key).stem.lower())
        if resolved is None:
            unmatched.append(key)
            continue
        aligned[resolved] = prediction
    return aligned, unmatched


def _scale_absolute(
    project: Project,
    predictions: dict[str, ImagePrediction],
    sizes: dict[str, tuple[float, float]],
) -> None:
    """Converts pixel coordinates to normalized ones.

    Sizes declared in the file are trusted first — reading tens of thousands of
    image headers just to learn what the CSV already says is pure waste.
    """
    from PIL import Image

    by_stem = {Path(key).stem.lower(): value for key, value in sizes.items()}
    lookup = {s["name"]: s["image"] for s in project.samples()}

    for name, prediction in predictions.items():
        if not len(prediction.boxes):
            continue

        size = sizes.get(name) or by_stem.get(Path(name).stem.lower())
        if size is None:
            try:
                with Image.open(lookup[name]) as img:
                    size = img.size
            except (OSError, KeyError):
                continue

        width, height = size
        prediction.boxes[:, [0, 2]] /= max(width, 1)
        prediction.boxes[:, [1, 3]] /= max(height, 1)


# --------------------------------------------------------------------- cache
def cache_path(project: Project) -> Path:
    return project.cache_dir / "submission.npz"


def _signature(path: Path) -> str:
    stat = path.stat()
    return f"{CACHE_VERSION}:{path.resolve()}:{stat.st_size}:{int(stat.st_mtime)}"


def _store_cache(project: Project, path: Path, submission: Submission) -> None:
    """Flattens the per-image arrays into one block so reloads are a single read."""
    names = list(submission.by_image)
    counts = [len(submission.by_image[name].boxes) for name in names]
    offsets = np.zeros(len(names) + 1, dtype=np.int64)
    np.cumsum(counts, out=offsets[1:])

    empty = ImagePrediction()
    boxes = [submission.by_image[n].boxes for n in names] or [empty.boxes]
    scores = [submission.by_image[n].scores for n in names] or [empty.scores]
    classes = [submission.by_image[n].classes for n in names] or [empty.classes]

    project.cache_dir.mkdir(parents=True, exist_ok=True)
    np.savez(
        cache_path(project),
        signature=_signature(path),
        names=np.array(names),
        offsets=offsets,
        boxes=np.concatenate(boxes) if boxes else np.empty((0, 4), np.float32),
        scores=np.concatenate(scores) if scores else np.empty(0, np.float32),
        classes=np.concatenate(classes) if classes else np.empty(0, np.int32),
        meta=json.dumps(
            {
                "layout": submission.layout,
                "rows": submission.rows,
                "matched": submission.matched,
                "unmatched": submission.unmatched,
                "skipped": submission.skipped,
            }
        ),
    )


def _read_cache(project: Project, path: Path) -> Submission | None:
    target = cache_path(project)
    if not target.exists():
        return None
    try:
        data = np.load(target, allow_pickle=False)
        if str(data["signature"]) != _signature(path):
            return None

        names = [str(n) for n in data["names"]]
        offsets = data["offsets"]
        boxes, scores, classes = data["boxes"], data["scores"], data["classes"]
        meta = json.loads(str(data["meta"]))
    except (OSError, KeyError, ValueError, json.JSONDecodeError):
        return None

    by_image = {
        name: ImagePrediction(
            boxes=boxes[offsets[i] : offsets[i + 1]],
            scores=scores[offsets[i] : offsets[i + 1]],
            classes=classes[offsets[i] : offsets[i + 1]],
        )
        for i, name in enumerate(names)
    }
    return Submission(
        by_image=by_image,
        layout=meta.get("layout", "cache"),
        rows=int(meta.get("rows", 0)),
        matched=int(meta.get("matched", len(by_image))),
        unmatched=list(meta.get("unmatched", [])),
        skipped=dict(meta.get("skipped", {})),
    )


_MEMO: dict[str, tuple[str, Submission]] = {}


def forget(project: Project) -> None:
    """Drops both the in-process and on-disk copies of a parsed submission."""
    _MEMO.pop(project.id, None)
    cache_path(project).unlink(missing_ok=True)


def load(project: Project) -> Submission | None:
    """Submission for a project, reusing the parsed form across requests."""
    path = project.submission_path
    if path is None:
        return None

    signature = _signature(path)
    cached = _MEMO.get(project.id)
    if cached and cached[0] == signature:
        return cached[1]

    submission = _read_cache(project, path)
    if submission is None:
        submission = load_from_path(project, path)
        _store_cache(project, path, submission)

    _MEMO[project.id] = (signature, submission)
    return submission


def load_from_path(project: Project, path: Path) -> Submission:
    result = parse_csv(path)
    aligned, unmatched = _align_names(project, result.predictions)
    if result.layout.endswith("-absolute"):
        _scale_absolute(project, aligned, result.sizes)

    return Submission(
        by_image=aligned,
        layout=result.layout,
        rows=result.rows,
        matched=len(aligned),
        unmatched=unmatched[:20],
        skipped={k: v for k, v in result.skipped.items() if v},
    )


def register(project: Project, path: Path) -> Submission:
    submission = load_from_path(project, path)
    project.meta["submission"] = {
        "path": str(path),
        "file": path.name,
        "rows": submission.rows,
        "images": submission.images,
        "boxes": submission.boxes,
        "layout": submission.layout,
        "matched": submission.matched,
        "skipped": submission.skipped,
        "uploaded_at": utc_now(),
    }
    project.save()

    _MEMO.pop(project.id, None)
    _store_cache(project, path, submission)
    for name in ("analysis.json", "dashboard.json", "heatmap.json"):
        (project.cache_dir / name).unlink(missing_ok=True)
    return submission
