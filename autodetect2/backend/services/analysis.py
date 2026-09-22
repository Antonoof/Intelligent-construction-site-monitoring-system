from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

from .dataset import READ_WORKERS, iou_matrix, read_label, xywh_to_xyxy
from .project import Project
from .submission import ImagePrediction, Submission

IOU_SWEEP = np.round(np.arange(0.5, 1.0, 0.05), 2)
CONFIDENT = 0.5
AMBIGUOUS = (0.15, 0.45)
EMPTY_PRED = ImagePrediction()


@dataclass
class GroundTruth:
    boxes: np.ndarray
    classes: np.ndarray


EMPTY_GT = GroundTruth(np.empty((0, 4), np.float32), np.empty(0, np.int32))


GT_CACHE_VERSION = 1
_GT_MEMO: dict[str, tuple[str, dict[str, "GroundTruth"]]] = {}


def _gt_signature(project: Project) -> str:
    """Changes whenever the index or any hand-edited label changes."""
    index_mtime = int(project.index_path.stat().st_mtime) if project.index_path.exists() else 0
    edited = list(project.annotations_dir.glob("*.txt")) if project.annotations_dir.exists() else []
    newest = max((int(p.stat().st_mtime) for p in edited), default=0)
    return f"{GT_CACHE_VERSION}:{index_mtime}:{len(edited)}:{newest}"


def _gt_cache_path(project: Project) -> Path:
    return project.cache_dir / "ground_truth.npz"


def _read_gt_cache(project: Project, signature: str) -> dict[str, GroundTruth] | None:
    path = _gt_cache_path(project)
    if not path.exists():
        return None
    try:
        data = np.load(path, allow_pickle=False)
        if str(data["signature"]) != signature:
            return None
        names = [str(n) for n in data["names"]]
        offsets, boxes, classes = data["offsets"], data["boxes"], data["classes"]
    except (OSError, KeyError, ValueError):
        return None

    return {
        name: GroundTruth(
            boxes=boxes[offsets[i] : offsets[i + 1]],
            classes=classes[offsets[i] : offsets[i + 1]],
        )
        for i, name in enumerate(names)
    }


def _store_gt_cache(project: Project, signature: str, truth: dict[str, GroundTruth]) -> None:
    names = list(truth)
    counts = [len(truth[name].boxes) for name in names]
    offsets = np.zeros(len(names) + 1, dtype=np.int64)
    np.cumsum(counts, out=offsets[1:])

    project.cache_dir.mkdir(parents=True, exist_ok=True)
    np.savez(
        _gt_cache_path(project),
        signature=signature,
        names=np.array(names),
        offsets=offsets,
        boxes=np.concatenate([truth[n].boxes for n in names]) if names else np.empty((0, 4), np.float32),
        classes=np.concatenate([truth[n].classes for n in names]) if names else np.empty(0, np.int32),
    )


def load_ground_truth(project: Project) -> dict[str, GroundTruth]:
    """Ground truth for every labelled image.

    A dataset-sized project means tens of thousands of tiny label files, and on a
    cold cache that is minutes of pure I/O latency. Reads run in parallel and the
    result is kept in a single array file, so only the first call pays for it.
    """
    signature = _gt_signature(project)
    memo = _GT_MEMO.get(project.id)
    if memo and memo[0] == signature:
        return memo[1]

    cached = _read_gt_cache(project, signature)
    if cached is not None:
        _GT_MEMO[project.id] = (signature, cached)
        return cached

    entries = []
    for sample in project.samples():
        label = project.label_path(sample)
        if label is not None:
            entries.append((sample["name"], label))

    truth: dict[str, GroundTruth] = {}
    with ThreadPoolExecutor(max_workers=READ_WORKERS) as pool:
        for (name, _), data in zip(entries, pool.map(read_label, [path for _, path in entries])):
            truth[name] = GroundTruth(
                boxes=xywh_to_xyxy(data[:, 1:5]) if len(data) else np.empty((0, 4), np.float32),
                classes=data[:, 0].astype(np.int32) if len(data) else np.empty(0, np.int32),
            )

    _store_gt_cache(project, signature, truth)
    _GT_MEMO[project.id] = (signature, truth)
    return truth


def greedy_match(iou: np.ndarray, order: np.ndarray, threshold: float) -> tuple[np.ndarray, np.ndarray]:
    """Greedy highest-score-first matching. Returns (pred -> gt index, gt matched mask)."""
    n_pred, n_gt = iou.shape
    pred_match = np.full(n_pred, -1, dtype=np.int32)
    gt_taken = np.zeros(n_gt, dtype=bool)
    if not n_pred or not n_gt:
        return pred_match, gt_taken

    for p in order:
        candidates = np.where(~gt_taken, iou[p], -1.0)
        best = int(np.argmax(candidates))
        if candidates[best] >= threshold:
            pred_match[p] = best
            gt_taken[best] = True
    return pred_match, gt_taken


def average_precision(scores: np.ndarray, matched: np.ndarray, n_gt: int) -> float:
    """101-point interpolated AP (COCO convention)."""
    if n_gt == 0 or not len(scores):
        return 0.0
    order = np.argsort(-scores)
    matched = matched[order].astype(bool)
    tp = np.cumsum(matched)
    fp = np.cumsum(~matched)
    recall = tp / n_gt
    precision = tp / np.maximum(tp + fp, 1e-9)
    precision = np.maximum.accumulate(precision[::-1])[::-1]

    grid = np.linspace(0, 1, 101)
    idx = np.searchsorted(recall, grid, side="left")
    values = np.where(idx < len(precision), precision[np.clip(idx, 0, len(precision) - 1)], 0.0)
    return float(values.mean())


# --------------------------------------------------------------- per image
@dataclass
class ImageReport:
    name: str
    split: str
    n_gt: int
    n_pred: int
    tp: int
    fp: int
    fn: int
    ghost: int  # confident prediction with no ground truth anywhere near it
    missed: int  # ground truth the model never proposed
    loc: int  # right place, not enough overlap
    ambiguous: int  # predictions inside the uncertainty band
    conf_max: float
    conf_mean: float
    iou_mean: float

    def as_dict(self) -> dict:
        payload = self.__dict__.copy()
        for key in ("conf_max", "conf_mean", "iou_mean"):
            payload[key] = round(payload[key], 4)
        return payload


@dataclass
class MatchResult:
    """Matching is done once over every prediction; thresholds are applied afterwards.

    Greedy matching walks predictions in descending score, so dropping low-score
    predictions never changes what the high-score ones matched. That makes a single
    pass sufficient for every confidence threshold we report.
    """

    reports: list[ImageReport]
    scores: np.ndarray
    matched: np.ndarray
    total_gt: int


def match_all(
    project: Project,
    truth: dict[str, GroundTruth],
    submission: Submission,
    iou_thr: float = 0.5,
    conf_eval: float = 0.25,
) -> MatchResult:
    reports: list[ImageReport] = []
    all_scores: list[np.ndarray] = []
    all_matched: list[np.ndarray] = []
    total_gt = 0

    for sample in project.samples():
        name = sample["name"]
        gt = truth.get(name, EMPTY_GT)
        pred = submission.by_image.get(name, EMPTY_PRED)
        boxes, scores, classes = pred.boxes, pred.scores, pred.classes
        total_gt += len(gt.boxes)

        iou = iou_matrix(boxes, gt.boxes)
        if iou.size:
            iou = np.where(classes[:, None] == gt.classes[None, :], iou, 0.0)

        order = np.argsort(-scores) if len(scores) else np.empty(0, dtype=int)
        pred_match, _ = greedy_match(iou, order, iou_thr)
        is_matched = pred_match >= 0

        all_scores.append(scores)
        all_matched.append(is_matched)

        best_iou = iou.max(axis=1) if iou.size else np.zeros(len(boxes))
        gt_best = iou.max(axis=0) if iou.size else np.zeros(len(gt.boxes))

        keep = scores >= conf_eval if len(scores) else np.zeros(0, dtype=bool)
        tp = int((is_matched & keep).sum())
        n_pred = int(keep.sum())
        gt_covered = np.zeros(len(gt.boxes), dtype=bool)
        if iou.size:
            for p in np.where(is_matched & keep)[0]:
                gt_covered[pred_match[p]] = True

        ghost_mask = (~is_matched) & keep & (scores >= CONFIDENT) & (best_iou < 0.1)
        matched_iou = best_iou[is_matched & keep] if len(boxes) else np.empty(0)

        reports.append(
            ImageReport(
                name=name,
                split=sample.get("split", "train"),
                n_gt=int(len(gt.boxes)),
                n_pred=n_pred,
                tp=tp,
                fp=n_pred - tp,
                fn=int(len(gt.boxes) - tp),
                ghost=int(ghost_mask.sum()),
                missed=int(((~gt_covered) & (gt_best < 0.1)).sum()),
                loc=int(((~is_matched) & keep & (best_iou >= 0.1)).sum()),
                ambiguous=int(((scores >= AMBIGUOUS[0]) & (scores <= AMBIGUOUS[1])).sum()),
                conf_max=float(scores.max()) if len(scores) else 0.0,
                conf_mean=float(scores.mean()) if len(scores) else 0.0,
                iou_mean=float(matched_iou.mean()) if len(matched_iou) else 0.0,
            )
        )

    return MatchResult(
        reports=reports,
        scores=np.concatenate(all_scores) if all_scores else np.zeros(0, dtype=np.float32),
        matched=np.concatenate(all_matched) if all_matched else np.zeros(0, dtype=bool),
        total_gt=total_gt,
    )


def confidence_curve(match: MatchResult, steps: Sequence[float]) -> list[dict]:
    order = np.argsort(-match.scores)
    scores = match.scores[order]
    matched = match.matched[order]
    tp_cum = np.cumsum(matched)
    fp_cum = np.cumsum(~matched)

    curve = []
    for threshold in steps:
        taken = int(np.searchsorted(-scores, -float(threshold), side="right"))
        tp = int(tp_cum[taken - 1]) if taken else 0
        fp = int(fp_cum[taken - 1]) if taken else 0
        curve.append({"conf": round(float(threshold), 3), **_prf(tp, fp, max(match.total_gt - tp, 0))})
    return curve


# ------------------------------------------------------------------ metrics
def _class_ap(
    names: Sequence[str],
    truth: dict[str, GroundTruth],
    submission: Submission,
    class_id: int,
    thresholds: Sequence[float],
) -> tuple[list[float], int, int]:
    per_image: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
    n_gt = 0
    n_pred = 0

    for name in names:
        gt = truth.get(name, EMPTY_GT)
        pred = submission.by_image.get(name, EMPTY_PRED)
        gt_boxes = gt.boxes[gt.classes == class_id] if len(gt.boxes) else gt.boxes
        n_gt += len(gt_boxes)

        if len(pred.classes):
            mask = pred.classes == class_id
            boxes, scores = pred.boxes[mask], pred.scores[mask]
        else:
            boxes, scores = pred.boxes, pred.scores
        if not len(boxes):
            continue
        n_pred += len(boxes)
        per_image.append((scores, iou_matrix(boxes, gt_boxes), np.argsort(-scores)))

    aps = []
    for threshold in thresholds:
        scores_all, matched_all = [], []
        for scores, iou, order in per_image:
            pred_match, _ = greedy_match(iou, order, float(threshold))
            scores_all.append(scores)
            matched_all.append(pred_match >= 0)
        if scores_all:
            aps.append(
                average_precision(np.concatenate(scores_all), np.concatenate(matched_all), n_gt)
            )
        else:
            aps.append(0.0)
    return aps, n_gt, n_pred


def mean_ap(
    names: Sequence[str],
    truth: dict[str, GroundTruth],
    submission: Submission,
    class_ids: Sequence[int],
    full_sweep: bool = True,
) -> dict:
    thresholds = IOU_SWEEP if full_sweep else IOU_SWEEP[:1]
    per_class = []
    for class_id in class_ids:
        aps, n_gt, n_pred = _class_ap(names, truth, submission, class_id, thresholds)
        per_class.append(
            {
                "class_id": int(class_id),
                "ap50": round(aps[0], 4),
                "ap": round(float(np.mean(aps)), 4),
                "n_gt": int(n_gt),
                "n_pred": int(n_pred),
            }
        )

    scored = [c for c in per_class if c["n_gt"] > 0]
    return {
        "map50": round(float(np.mean([c["ap50"] for c in scored])) if scored else 0.0, 4),
        "map": round(float(np.mean([c["ap"] for c in scored])) if scored else 0.0, 4),
        "per_class": per_class,
    }


def _histogram(values: np.ndarray, bins: int, value_range: tuple[float, float]) -> dict:
    if not len(values):
        return {"edges": [], "counts": []}
    counts, edges = np.histogram(values, bins=bins, range=value_range)
    return {"edges": [round(float(e), 4) for e in edges], "counts": [int(c) for c in counts]}


def _prf(tp: int, fp: int, fn: int) -> dict:
    precision = tp / max(tp + fp, 1e-9)
    recall = tp / max(tp + fn, 1e-9)
    f1 = 2 * precision * recall / max(precision + recall, 1e-9)
    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "tp": tp,
        "fp": fp,
        "fn": fn,
    }


def build_dashboard(
    project: Project,
    submission: Submission,
    conf_eval: float = 0.25,
    iou_thr: float = 0.5,
    clusters: dict[str, int] | None = None,
    progress=None,
) -> dict:
    step = progress or (lambda message, value=None: None)

    step("Чтение разметки", 0.1)
    truth = load_ground_truth(project)

    step("Сопоставление боксов", 0.3)
    match = match_all(project, truth, submission, iou_thr=iou_thr, conf_eval=conf_eval)
    reports = match.reports
    names = [r.name for r in reports]
    class_ids = sorted({int(c) for gt in truth.values() for c in gt.classes.tolist()})
    if not class_ids:
        class_ids = [0]

    step(f"AP по {len(class_ids)} классам на 10 порогах IoU", 0.4)
    overall = mean_ap(names, truth, submission, class_ids)
    totals = {
        "tp": sum(r.tp for r in reports),
        "fp": sum(r.fp for r in reports),
        "fn": sum(r.fn for r in reports),
    }

    step("Метрики по сплитам", 0.7)
    splits = {}
    for split in sorted({r.split for r in reports}):
        subset = [r for r in reports if r.split == split]
        splits[split] = {
            **_prf(sum(r.tp for r in subset), sum(r.fp for r in subset), sum(r.fn for r in subset)),
            "images": len(subset),
            "map50": mean_ap([r.name for r in subset], truth, submission, class_ids, full_sweep=False)["map50"],
        }

    all_scores = np.concatenate([p.scores for p in submission.by_image.values() if len(p.scores)] or [np.zeros(0)])
    gt_areas = np.concatenate(
        [
            ((g.boxes[:, 2] - g.boxes[:, 0]) * (g.boxes[:, 3] - g.boxes[:, 1]))
            for g in truth.values()
            if len(g.boxes)
        ]
        or [np.zeros(0)]
    )
    pred_areas = np.concatenate(
        [
            ((p.boxes[:, 2] - p.boxes[:, 0]) * (p.boxes[:, 3] - p.boxes[:, 1]))
            for p in submission.by_image.values()
            if len(p.boxes)
        ]
        or [np.zeros(0)]
    )

    curve = confidence_curve(match, np.round(np.arange(0.05, 0.96, 0.05), 2).tolist())

    step("Метрики по кластерам", 0.85)
    cluster_metrics = []
    if clusters:
        # One pass over the reports; rescanning the whole dataset per cluster turned
        # this block into the slowest part of the dashboard.
        grouped: dict[int, list[ImageReport]] = {}
        for report in reports:
            cluster_id = clusters.get(report.name)
            if cluster_id is not None:
                grouped.setdefault(int(cluster_id), []).append(report)

        for cluster_id in sorted(grouped):
            subset = grouped[cluster_id]
            members = [r.name for r in subset]
            cluster_metrics.append(
                {
                    "cluster": int(cluster_id),
                    "images": len(members),
                    "map50": mean_ap(members, truth, submission, class_ids, full_sweep=False)["map50"],
                    **_prf(
                        sum(r.tp for r in subset),
                        sum(r.fp for r in subset),
                        sum(r.fn for r in subset),
                    ),
                    "ghost": sum(r.ghost for r in subset),
                    "missed": sum(r.missed for r in subset),
                }
            )

    step("Сборка дашборда", 0.95)
    worst = sorted(reports, key=lambda r: (r.ghost + r.missed, r.fp + r.fn), reverse=True)[:25]

    return {
        "conf_eval": conf_eval,
        "iou_thr": iou_thr,
        "classes": project.meta.get("names", [f"class_{i}" for i in class_ids]),
        "overall": {**overall, **_prf(totals["tp"], totals["fp"], totals["fn"])},
        "splits": splits,
        "errors": {
            "ghost": sum(r.ghost for r in reports),
            "missed": sum(r.missed for r in reports),
            "loc": sum(r.loc for r in reports),
            "ambiguous": sum(r.ambiguous for r in reports),
            "clean_images": sum(1 for r in reports if r.fp == 0 and r.fn == 0),
            "images_with_errors": sum(1 for r in reports if r.fp or r.fn),
        },
        "coverage": {
            "images": len(reports),
            "with_predictions": sum(1 for r in reports if r.n_pred),
            "with_labels": sum(1 for r in reports if r.n_gt),
            "pred_boxes": int(submission.boxes),
            "gt_boxes": int(sum(r.n_gt for r in reports)),
        },
        "distributions": {
            "confidence": _histogram(all_scores, 20, (0.0, 1.0)),
            "gt_area": _histogram(np.sqrt(gt_areas), 20, (0.0, 1.0)),
            "pred_area": _histogram(np.sqrt(pred_areas), 20, (0.0, 1.0)),
            "boxes_per_image": _histogram(
                np.asarray([r.n_gt for r in reports], dtype=float), 20, (0.0, 40.0)
            ),
        },
        "curve": curve,
        "clusters": cluster_metrics,
        "worst": [r.as_dict() for r in worst],
    }


def dashboard_path(project: Project) -> Path:
    return project.cache_dir / "dashboard.json"


def cached_dashboard(project: Project) -> dict | None:
    path = dashboard_path(project)
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def store_dashboard(project: Project, payload: dict) -> None:
    project.cache_dir.mkdir(parents=True, exist_ok=True)
    dashboard_path(project).write_text(json.dumps(payload), encoding="utf-8")
