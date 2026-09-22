from __future__ import annotations

import logging
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence

import numpy as np

from . import embeddings as embeddings_service
from .analysis import greedy_match
from .dataset import iou_matrix, read_label, xywh_to_xyxy
from .project import Project

log = logging.getLogger(__name__)
LOW_CONF = 0.001


@dataclass
class Detections:
    """Raw boxes of one model over the tuning subset, kept in memory.

    Models run once; every Optuna trial only replays WBF over these arrays,
    which is what makes a 100-trial search finish in seconds instead of hours.
    """

    boxes: list[np.ndarray]
    scores: list[np.ndarray]
    classes: list[np.ndarray]


def diverse_validation_images(project: Project, k: int, split: str | None = "val") -> list[dict]:
    """k-center greedy over the embedding space — maximal spread, no duplicates."""
    samples = [s for s in project.samples() if s.get("label")]
    if split:
        subset = [s for s in samples if s.get("split") == split]
        samples = subset or samples
    if not samples:
        raise ValueError("Нет размеченных изображений для валидации")

    k = int(max(1, min(k, len(samples))))
    emb = embeddings_service.compute(project)
    index = {name: i for i, name in enumerate(emb.names)}
    rows = [(s, index[s["name"]]) for s in samples if s["name"] in index]
    if not rows:
        return samples[:k]

    z = emb.z[[i for _, i in rows]]
    picked = [int(np.argmax(np.linalg.norm(z - z.mean(axis=0), axis=1)))]
    distances = np.linalg.norm(z - z[picked[0]], axis=1)

    while len(picked) < k:
        nxt = int(np.argmax(distances))
        if distances[nxt] <= 0:
            break
        picked.append(nxt)
        distances = np.minimum(distances, np.linalg.norm(z - z[nxt], axis=1))

    return [rows[i][0] for i in picked]


def collect_detections(
    model_path: Path,
    images: Sequence[dict],
    device: str,
    log_fn: Callable[[str], None] | None = None,
) -> Detections:
    from ultralytics import YOLO

    model = YOLO(str(model_path))
    boxes_all, scores_all, classes_all = [], [], []

    for position, sample in enumerate(images, start=1):
        result = model.predict(
            sample["image"], conf=LOW_CONF, iou=1.0, max_det=1000, verbose=False, device=device
        )[0]
        if result.boxes is None or not len(result.boxes):
            boxes_all.append(np.empty((0, 4), np.float32))
            scores_all.append(np.empty(0, np.float32))
            classes_all.append(np.empty(0, np.float32))
            continue

        boxes = result.boxes.xyxy.cpu().numpy().astype(np.float32)
        height, width = result.orig_shape
        boxes[:, [0, 2]] /= max(width, 1)
        boxes[:, [1, 3]] /= max(height, 1)
        boxes_all.append(np.clip(boxes, 0.0, 1.0))
        scores_all.append(result.boxes.conf.cpu().numpy().astype(np.float32))
        classes_all.append(result.boxes.cls.cpu().numpy().astype(np.float32))

        if log_fn and position % 20 == 0:
            log_fn(f"{model_path.name}: {position}/{len(images)}")

    del model
    return Detections(boxes=boxes_all, scores=scores_all, classes=classes_all)


def load_targets(images: Sequence[dict]) -> list[tuple[np.ndarray, np.ndarray]]:
    targets = []
    for sample in images:
        data = read_label(Path(sample["label"])) if sample.get("label") else np.empty((0, 5), np.float32)
        boxes = xywh_to_xyxy(data[:, 1:5]) if len(data) else np.empty((0, 4), np.float32)
        classes = data[:, 0].astype(np.int32) if len(data) else np.empty(0, np.int32)
        targets.append((boxes, classes))
    return targets


def fuse_and_score(
    detections: Sequence[Detections],
    targets: Sequence[tuple[np.ndarray, np.ndarray]],
    confs: Sequence[float],
    wbf_iou: float,
    skip_thr: float,
    conf_type: str,
    weights: Sequence[float],
    eval_iou: float = 0.5,
) -> float:
    from ensemble_boxes import weighted_boxes_fusion

    tp = fp = fn = 0
    for idx, (gt_boxes, gt_classes) in enumerate(targets):
        boxes_list, scores_list, labels_list = [], [], []
        for model_idx, det in enumerate(detections):
            keep = det.scores[idx] >= confs[model_idx]
            boxes_list.append(det.boxes[idx][keep].tolist())
            scores_list.append(det.scores[idx][keep].tolist())
            labels_list.append(det.classes[idx][keep].tolist())

        if all(not b for b in boxes_list):
            fn += len(gt_boxes)
            continue

        fused_boxes, fused_scores, fused_classes = weighted_boxes_fusion(
            boxes_list,
            scores_list,
            labels_list,
            weights=list(weights),
            iou_thr=wbf_iou,
            skip_box_thr=skip_thr,
            conf_type=conf_type,
        )
        if not len(fused_boxes):
            fn += len(gt_boxes)
            continue

        iou = iou_matrix(np.asarray(fused_boxes, np.float32), gt_boxes)
        if iou.size:
            iou = np.where(np.asarray(fused_classes, np.int32)[:, None] == gt_classes[None, :], iou, 0.0)
        pred_match, gt_taken = greedy_match(iou, np.argsort(-np.asarray(fused_scores)), eval_iou)

        hits = int((pred_match >= 0).sum())
        tp += hits
        fp += len(fused_boxes) - hits
        fn += int((~gt_taken).sum()) if len(gt_boxes) else 0

    precision = tp / max(tp + fp, 1e-9)
    recall = tp / max(tp + fn, 1e-9)
    return float(2 * precision * recall / max(precision + recall, 1e-9))


def _random_search(space: dict, trials: int, objective, log_fn, seed: int = 42) -> tuple[dict, float]:
    """Fallback when Optuna is not installed: random sampling plus local refinement."""
    rng = random.Random(seed)
    best_params, best_score = None, -1.0

    for trial in range(trials):
        if best_params is not None and trial > trials // 3:
            params = {
                key: float(np.clip(rng.gauss(best_params[key], (hi - lo) * 0.12), lo, hi))
                for key, (lo, hi) in space.items()
            }
        else:
            params = {key: rng.uniform(lo, hi) for key, (lo, hi) in space.items()}

        score = objective(params)
        if score > best_score:
            best_params, best_score = params, score
            log_fn(f"trial {trial + 1}/{trials}: F1={score:.4f} ★")
        elif (trial + 1) % 5 == 0:
            log_fn(f"trial {trial + 1}/{trials}: F1={score:.4f}")

    return best_params or {}, best_score


def _optuna_search(space: dict, trials: int, objective, log_fn, seed: int = 42):
    import optuna

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=seed))

    def wrapped(trial):
        params = {key: trial.suggest_float(key, lo, hi) for key, (lo, hi) in space.items()}
        score = objective(params)
        log_fn(f"trial {trial.number + 1}/{trials}: F1={score:.4f}")
        return score

    study.optimize(wrapped, n_trials=trials, show_progress_bar=False)
    return study.best_params, float(study.best_value)


def tune(
    project: Project,
    model_paths: Sequence[Path],
    val_images: Sequence[dict],
    trials: int = 40,
    conf_range: tuple[float, float] = (0.005, 0.5),
    iou_range: tuple[float, float] = (0.3, 0.7),
    skip_range: tuple[float, float] = (0.001, 0.05),
    conf_type: str = "avg",
    model_weights: Sequence[float] | None = None,
    device: str = "auto",
    ctx=None,
) -> dict:
    import torch

    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    log_fn = ctx.log if ctx else (lambda message: None)
    progress = ctx.progress if ctx else (lambda message, value=None: None)

    progress(f"Прогон {len(model_paths)} моделей по {len(val_images)} изображениям", 0.1)
    detections = []
    for index, path in enumerate(model_paths):
        detections.append(collect_detections(path, val_images, device, log_fn))
        progress(f"Модель {index + 1}/{len(model_paths)} готова", 0.1 + 0.5 * (index + 1) / len(model_paths))

    targets = load_targets(val_images)
    weights = list(model_weights) if model_weights else [1.0] * len(model_paths)

    space: dict[str, tuple[float, float]] = {f"conf_{i}": conf_range for i in range(len(model_paths))}
    space["wbf_iou"] = iou_range
    space["skip_box_thr"] = skip_range

    def objective(params: dict) -> float:
        return fuse_and_score(
            detections,
            targets,
            [params[f"conf_{i}"] for i in range(len(model_paths))],
            params["wbf_iou"],
            params["skip_box_thr"],
            conf_type,
            weights,
        )

    progress("Оптимизация WBF", 0.65)
    try:
        best, score = _optuna_search(space, trials, objective, log_fn)
        sampler = "optuna-tpe"
    except ImportError:
        log_fn("Optuna не установлена — используется случайный поиск с уточнением")
        best, score = _random_search(space, trials, objective, log_fn)
        sampler = "random-refine"

    baseline = objective({**{f"conf_{i}": 0.25 for i in range(len(model_paths))}, "wbf_iou": 0.55, "skip_box_thr": 0.01})
    progress("Готово", 1.0)

    return {
        "conf": {path.name: round(float(best[f"conf_{i}"]), 5) for i, path in enumerate(model_paths)},
        "conf_default": round(float(np.mean([best[f"conf_{i}"] for i in range(len(model_paths))])), 5),
        "wbf_iou": round(float(best["wbf_iou"]), 5),
        "skip_box_thr": round(float(best["skip_box_thr"]), 5),
        "conf_type": conf_type,
        "weights": {path.name: float(w) for path, w in zip(model_paths, weights)},
        "score": round(float(score), 5),
        "baseline_score": round(float(baseline), 5),
        "trials": int(trials),
        "val_images": len(val_images),
        "sampler": sampler,
        "device": device,
    }
