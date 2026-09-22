from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

from ..config import STATUS_ASSIGNED
from . import embeddings as embeddings_service
from . import quality as quality_service
from . import submission as submission_service
from . import workflow
from .analysis import ImageReport, load_ground_truth, match_all
from .project import Project, utc_now

DEFAULT_WEIGHTS = {"ghost": 0.35, "missed": 0.35, "ambiguous": 0.20, "count": 0.10}
REASONS = {
    "ghost": "Уверенный бокс без разметки",
    "missed": "Пропущенный объект",
    "ambiguous": "Неуверенные предсказания",
    "count": "Расхождение по количеству",
    "interest": "Ошибки модели",
    "diversity": "Новая область признаков",
    "medoid": "Центр кластера",
    "assignment": "Из задания",
    "issue": "Подозрительная разметка",
}


@dataclass
class Selection:
    items: list[dict]
    clusters: list[dict]
    signals: dict
    backend: str
    created_at: str = field(default_factory=utc_now)

    def as_dict(self) -> dict:
        return {
            "items": self.items,
            "clusters": self.clusters,
            "signals": self.signals,
            "backend": self.backend,
            "created_at": self.created_at,
            "budget": len(self.items),
        }


def rank_normalize(values: np.ndarray) -> np.ndarray:
    """Rank based scaling to [0, 1] — immune to the heavy tails of error counts."""
    if not len(values):
        return values
    spread = float(values.max() - values.min())
    if spread <= 0:
        return np.zeros_like(values, dtype=np.float32)
    order = np.argsort(values, kind="stable")
    ranks = np.empty(len(values), dtype=np.float32)
    ranks[order] = np.arange(len(values), dtype=np.float32)
    return (ranks / max(len(values) - 1, 1)).astype(np.float32)


def kmeans(z: np.ndarray, k: int, seed: int = 0):
    from sklearn.cluster import KMeans, MiniBatchKMeans

    k = int(max(1, min(k, len(z))))
    if len(z) > 8000:
        model = MiniBatchKMeans(n_clusters=k, random_state=seed, n_init=3, batch_size=2048)
    else:
        model = KMeans(n_clusters=k, random_state=seed, n_init=4)
    labels = model.fit_predict(z)
    return labels.astype(np.int32), model.cluster_centers_.astype(np.float32)


def cluster_cache_path(project: Project):
    return project.cache_dir / "clusters.npz"


def assign_clusters(project: Project, n_clusters: int, force: bool = False) -> tuple[list[str], np.ndarray]:
    path = cluster_cache_path(project)
    if path.exists() and not force:
        data = np.load(path, allow_pickle=True)
        if int(data["k"]) == int(n_clusters):
            return [str(n) for n in data["names"]], data["labels"].astype(np.int32)

    emb = embeddings_service.compute(project)
    labels, _ = kmeans(emb.z, n_clusters)
    project.cache_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, names=np.array(emb.names), labels=labels, k=int(n_clusters))
    return emb.names, labels


def interest_signals(
    reports: Sequence[ImageReport],
    weights: dict[str, float],
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Turns detection errors into a per-image `worth labelling` score in [0, 1]."""
    ghost = np.asarray([r.ghost for r in reports], dtype=np.float32)
    missed = np.asarray([r.missed for r in reports], dtype=np.float32)
    ambiguous = np.asarray([r.ambiguous for r in reports], dtype=np.float32)
    n_pred = np.asarray([r.n_pred for r in reports], dtype=np.float32)
    n_gt = np.asarray([r.n_gt for r in reports], dtype=np.float32)
    count = np.abs(n_pred - n_gt) / np.maximum(np.maximum(n_pred, n_gt), 1.0)

    components = {
        "ghost": rank_normalize(ghost),
        "missed": rank_normalize(missed),
        "ambiguous": rank_normalize(ambiguous),
        "count": rank_normalize(count),
    }
    total = sum(weights.values()) or 1.0
    interest = sum(components[key] * (weights.get(key, 0.0) / total) for key in components)
    return np.asarray(interest, dtype=np.float32), components


def allocate(sizes: np.ndarray, interest: np.ndarray, budget: int) -> np.ndarray:
    """Splits the labelling budget between clusters by size and error mass."""
    sizes = sizes.astype(int)
    mass = np.sqrt(np.maximum(sizes, 1)) * (1.0 + 2.0 * interest)
    share = mass / max(mass.sum(), 1e-9) * budget

    quota = np.minimum(np.floor(share).astype(int), sizes)
    remainder = budget - int(quota.sum())
    if remainder > 0:
        priority = np.argsort(-(share - np.floor(share)))
        for idx in priority:
            if remainder <= 0:
                break
            if quota[idx] < sizes[idx]:
                quota[idx] += 1
                remainder -= 1
        while remainder > 0 and np.any(quota < sizes):
            for idx in np.argsort(-mass):
                if remainder <= 0:
                    break
                if quota[idx] < sizes[idx]:
                    quota[idx] += 1
                    remainder -= 1
    return quota


def _greedy_pick(
    z: np.ndarray,
    indices: np.ndarray,
    interest: np.ndarray,
    quota: int,
    alpha: float,
    center: np.ndarray,
) -> list[tuple[int, str, float]]:
    """Alternates between the most informative and the most distant candidate."""
    chosen: list[tuple[int, str, float]] = []
    if quota <= 0 or not len(indices):
        return chosen

    local_z = z[indices]
    local_interest = interest[indices]
    distances = np.full(len(indices), np.inf, dtype=np.float32)

    if local_interest.max() > 0:
        first = int(np.argmax(local_interest))
        reason = "interest"
    else:
        first = int(np.argmin(np.linalg.norm(local_z - center, axis=1)))
        reason = "medoid"

    for step in range(int(min(quota, len(indices)))):
        if step == 0:
            pick = first
        else:
            finite = distances[np.isfinite(distances)]
            scale = float(finite.max()) if finite.size else 1.0
            novelty = np.clip(distances / max(scale, 1e-9), 0.0, 1.0)
            score = alpha * novelty + (1.0 - alpha) * local_interest
            score[[c[0] for c in chosen]] = -np.inf
            pick = int(np.argmax(score))
            reason = "diversity" if alpha * novelty[pick] >= (1.0 - alpha) * local_interest[pick] else "interest"

        chosen.append((pick, reason, float(local_interest[pick])))
        gap = np.linalg.norm(local_z - local_z[pick], axis=1)
        distances = np.minimum(distances, gap)
        distances[pick] = 0.0

    return [(int(indices[p]), reason, value) for p, reason, value in chosen]


def eligible(
    project: Project,
    names: Sequence[str],
    split: str | None = None,
    scope: str = "all",
    include_assigned: bool = False,
) -> np.ndarray:
    """Булева маска кадров, которые вообще можно предлагать к разметке.

    Кадр, отданный в задание, исключается по умолчанию: смысл заданий ровно в
    том, чтобы двое не размечали одно и то же.
    """
    samples = project.by_name()
    state = workflow.load(project)
    assigned = set(state.names_with(STATUS_ASSIGNED)) if not include_assigned else set()

    flagged: set[str] = set()
    if scope == "issues":
        report = quality_service.cached(project)
        if not report:
            raise ValueError("Сначала запустите проверку разметки")
        flagged = {row["name"] for row in report["images"]}

    out = np.zeros(len(names), dtype=bool)
    for i, name in enumerate(names):
        sample = samples.get(name)
        if sample is None or name in assigned:
            continue
        if split and sample.get("split") != split:
            continue
        if scope == "unlabeled" and (sample.get("boxes") or sample.get("label")):
            continue
        if scope == "labeled" and not (sample.get("boxes") or sample.get("label")):
            continue
        if scope == "issues" and name not in flagged:
            continue
        out[i] = True
    return out


def build_selection(
    project: Project,
    budget: int,
    n_clusters: int | None = None,
    alpha: float = 0.5,
    weights: dict[str, float] | None = None,
    split: str | None = None,
    scope: str = "all",
    include_assigned: bool = False,
    progress=None,
) -> Selection:
    weights = {**DEFAULT_WEIGHTS, **(weights or {})}
    if progress:
        progress("Загрузка эмбеддингов", 0.1)
    emb = embeddings_service.compute(project, progress=progress)

    samples = project.by_name()
    mask = eligible(project, emb.names, split=split, scope=scope, include_assigned=include_assigned)
    pool = np.where(mask)[0]
    if not len(pool):
        raise ValueError("Нет изображений, подходящих под фильтр — возможно, всё уже занято заданиями")

    budget = int(max(1, min(budget, len(pool))))
    n_clusters = int(n_clusters or min(budget, max(2, len(pool) // 10)))
    n_clusters = max(1, min(n_clusters, len(pool)))

    if progress:
        progress("Кластеризация", 0.45)
    z = emb.z[pool]
    labels, centers = kmeans(z, n_clusters)

    if progress:
        progress("Анализ ошибок", 0.7)
    sub = submission_service.load(project)
    reports_by_name: dict[str, ImageReport] = {}
    if sub is not None:
        truth = load_ground_truth(project)
        result = match_all(project, truth, sub)
        reports_by_name = {r.name: r for r in result.reports}

    if reports_by_name:
        ordered = [reports_by_name.get(emb.names[i]) for i in pool]
        ordered = [r for r in ordered if r is not None]
        if len(ordered) == len(pool):
            interest, components = interest_signals(ordered, weights)
        else:
            interest, components = np.zeros(len(pool), dtype=np.float32), {}
    else:
        interest, components = np.zeros(len(pool), dtype=np.float32), {}
        alpha = 1.0

    if progress:
        progress("Отбор изображений", 0.85)
    sizes = np.asarray([(labels == c).sum() for c in range(n_clusters)], dtype=np.float32)
    cluster_interest = np.asarray(
        [float(interest[labels == c].mean()) if (labels == c).any() else 0.0 for c in range(n_clusters)],
        dtype=np.float32,
    )
    quota = allocate(sizes, cluster_interest, budget)

    items: list[dict] = []
    for cluster_id in range(n_clusters):
        local = np.where(labels == cluster_id)[0]
        for local_idx, reason, value in _greedy_pick(
            z, local, interest, int(quota[cluster_id]), alpha, centers[cluster_id]
        ):
            global_idx = int(pool[local_idx])
            name = emb.names[global_idx]
            report = reports_by_name.get(name)
            detail = reason
            if report is not None and reason == "interest":
                ranked = {
                    "ghost": report.ghost * weights["ghost"],
                    "missed": report.missed * weights["missed"],
                    "ambiguous": report.ambiguous * weights["ambiguous"],
                }
                detail = max(ranked, key=ranked.get) if any(ranked.values()) else "count"

            items.append(
                {
                    "name": name,
                    "cluster": int(cluster_id),
                    "interest": round(float(value), 4),
                    "reason": detail,
                    "reason_label": REASONS.get(detail, REASONS["diversity"]),
                    "split": samples.get(name, {}).get("split", "train"),
                    "n_gt": report.n_gt if report else samples.get(name, {}).get("boxes", 0),
                    "n_pred": report.n_pred if report else 0,
                    "ghost": report.ghost if report else 0,
                    "missed": report.missed if report else 0,
                }
            )

    items.sort(key=lambda item: (-item["interest"], item["cluster"]))
    for rank, item in enumerate(items, start=1):
        item["rank"] = rank

    clusters = [
        {
            "cluster": int(c),
            "size": int(sizes[c]),
            "quota": int(quota[c]),
            "interest": round(float(cluster_interest[c]), 4),
        }
        for c in range(n_clusters)
    ]
    signals = {
        "weights": weights,
        "alpha": alpha,
        "scope": scope,
        "split": split,
        "with_submission": sub is not None,
        "components": {k: round(float(v.mean()), 4) for k, v in components.items()},
        "pool": int(len(pool)),
        "clusters": n_clusters,
    }

    if progress:
        progress("Готово", 1.0)
    return Selection(items=items, clusters=clusters, signals=signals, backend=emb.backend)


def save_selection(project: Project, selection: Selection) -> str:
    project.selections_dir.mkdir(parents=True, exist_ok=True)
    stamp = selection.created_at.replace(":", "").replace("-", "").replace("+0000", "")
    path = project.selections_dir / f"selection_{stamp}.json"
    path.write_text(json.dumps(selection.as_dict(), ensure_ascii=False), encoding="utf-8")
    project.meta["last_selection"] = {"file": path.name, "budget": len(selection.items), "at": selection.created_at}
    project.save()
    return path.name


def list_selections(project: Project) -> list[dict]:
    out = []
    for path in sorted(project.selections_dir.glob("selection_*.json"), reverse=True):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        out.append({"file": path.name, "budget": payload.get("budget"), "created_at": payload.get("created_at")})
    return out
