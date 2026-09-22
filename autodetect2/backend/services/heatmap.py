from __future__ import annotations

import json

import numpy as np

from . import embeddings as embeddings_service
from . import submission as submission_service
from .analysis import load_ground_truth, match_all
from .project import Project
from .projection import project_2d
from .selection import DEFAULT_WEIGHTS, interest_signals, kmeans

STATES = ["no-pred", "ok", "ghost", "missed", "mixed", "low-conf"]


def _state(report) -> int:
    if report is None or (report.n_pred == 0 and report.n_gt == 0):
        return 0
    if report.ghost and report.missed:
        return 4
    if report.ghost:
        return 2
    if report.missed:
        return 3
    if report.conf_max and report.conf_max < 0.4:
        return 5
    return 1


def build(project: Project, n_clusters: int = 24, force: bool = False) -> dict:
    cache = project.cache_dir / "heatmap.json"
    if cache.exists() and not force:
        try:
            payload = json.loads(cache.read_text(encoding="utf-8"))
            if payload.get("clusters_k") == n_clusters:
                return payload
        except (OSError, json.JSONDecodeError):
            pass

    emb = embeddings_service.compute(project)
    z2d, method = project_2d(project, emb, force=force)
    labels, _ = kmeans(emb.z, n_clusters)

    samples = {s["name"]: s for s in project.samples()}
    sub = submission_service.load(project)
    reports = {}
    interest = np.zeros(len(emb.names), dtype=np.float32)

    if sub is not None:
        truth = load_ground_truth(project)
        result = match_all(project, truth, sub)
        reports = {r.name: r for r in result.reports}
        ordered = [reports.get(name) for name in emb.names]
        if all(r is not None for r in ordered):
            interest, _ = interest_signals(ordered, DEFAULT_WEIGHTS)

    states = np.asarray([_state(reports.get(name)) for name in emb.names], dtype=np.int8)
    conf = np.asarray(
        [reports[name].conf_max if name in reports else 0.0 for name in emb.names], dtype=np.float32
    )
    errors = np.asarray(
        [
            (reports[name].ghost + reports[name].missed) if name in reports else 0
            for name in emb.names
        ],
        dtype=np.int32,
    )

    def column(field: str) -> list[int]:
        return [int(getattr(reports[name], field)) if name in reports else 0 for name in emb.names]

    span = np.ptp(z2d, axis=0)
    span[span == 0] = 1.0
    norm = (z2d - z2d.min(axis=0)) / span

    clusters = []
    for cluster_id in range(int(labels.max()) + 1):
        mask = labels == cluster_id
        if not mask.any():
            continue
        clusters.append(
            {
                "cluster": int(cluster_id),
                "size": int(mask.sum()),
                "cx": round(float(norm[mask, 0].mean()), 4),
                "cy": round(float(norm[mask, 1].mean()), 4),
                "interest": round(float(interest[mask].mean()), 4),
                "errors": int(errors[mask].sum()),
                "conf": round(float(conf[mask].mean()), 4),
            }
        )

    payload = {
        "method": method,
        "backend": emb.backend,
        "clusters_k": n_clusters,
        "count": len(emb.names),
        "has_submission": sub is not None,
        "states": STATES,
        "points": {
            "names": emb.names,
            "x": [round(float(v), 4) for v in norm[:, 0]],
            "y": [round(float(v), 4) for v in norm[:, 1]],
            "cluster": labels.tolist(),
            "interest": [round(float(v), 3) for v in interest],
            "conf": [round(float(v), 3) for v in conf],
            "errors": errors.tolist(),
            "fp": column("fp"),
            "fn": column("fn"),
            "loc": column("loc"),
            "n_gt": column("n_gt"),
            "n_pred": column("n_pred"),
            "state": states.tolist(),
            "split": [samples.get(name, {}).get("split", "train") for name in emb.names],
        },
        "clusters": clusters,
    }

    project.cache_dir.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(payload), encoding="utf-8")
    return payload
