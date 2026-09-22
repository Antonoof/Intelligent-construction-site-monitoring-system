from __future__ import annotations

import logging

import numpy as np

from .embeddings import Embeddings
from .project import Project

log = logging.getLogger(__name__)


def _umap(z: np.ndarray) -> np.ndarray | None:
    try:
        import umap  # noqa: PLC0415 - optional dependency
    except ImportError:
        return None
    reducer = umap.UMAP(n_neighbors=30, min_dist=0.1, metric="cosine", random_state=0)
    return np.asarray(reducer.fit_transform(z), dtype=np.float32)


def _tsne(z: np.ndarray) -> np.ndarray | None:
    if len(z) > 6000:
        return None
    from sklearn.manifold import TSNE

    model = TSNE(n_components=2, init="pca", perplexity=min(30, max(5, len(z) // 20)), random_state=0)
    return np.asarray(model.fit_transform(z), dtype=np.float32)


def _pca(z: np.ndarray) -> np.ndarray:
    from sklearn.decomposition import PCA

    return np.asarray(PCA(n_components=2, random_state=0).fit_transform(z), dtype=np.float32)


def place_missing(z: np.ndarray, z2d: np.ndarray, known: np.ndarray, neighbours: int = 12) -> np.ndarray:
    """Out-of-sample placement: a new point lands where its nearest anchors sit.

    Cheaper and far more faithful than re-running the projection, and it keeps the
    cached UMAP layout intact for the images that came with coordinates.
    """
    from sklearn.neighbors import NearestNeighbors

    filled = z2d.copy()
    missing = ~known
    if not missing.any() or not known.any():
        return filled

    index = NearestNeighbors(n_neighbors=min(neighbours, int(known.sum())), metric="cosine")
    index.fit(z[known])
    distances, indices = index.kneighbors(z[missing])

    weights = 1.0 / np.maximum(distances, 1e-6)
    weights /= weights.sum(axis=1, keepdims=True)
    anchors = z2d[known]
    filled[missing] = np.einsum("nk,nkd->nd", weights, anchors[indices])
    return filled.astype(np.float32)


def project_2d(project: Project, emb: Embeddings, force: bool = False) -> tuple[np.ndarray, str]:
    """2D layout for the heatmap. Selection always happens in the full space."""
    cache = project.cache_dir / "projection.npz"
    if cache.exists() and not force:
        try:
            data = np.load(cache)
            if len(data["z2d"]) == len(emb.names):
                return data["z2d"].astype(np.float32), str(data["method"])
        except (OSError, KeyError, ValueError):
            pass

    if emb.z2d is not None and len(emb.z2d) == len(emb.names):
        z2d = emb.z2d.astype(np.float32)
        method = "umap-cache"
        if emb.known is not None and not emb.known.all():
            z2d = place_missing(emb.z, z2d, emb.known)
            method = "umap-cache+knn"
    else:
        z2d = _umap(emb.z)
        method = "umap"
        if z2d is None:
            z2d = _tsne(emb.z)
            method = "tsne"
        if z2d is None:
            z2d = _pca(emb.z)
            method = "pca"

    project.cache_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache, z2d=z2d, method=method)
    return z2d, method
