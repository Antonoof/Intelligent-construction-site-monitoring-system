from __future__ import annotations

import logging
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence

import numpy as np

from ..config import EMBED_DIM, MAX_EMBED_IMAGES, REFERENCE_DIR
from .project import Project

log = logging.getLogger(__name__)
ProgressFn = Callable[[str, float], None]

DINO_MODEL = "dinov2_vits14_reg"
DINO_SIZE = (252, 448)
# Decoding thousands of JPEGs is latency bound, so images are read in parallel.
IO_WORKERS = 12
LOAD_TIMEOUT = 45.0
DINO_CPU_LIMIT = 1500
BRIDGE_ANCHORS = 6000


MIN_ANCHORS = 0.2


@dataclass
class Embeddings:
    names: list[str]
    z: np.ndarray  # (N, D) L2-normalised working space
    z2d: np.ndarray | None  # (N, 2) cached projection, may contain NaN rows
    backend: str
    known: np.ndarray | None = None  # rows taken verbatim from the DINOv2 cache

    def index_of(self) -> dict[str, int]:
        return {name: i for i, name in enumerate(self.names)}


def l2_normalize(a: np.ndarray) -> np.ndarray:
    return a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-8)


def build_workspace(feats: np.ndarray, dim: int = EMBED_DIM) -> np.ndarray:
    """L2 -> PCA whitening -> L2, the space used for clustering and selection."""
    from sklearn.decomposition import PCA

    dim = int(min(dim, feats.shape[1], max(feats.shape[0] - 1, 2)))
    pca = PCA(n_components=dim, whiten=True, random_state=0)
    return l2_normalize(pca.fit_transform(l2_normalize(feats))).astype(np.float32)


# ------------------------------------------------------- опорный кэш DINOv2
def _read_reference_npz(path: Path) -> tuple[list[str], np.ndarray, np.ndarray | None] | None:
    """Читает опорный .npz в любой из двух раскладок.

    Плоская — `names` + `z` (+ необязательный `z2d`); разбитая по сплитам —
    `names_tr`/`ztr` и `names_te`/`zte`. Вторая осталась от прежних выгрузок,
    первая — то, что пишет `tools/build_reference.py`.
    """
    try:
        data = np.load(path, allow_pickle=True)
        files = set(data.files)
        if {"names", "z"} <= files:
            names = [str(n) for n in data["names"]]
            z = data["z"].astype(np.float32)
            z2d = data["z2d"].astype(np.float32) if "z2d" in files and data["z2d"].size else None
        elif {"names_tr", "ztr"} <= files:
            names = [str(n) for n in data["names_tr"]] + [str(n) for n in data.get("names_te", [])]
            parts = [data["ztr"]] + ([data["zte"]] if "zte" in files else [])
            z = np.concatenate(parts).astype(np.float32)
            z2d = None
        else:
            return None
    except (OSError, KeyError, ValueError) as exc:
        log.warning("Опорный кэш %s не прочитан: %s", path.name, exc)
        return None

    if len(names) != len(z):
        log.warning("Опорный кэш %s: %d имён на %d векторов", path.name, len(names), len(z))
        return None
    return names, z, z2d


def _load_reference_cache() -> tuple[dict[str, int], np.ndarray, np.ndarray | None] | None:
    """Склеивает все .npz из папки опорных эмбеддингов в один индекс.

    Папки может не быть вовсе — это нормальный режим работы: признаки тогда
    считаются по самому проекту. Опорный кэш нужен, только когда те же кадры уже
    прогонялись через DINOv2 раньше и считать их заново незачем.
    """
    if not REFERENCE_DIR.is_dir():
        return None

    lookup: dict[str, int] = {}
    vectors: list[np.ndarray] = []
    plane: list[np.ndarray] = []
    dim: int | None = None
    offset = 0

    for path in sorted(REFERENCE_DIR.glob("*.npz")):
        entry = _read_reference_npz(path)
        if entry is None:
            continue
        names, z, z2d = entry
        if dim is None:
            dim = z.shape[1]
        elif z.shape[1] != dim:
            log.warning("Опорный кэш %s пропущен: размерность %d вместо %d", path.name, z.shape[1], dim)
            continue

        vectors.append(z)
        plane.append(z2d if z2d is not None else np.full((len(names), 2), np.nan, dtype=np.float32))
        for i, name in enumerate(names):
            lookup.setdefault(name, offset + i)
        offset += len(names)

    if not vectors:
        return None

    z_all = np.concatenate(vectors)
    z2d_all = np.concatenate(plane)
    return lookup, z_all, (None if np.isnan(z2d_all).all() else z2d_all)


@dataclass
class Reference:
    """Whatever the shipped DINOv2 cache knows about this project's images."""

    z: np.ndarray  # (N, 128), rows outside `known` are zero
    z2d: np.ndarray | None  # (N, 2), rows outside `known` are NaN
    known: np.ndarray  # bool mask

    @property
    def coverage(self) -> float:
        return float(self.known.mean()) if len(self.known) else 0.0


def _from_reference(names: Sequence[str]) -> Reference | None:
    cache = _load_reference_cache()
    if cache is None:
        return None

    lookup, z, z2d = cache
    idx = np.array([lookup.get(name, -1) for name in names])
    known = idx >= 0
    if not known.any():
        return None

    resolved_z = np.zeros((len(names), z.shape[1]), dtype=np.float32)
    resolved_z[known] = z[idx[known]]

    resolved_2d = None
    if z2d is not None:
        resolved_2d = np.full((len(names), 2), np.nan, dtype=np.float32)
        resolved_2d[known] = z2d[idx[known]]

    return Reference(z=resolved_z, z2d=resolved_2d, known=known)


def bridge(anchor_feats: np.ndarray, anchor_z: np.ndarray, query_feats: np.ndarray) -> np.ndarray:
    """Places uncached images into the cached DINOv2 space.

    The cache stores vectors, not the transform that produced them, so images it
    never saw cannot be embedded exactly. Ridge regression from locally computed
    features onto the cached vectors — fitted on the images present in both — is
    what keeps the whole dataset inside a single space instead of discarding the
    cache and recomputing everything.
    """
    x = np.column_stack([anchor_feats, np.ones(len(anchor_feats), dtype=np.float32)]).astype(np.float64)

    # Normal equations on a ~200-wide design matrix: one small solve.
    gram = x.T @ x
    gram[np.diag_indices_from(gram)] += 1.0
    weights = np.linalg.solve(gram, x.T @ anchor_z.astype(np.float64))

    query = np.column_stack([query_feats, np.ones(len(query_feats), dtype=np.float32)]).astype(np.float64)
    return l2_normalize(query @ weights).astype(np.float32)


# ------------------------------------------------------------------- DINOv2
_dino_available: bool | None = None


def _load_dino(timeout: float = LOAD_TIMEOUT):
    """Loads DINOv2 from the torch hub, giving up quickly when offline.

    torch.hub retries hard against GitHub, which can stall for minutes on a rate
    limit. The dataset scan must not wait for that, so the load runs on a side
    thread and the caller moves on when it overruns.
    """
    global _dino_available
    if _dino_available is False:
        raise RuntimeError("DINOv2 недоступна в этом окружении")

    import torch

    result: dict[str, object] = {}

    def worker() -> None:
        try:
            result["model"] = torch.hub.load("facebookresearch/dinov2", DINO_MODEL)
        except Exception as exc:  # noqa: BLE001 - reported through `result`
            result["error"] = exc

    thread = threading.Thread(target=worker, name="dinov2-load", daemon=True)
    thread.start()
    thread.join(timeout)

    if "model" not in result:
        _dino_available = False
        reason = result.get("error") or f"загрузка дольше {timeout:.0f}s"
        raise RuntimeError(f"DINOv2 недоступна: {reason}")

    _dino_available = True
    return result["model"]


def _dino_features(paths: Sequence[Path], progress: ProgressFn | None) -> np.ndarray:
    import torch
    from PIL import Image

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = _load_dino().to(device).eval()

    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    batch_size = 16 if device == "cpu" else 32
    out: list[np.ndarray] = []

    def load(path: Path) -> torch.Tensor:
        img = Image.open(path)
        img.draft("RGB", (DINO_SIZE[1], DINO_SIZE[0]))
        img = img.convert("RGB").resize((DINO_SIZE[1], DINO_SIZE[0]))
        x = torch.from_numpy(np.asarray(img).copy()).permute(2, 0, 1).float().div_(255.0)
        return (x - mean) / std

    with torch.no_grad(), ThreadPoolExecutor(max_workers=IO_WORKERS) as pool:
        for start in range(0, len(paths), batch_size):
            chunk = paths[start : start + batch_size]
            batch = torch.stack(list(pool.map(load, chunk))).to(device)
            res = model.forward_features(batch)
            feat = torch.cat([res["x_norm_clstoken"], res["x_norm_patchtokens"].mean(1)], dim=1)
            out.append(feat.float().cpu().numpy())
            if progress:
                done = min(start + batch_size, len(paths))
                progress(f"DINOv2: {done}/{len(paths)}", done / len(paths))

    return np.concatenate(out).astype(np.float32)


GRID = 12
DESCRIPTOR_DIM = GRID * GRID + 48 + 16


def _describe(path: Path) -> np.ndarray:
    """Cheap deterministic descriptor: tone map + colour histograms + edge profile."""
    from PIL import Image

    try:
        img = Image.open(path)
        img.draft("RGB", (64, 64))  # JPEG decodes at 1/8 scale — same descriptor, less work
        arr = np.asarray(img.convert("RGB").resize((64, 64)), dtype=np.float32) / 255.0
    except (OSError, ValueError):
        return np.zeros(DESCRIPTOR_DIM, dtype=np.float32)

    gray = arr.mean(axis=2)
    blocks = gray[: (64 // GRID) * GRID, : (64 // GRID) * GRID]
    tone = blocks.reshape(GRID, blocks.shape[0] // GRID, GRID, blocks.shape[1] // GRID).mean(axis=(1, 3)).ravel()

    hist = np.concatenate(
        [np.histogram(arr[..., c], bins=16, range=(0, 1), density=True)[0] for c in range(3)]
    )
    gx = np.abs(np.diff(gray, axis=1)).mean(axis=1)
    gy = np.abs(np.diff(gray, axis=0)).mean(axis=0)
    edges = np.concatenate([gx.reshape(8, -1).mean(axis=1), gy.reshape(8, -1).mean(axis=1)])

    return np.nan_to_num(np.concatenate([tone, hist, edges]).astype(np.float32))


def _fallback_features(paths: Sequence[Path], progress: ProgressFn | None) -> np.ndarray:
    out = np.zeros((len(paths), DESCRIPTOR_DIM), dtype=np.float32)

    with ThreadPoolExecutor(max_workers=IO_WORKERS) as pool:
        for i, vector in enumerate(pool.map(_describe, paths)):
            out[i] = vector
            if progress and i % 500 == 0:
                progress(f"Признаки: {i + 1}/{len(paths)}", (i + 1) / len(paths))

    return out


def _extract(paths: Sequence[Path], progress: ProgressFn | None) -> tuple[np.ndarray, str]:
    """DINOv2 when a GPU can carry it, the cheap descriptor otherwise.

    On CPU the transformer runs at a couple of images per second — fine for a few
    hundred images, hopeless for a whole dataset — so the size of the job decides.
    """
    import torch

    if torch.cuda.is_available() or len(paths) <= DINO_CPU_LIMIT:
        try:
            if progress:
                progress("Загрузка DINOv2", 0.0)
            return _dino_features(paths, progress), f"dinov2:{DINO_MODEL}"
        except Exception as exc:  # noqa: BLE001 - offline, no hub, no GPU memory
            log.warning("DINOv2 недоступна (%s), используется резервный дескриптор", exc)
    else:
        log.info("DINOv2 пропущена: %d изображений на CPU", len(paths))

    return _fallback_features(paths, progress), "fallback-descriptor"


# --------------------------------------------------------------------- store
def cache_path(project: Project) -> Path:
    return project.cache_dir / "embeddings.npz"


def load_cached(project: Project) -> Embeddings | None:
    path = cache_path(project)
    if not path.exists():
        return None
    try:
        data = np.load(path, allow_pickle=True)
        z2d = data["z2d"] if "z2d" in data.files and data["z2d"].size else None
        known = data["known"] if "known" in data.files and data["known"].size else None
        return Embeddings(
            names=[str(n) for n in data["names"]],
            z=data["z"].astype(np.float32),
            z2d=z2d.astype(np.float32) if z2d is not None else None,
            backend=str(data["backend"]) if "backend" in data.files else "cache",
            known=known.astype(bool) if known is not None else None,
        )
    except (OSError, KeyError, ValueError):
        return None


def compute(project: Project, progress: ProgressFn | None = None, force: bool = False) -> Embeddings:
    if not force:
        cached = load_cached(project)
        if cached is not None:
            return cached

    samples = project.samples()
    if not samples:
        raise ValueError("В проекте нет изображений")
    if len(samples) > MAX_EMBED_IMAGES:
        samples = samples[:MAX_EMBED_IMAGES]

    names = [s["name"] for s in samples]
    paths = [Path(s["image"]) for s in samples]

    reference = _from_reference(names)
    known = None
    z2d = None

    if reference is not None and reference.coverage > 0.999:
        z, z2d, known = reference.z, reference.z2d, reference.known
        backend = "dinov2-cache"

    elif reference is not None and reference.coverage >= MIN_ANCHORS:
        # Only the uncovered images plus a set of anchors need local features.
        rng = np.random.default_rng(0)
        covered = np.where(reference.known)[0]
        anchors = rng.choice(covered, size=min(BRIDGE_ANCHORS, len(covered)), replace=False)
        missing = np.where(~reference.known)[0]
        wanted = np.concatenate([anchors, missing])

        if progress:
            progress(
                f"Кэш DINOv2 покрывает {reference.coverage:.0%} — считаем {len(wanted)} признаков",
                0.05,
            )
        feats, source = _extract([paths[i] for i in wanted], progress)

        z = reference.z.copy()
        z[missing] = bridge(feats[: len(anchors)], reference.z[anchors], feats[len(anchors) :])
        z2d, known = reference.z2d, reference.known
        backend = f"dinov2-cache+{source}"

    else:
        feats, backend = _extract(paths, progress)
        z = build_workspace(feats)

    embeddings = Embeddings(
        names=names, z=z.astype(np.float32), z2d=z2d, backend=backend, known=known
    )
    project.cache_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        cache_path(project),
        names=np.array(embeddings.names),
        z=embeddings.z,
        z2d=embeddings.z2d if embeddings.z2d is not None else np.empty(0, dtype=np.float32),
        known=embeddings.known if embeddings.known is not None else np.empty(0, dtype=bool),
        backend=embeddings.backend,
    )
    if progress:
        progress("Эмбеддинги готовы", 1.0)
    return embeddings
