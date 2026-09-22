from __future__ import annotations

import json
import re
import shutil
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import yaml

from ..config import CLASS_NAMES, MODEL_FAMILIES, PROJECTS_DIR, ensure_workspace
from .dataset import ScanResult, Sample, class_names, scan_sources

SLUG_RE = re.compile(r"[^a-z0-9._-]+")

# Кириллица в имени папки проекта превращается в один дефис, и все проекты
# начинают называться «project», «project-2», «project-3».
TRANSLIT = str.maketrans(
    {
        "а": "a", "б": "b", "в": "v", "г": "g", "д": "d", "е": "e", "ё": "e",
        "ж": "zh", "з": "z", "и": "i", "й": "y", "к": "k", "л": "l", "м": "m",
        "н": "n", "о": "o", "п": "p", "р": "r", "с": "s", "т": "t", "у": "u",
        "ф": "f", "х": "h", "ц": "c", "ч": "ch", "ш": "sh", "щ": "sch",
        "ъ": "", "ы": "y", "ь": "", "э": "e", "ю": "yu", "я": "ya",
    }
)


def slugify(value: str) -> str:
    slug = SLUG_RE.sub("-", value.strip().lower().translate(TRANSLIT)).strip("-.")
    return slug or "project"


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


@dataclass
class Project:
    root: Path
    meta: dict[str, Any]

    # ------------------------------------------------------------------ paths
    @property
    def id(self) -> str:
        return self.meta["id"]

    @property
    def name(self) -> str:
        return self.meta.get("name", self.id)

    @property
    def meta_path(self) -> Path:
        return self.root / "project.json"

    @property
    def index_path(self) -> Path:
        return self.root / "index.json"

    @property
    def data_yaml(self) -> Path:
        return self.root / "data.yaml"

    @property
    def train_yaml(self) -> Path:
        return self.root / "train.yaml"

    @property
    def cache_dir(self) -> Path:
        return self.root / "cache"

    @property
    def submissions_dir(self) -> Path:
        return self.root / "submissions"

    @property
    def exports_dir(self) -> Path:
        return self.root / "exports"

    @property
    def models_dir(self) -> Path:
        return self.root / "models"

    @property
    def annotations_dir(self) -> Path:
        return self.root / "annotations"

    @property
    def selections_dir(self) -> Path:
        return self.root / "selections"

    @property
    def assignments_dir(self) -> Path:
        return self.root / "assignments"

    @property
    def splits_dir(self) -> Path:
        return self.root / "splits"

    def dirs(self) -> list[Path]:
        return [
            self.cache_dir,
            self.submissions_dir,
            self.exports_dir,
            self.models_dir,
            self.annotations_dir,
            self.selections_dir,
            self.assignments_dir,
            self.splits_dir,
        ]

    # ------------------------------------------------------------------ state
    def save(self) -> None:
        self.meta["updated_at"] = utc_now()
        self.meta_path.write_text(
            json.dumps(self.meta, indent=2, ensure_ascii=False), encoding="utf-8"
        )

    def samples(self) -> list[dict]:
        if not self.index_path.exists():
            return []
        cached = _INDEX_CACHE.get(self.id)
        stamp = self.index_path.stat().st_mtime
        if cached and cached[0] == stamp:
            return cached[1]
        data = json.loads(self.index_path.read_text(encoding="utf-8"))
        _INDEX_CACHE[self.id] = (stamp, data)
        return data

    def write_index(self, samples: Sequence[Sample]) -> None:
        self.write_raw_index([s.as_dict() for s in samples])

    def write_raw_index(self, rows: Sequence[dict]) -> None:
        self.index_path.write_text(json.dumps(list(rows)), encoding="utf-8")
        _INDEX_CACHE.pop(self.id, None)
        _LOOKUP_CACHE.pop(self.id, None)

    def label_path(self, sample: dict) -> Path | None:
        """Labels edited inside AutoDetect2 shadow the ones from the source folder."""
        override = self.annotations_dir / f"{Path(sample['name']).stem}.txt"
        if override.exists():
            return override
        return Path(sample["label"]) if sample.get("label") else None

    def by_name(self) -> dict[str, dict]:
        """Индекс имя -> кадр. Редактор дёргает его на каждое нажатие стрелки,
        и линейный поиск по сорока тысячам записей там сразу заметен."""
        stamp = self.index_path.stat().st_mtime if self.index_path.exists() else 0.0
        cached = _LOOKUP_CACHE.get(self.id)
        if cached and cached[0] == stamp:
            return cached[1]
        table = {sample["name"]: sample for sample in self.samples()}
        _LOOKUP_CACHE[self.id] = (stamp, table)
        return table

    def sample_by_name(self, name: str) -> dict | None:
        return self.by_name().get(name)

    @property
    def submission_path(self) -> Path | None:
        info = self.meta.get("submission")
        if not info:
            return None
        path = Path(info["path"])
        return path if path.exists() else None

    def summary(self) -> dict:
        return {
            **self.meta,
            "root": str(self.root),
            "has_submission": self.submission_path is not None,
            "has_embeddings": (self.cache_dir / "embeddings.npz").exists(),
            "exports": sorted(p.name for p in self.exports_dir.glob("*.zip")),
            "models": sorted(p.name for p in self.models_dir.glob("*.pt")),
        }


_INDEX_CACHE: dict[str, tuple[float, list[dict]]] = {}
_LOOKUP_CACHE: dict[str, tuple[float, dict[str, dict]]] = {}


# ---------------------------------------------------------------- persistence
def project_path(project_id: str) -> Path:
    return PROJECTS_DIR / project_id


def load_project(project_id: str) -> Project:
    root = project_path(project_id)
    meta_file = root / "project.json"
    if not meta_file.exists():
        raise FileNotFoundError(f"Проект не найден: {project_id}")
    meta = json.loads(meta_file.read_text(encoding="utf-8"))
    return Project(root=root, meta=meta)


def load_project_from_path(path: str | Path) -> Project:
    """Imports an existing project folder from anywhere on disk."""
    source = Path(path)
    if source.is_file() and source.name == "project.json":
        source = source.parent
    meta_file = source / "project.json"
    if not meta_file.exists():
        raise FileNotFoundError(f"В папке нет project.json: {source}")

    meta = json.loads(meta_file.read_text(encoding="utf-8"))
    ensure_workspace()
    target = project_path(meta["id"])
    if source.resolve() != target.resolve():
        if target.exists():
            meta["id"] = f"{meta['id']}-{int(time.time())}"
            target = project_path(meta["id"])
        shutil.copytree(source, target)
        (target / "project.json").write_text(
            json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8"
        )
    return load_project(meta["id"])


def list_projects() -> list[dict]:
    ensure_workspace()
    items = []
    for meta_file in PROJECTS_DIR.glob("*/project.json"):
        try:
            meta = json.loads(meta_file.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        items.append(
            {
                "id": meta.get("id", meta_file.parent.name),
                "name": meta.get("name", meta_file.parent.name),
                "created_at": meta.get("created_at"),
                "updated_at": meta.get("updated_at"),
                "counts": meta.get("counts", {}),
                "nc": meta.get("nc", 0),
                "root": str(meta_file.parent),
            }
        )
    return sorted(items, key=lambda item: item.get("updated_at") or "", reverse=True)


def delete_project(project_id: str) -> None:
    root = project_path(project_id)
    if root.exists():
        shutil.rmtree(root)
    _INDEX_CACHE.pop(project_id, None)
    _LOOKUP_CACHE.pop(project_id, None)


# ------------------------------------------------------------------- creation
def unique_project_id(name: str) -> str:
    base = slugify(name)
    candidate = base
    counter = 2
    while project_path(candidate).exists():
        candidate = f"{base}-{counter}"
        counter += 1
    return candidate


def write_data_yaml(project: Project, scan: ScanResult, names: list[str]) -> None:
    train_dirs, val_dirs = [], []
    for source in scan.sources:
        target = train_dirs if source["split"] == "train" else val_dirs
        path = Path(source["path"])
        images_dir = path / "images" if (path / "images").is_dir() else path
        target.append(str(images_dir).replace("\\", "/"))

    payload = {
        "path": str(project.root).replace("\\", "/"),
        "train": train_dirs or val_dirs,
        "val": val_dirs or train_dirs,
        "nc": len(names),
        "names": names,
    }
    project.data_yaml.write_text(
        yaml.safe_dump(payload, allow_unicode=True, sort_keys=False), encoding="utf-8"
    )


def update_classes(project: Project, names: Sequence[str]) -> list[str]:
    """Renames, adds or drops classes; data.yaml follows the project metadata."""
    cleaned = [str(name).strip() or f"class_{index}" for index, name in enumerate(names)]

    project.meta["names"] = cleaned
    project.meta["nc"] = len(cleaned)
    project.save()

    if project.data_yaml.exists():
        data = yaml.safe_load(project.data_yaml.read_text(encoding="utf-8")) or {}
        data["nc"] = len(cleaned)
        data["names"] = cleaned
        project.data_yaml.write_text(
            yaml.safe_dump(data, allow_unicode=True, sort_keys=False), encoding="utf-8"
        )
    return cleaned


def default_train_config(family: str = "11", size: str = "x") -> dict:
    return {
        "model": {
            "family": family,
            "size": size,
            "ensemble": [
                {"family": key, "size": size, "imgsz": spec["imgsz"], "enabled": key == family}
                for key, spec in MODEL_FAMILIES.items()
            ],
        },
        "runtime": {"device": "auto", "workers": 4, "seed": 42, "deterministic": False},
        "train": {
            "epochs": 30,
            "imgsz": MODEL_FAMILIES[family]["imgsz"],
            "batch": -1,
            "patience": 100,
            "optimizer": "SGD",
            "lr0": 0.001,
            "lrf": 0.001,
            "weight_decay": 0.0003,
            "cos_lr": True,
            "warmup_epochs": 3,
            "warmup_momentum": 0.95,
            "close_mosaic": 5,
            "freeze": 4,
            "save_period": 10,
            "conf": 0.001,
            "iou": 0.3,
        },
        "augment": {
            "augment": True,
            "hsv_h": 0.015,
            "hsv_s": 0.02,
            "hsv_v": 0.02,
            "flipud": 0.0,
            "fliplr": 0.5,
            "translate": 0.01,
            "scale": 0.25,
            "shear": 0.05,
            "mixup": 0.05,
            "cutmix": 0.1,
        },
        "predict": {"conf": 0.001, "iou": 0.7, "max_det": 300, "submission": "submission.csv"},
    }


def write_train_yaml(project: Project, config: dict | None = None) -> dict:
    config = config or default_train_config()
    project.train_yaml.write_text(
        yaml.safe_dump(config, allow_unicode=True, sort_keys=False), encoding="utf-8"
    )
    return config


def read_train_yaml(project: Project) -> dict:
    if not project.train_yaml.exists():
        return write_train_yaml(project)
    return yaml.safe_load(project.train_yaml.read_text(encoding="utf-8")) or {}


def create_project(
    name: str,
    sources: Sequence[dict],
    class_labels: Sequence[str] | None = None,
    progress=None,
) -> Project:
    if not sources:
        raise ValueError("Не выбрано ни одной папки с данными")
    if not any(s.get("split") == "train" for s in sources):
        raise ValueError("Нужна хотя бы одна папка train")

    ensure_workspace()
    project_id = unique_project_id(name)
    root = project_path(project_id)
    root.mkdir(parents=True)

    project = Project(
        root=root,
        meta={
            "id": project_id,
            "name": name.strip() or project_id,
            "created_at": utc_now(),
            "updated_at": utc_now(),
        },
    )
    for directory in project.dirs():
        directory.mkdir(parents=True, exist_ok=True)

    scan = scan_sources(sources, progress=progress)
    names = class_names(scan.class_ids, class_labels or CLASS_NAMES)

    project.write_index(scan.samples)
    write_data_yaml(project, scan, names)
    write_train_yaml(project)

    splits = {"train": 0, "val": 0}
    for sample in scan.samples:
        splits[sample.split] = splits.get(sample.split, 0) + 1

    project.meta.update(
        {
            "sources": scan.sources,
            "nc": len(names),
            "names": names,
            "counts": {
                "total": len(scan.samples),
                "train": splits.get("train", 0),
                "val": splits.get("val", 0),
                "labeled": sum(1 for s in scan.samples if s.label),
                "boxes": scan.boxes,
            },
            "unlabeled": sum(1 for s in scan.samples if not s.label),
            "data_yaml": str(project.data_yaml),
            "train_yaml": str(project.train_yaml),
            "submission": None,
        }
    )
    project.save()
    return project
