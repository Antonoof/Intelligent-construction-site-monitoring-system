from __future__ import annotations

import os
import time
from pathlib import Path

import yaml

from ..config import IMAGE_SUFFIXES, LABEL_SUFFIX

SPLIT_HINTS = {
    "train": "train",
    "training": "train",
    "val": "val",
    "valid": "val",
    "validation": "val",
    "test": "val",
    "eval": "val",
}
SHALLOW_CAP = 4000
DEEP_CAP = 400_000
BUDGET = 0.6


def is_image(name: str) -> bool:
    return os.path.splitext(name)[1].lower() in IMAGE_SUFFIXES


def guess_split(path: str | Path) -> str:
    for part in reversed(Path(path).parts):
        hint = SPLIT_HINTS.get(part.lower())
        if hint:
            return hint
    return "train"


def shallow_counts(path: str, cap: int = SHALLOW_CAP) -> dict:
    """Counts media in one directory without touching the filesystem twice per entry."""
    images = labels = subdirs = 0
    truncated = False
    try:
        with os.scandir(path) as entries:
            for index, entry in enumerate(entries):
                if index >= cap:
                    truncated = True
                    break
                if entry.is_dir(follow_symlinks=False):
                    subdirs += 1
                elif is_image(entry.name):
                    images += 1
                elif entry.name.lower().endswith(LABEL_SUFFIX):
                    labels += 1
    except (PermissionError, OSError):
        return {"images": 0, "labels": 0, "subdirs": 0, "truncated": False}
    return {"images": images, "labels": labels, "subdirs": subdirs, "truncated": truncated}


def deep_counts(root: str, cap: int = DEEP_CAP, budget: float = 4.0) -> dict:
    """Recursive media census — os.walk keeps this fast even on 50k-file datasets.

    The budget matters when the browser sits on a drive root: the answer stays
    approximate rather than walking the whole disk.
    """
    images = labels = 0
    seen = 0
    truncated = False
    deadline = time.monotonic() + budget

    for _, _, files in os.walk(root):
        for name in files:
            seen += 1
            if is_image(name):
                images += 1
            elif name.lower().endswith(LABEL_SUFFIX):
                labels += 1
        if seen > cap or time.monotonic() > deadline:
            truncated = True
            break
    return {"images": images, "labels": labels, "truncated": truncated}


def pairing(root: str, sample: int = 200) -> dict:
    """YOLO labels usually live in a mirrored `labels/` tree, so count real pairs."""
    from .dataset import resolve_label

    checked = paired = 0
    example = None

    for current, dirs, files in os.walk(root):
        dirs.sort()
        for name in sorted(files):
            if not is_image(name):
                continue
            checked += 1
            label = resolve_label(Path(current) / name)
            if label is not None:
                paired += 1
                example = example or str(label)
            if checked >= sample:
                break
        if checked >= sample:
            break

    return {
        "checked": checked,
        "paired": paired,
        "ratio": round(paired / checked, 3) if checked else 0.0,
        "label_example": example,
    }


def preview_images(path: str, limit: int = 8) -> list[str]:
    found: list[str] = []
    for current, dirs, files in os.walk(path):
        dirs.sort()
        for name in sorted(files):
            if is_image(name):
                found.append(os.path.join(current, name))
                if len(found) >= limit:
                    return found
    return found


def list_dir(path: str, show_files: bool = False, preview: int = 8) -> dict:
    target = Path(path).expanduser()
    if not target.is_dir():
        raise FileNotFoundError(f"Папка не найдена: {path}")

    dirs: list[dict] = []
    files: list[dict] = []
    images = labels = 0
    truncated = False

    with os.scandir(target) as entries:
        for index, entry in enumerate(entries):
            if index >= 20_000:
                truncated = True
                break
            if entry.name.startswith("."):
                continue
            try:
                if entry.is_dir(follow_symlinks=False):
                    dirs.append({"name": entry.name, "path": entry.path})
                    continue
            except OSError:
                continue

            if is_image(entry.name):
                images += 1
            elif entry.name.lower().endswith(LABEL_SUFFIX):
                labels += 1
            if show_files and len(files) < 500:
                files.append({"name": entry.name, "path": entry.path})

    dirs.sort(key=lambda item: item["name"].lower())
    deadline = time.monotonic() + BUDGET
    for item in dirs:
        if time.monotonic() > deadline:
            item["images"] = None
            continue
        counts = shallow_counts(item["path"])
        item.update(images=counts["images"], labels=counts["labels"], subdirs=counts["subdirs"])

    deep = deep_counts(str(target), budget=BUDGET)

    return {
        "path": str(target),
        "parent": str(target.parent) if target.parent != target else None,
        "dirs": dirs,
        "files": sorted(files, key=lambda item: item["name"].lower()),
        "images": images,
        "labels": labels,
        "deep_images": deep["images"],
        "deep_truncated": deep["truncated"],
        "truncated": truncated,
        "preview": preview_images(str(target), preview) if preview else [],
        "split": guess_split(target),
    }


def dataset_children(path: str) -> list[dict]:
    """Split-like subfolders that hold images — `images/{train,val,test}` and friends."""
    children = []
    with os.scandir(path) as entries:
        for entry in entries:
            if not entry.is_dir(follow_symlinks=False) or entry.name.startswith("."):
                continue
            counts = deep_counts(entry.path)
            if counts["images"]:
                children.append(
                    {
                        "path": entry.path,
                        "name": entry.name,
                        "split": guess_split(entry.path),
                        **counts,
                        "pairing": pairing(entry.path),
                    }
                )
    return sorted(children, key=lambda item: item["name"].lower())


def inspect(path: str) -> dict:
    target = Path(path).expanduser()
    if not target.is_dir():
        raise FileNotFoundError(f"Папка не найдена: {path}")

    counts = deep_counts(str(target))
    direct = shallow_counts(str(target))
    return {
        "path": str(target),
        "split": guess_split(target),
        "direct_images": direct["images"],
        "children": dataset_children(str(target)) if not direct["images"] else [],
        "pairing": pairing(str(target)),
        **counts,
    }


def search(root: str, pattern: str, limit: int = 200) -> list[dict]:
    base = Path(root).expanduser()
    if not base.is_dir():
        raise FileNotFoundError(f"Папка не найдена: {root}")

    matches = []
    for path in base.rglob(pattern):
        if path.is_file():
            matches.append({"name": path.name, "path": str(path), "size": path.stat().st_size})
        if len(matches) >= limit:
            break
    return matches


# ------------------------------------------------------------------- yaml
def _as_list(value) -> list[str]:
    if not value:
        return []
    if isinstance(value, str):
        return [value]
    return [str(item) for item in value]


def _resolve(entry: str, yaml_dir: Path, base: str | None) -> Path | None:
    """A data.yaml written on another machine still points at real folders here."""
    candidates: list[Path] = []
    raw = Path(entry)
    if raw.is_absolute():
        candidates.append(raw)
    candidates.append(yaml_dir / entry)
    if base:
        candidates.append(Path(base) / entry)
        candidates.append(yaml_dir / Path(base).name / entry)
    candidates.append(yaml_dir.parent / entry)

    for candidate in candidates:
        if candidate.is_dir():
            return candidate
        images = candidate.parent / "images" / candidate.name
        if images.is_dir():
            return images
    return None


def parse_data_yaml(path: str) -> dict:
    """Reads a YOLO data.yaml: class names plus whatever splits exist on this machine."""
    file = Path(path).expanduser()
    if not file.is_file():
        raise FileNotFoundError(f"Файл не найден: {path}")

    payload = yaml.safe_load(file.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise ValueError("Ожидается YAML-словарь с ключами names / train / val")

    raw_names = payload.get("names")
    if isinstance(raw_names, dict):
        names = [str(raw_names[key]) for key in sorted(raw_names, key=lambda k: int(k))]
    else:
        names = [str(item) for item in _as_list(raw_names)]

    if not names and payload.get("nc"):
        names = [f"class_{i}" for i in range(int(payload["nc"]))]

    yaml_dir = file.parent
    base = payload.get("path")
    sources: list[dict] = []
    missing: list[str] = []

    for key, split in (("train", "train"), ("val", "val"), ("test", "val")):
        for entry in _as_list(payload.get(key)):
            resolved = _resolve(entry, yaml_dir, str(base) if base else None)
            if resolved is None:
                missing.append(entry)
                continue
            counts = deep_counts(str(resolved))
            sources.append(
                {
                    "path": str(resolved),
                    "split": split,
                    "key": key,
                    "images": counts["images"],
                    "labels": counts["labels"],
                    "pairing": pairing(str(resolved)),
                }
            )

    return {
        "file": str(file),
        "names": names,
        "nc": int(payload.get("nc") or len(names)),
        "sources": sources,
        "missing": missing,
    }
