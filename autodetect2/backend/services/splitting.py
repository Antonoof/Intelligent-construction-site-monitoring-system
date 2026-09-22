"""Деление датасета на train/val и выгрузка его в обычную YOLO-структуру."""

from __future__ import annotations

import json
import re
import shutil
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence

import numpy as np
import yaml

from .analysis import load_ground_truth
from .project import Project, utc_now

ProgressFn = Callable[[str, float], None]

# «obj53359_photo2023-07_03.jpeg» -> «obj53359_photo2023-07»: кадры одной съёмки
# отличаются только хвостовым номером и похожи между собой до неразличимости.
SERIES_RE = re.compile(r"^(.*?)[ _-]?\d+$")


def group_key(sample: dict, mode: str) -> str:
    """Что считается одной неделимой пачкой кадров."""
    name = Path(sample["name"]).stem
    if mode == "folder":
        return str(Path(sample["image"]).parent)
    if mode == "series":
        match = SERIES_RE.match(name)
        return match.group(1) if match and match.group(1) else name
    return name


GROUP_MODES = {
    "none": "каждый кадр сам по себе",
    "series": "кадры одной съёмки не разъезжаются (общий префикс имени)",
    "folder": "кадры одной папки не разъезжаются",
}


@dataclass
class SplitPlan:
    train: list[str]
    val: list[str]
    groups: int
    balance: list[dict]
    leak_risk: str
    params: dict

    def as_dict(self) -> dict:
        return {
            "train": len(self.train),
            "val": len(self.val),
            "groups": self.groups,
            "balance": self.balance,
            "leak_risk": self.leak_risk,
            "params": self.params,
            "val_names": self.val[:500],
        }


def _class_vectors(project: Project, names: Sequence[str], nc: int) -> dict[str, np.ndarray]:
    truth = load_ground_truth(project)
    vectors: dict[str, np.ndarray] = {}
    for name in names:
        vector = np.zeros(max(nc, 1), dtype=np.int64)
        entry = truth.get(name)
        if entry is not None and len(entry.classes):
            for cls in entry.classes:
                if 0 <= cls < len(vector):
                    vector[int(cls)] += 1
        vectors[name] = vector
    return vectors


def _stratified_assign(
    groups: dict[str, list[str]],
    vectors: dict[str, np.ndarray],
    val_target: int,
    seed: int,
) -> tuple[list[str], list[str]]:
    """Итеративная стратификация по группам.

    Сначала раздаются группы с самыми редкими классами: если оставить их на
    потом, редкий класс целиком окажется в одной половине и метрика на валидации
    станет лотереей. Внутри — группа уходит туда, где дефицит по этому классу
    больше.
    """
    rng = np.random.default_rng(seed)
    keys = list(groups)
    rng.shuffle(keys)

    mass = {key: np.sum([vectors[n] for n in groups[key]], axis=0) for key in keys}
    total = np.sum(list(mass.values()), axis=0) if mass else np.zeros(1, dtype=np.int64)
    n_images = sum(len(groups[key]) for key in keys)
    share = val_target / max(n_images, 1)

    need_val = total * share
    need_train = total * (1.0 - share)
    room_val = float(val_target)
    room_train = float(n_images - val_target)

    # Группа с редчайшим классом — первой; дальше по убыванию числа боксов.
    def priority(key: str) -> tuple[float, float]:
        vector = mass[key]
        present = np.where(vector > 0)[0]
        rarest = float(total[present].min()) if len(present) else float("inf")
        return (rarest, -float(vector.sum()))

    val: list[str] = []
    train: list[str] = []
    for key in sorted(keys, key=priority):
        vector = mass[key]
        size = len(groups[key])
        present = np.where(vector > 0)[0]

        if not len(present):
            pick_val = room_val / max(room_val + room_train, 1e-9) > rng.random()
        else:
            target = int(present[np.argmin(total[present])])
            gap_val, gap_train = float(need_val[target]), float(need_train[target])
            if abs(gap_val - gap_train) < 1e-9:
                pick_val = room_val > room_train
            else:
                pick_val = gap_val > gap_train

        # Группа целиком больше остатка места в val — брать её значит
        # промахнуться мимо заказанной доли сильнее, чем не брать вовсе.
        if pick_val and val and room_val < size / 2:
            pick_val = False
        if pick_val and room_val <= 0 and room_train > 0:
            pick_val = False
        elif not pick_val and room_train <= 0 and room_val > 0:
            pick_val = True

        if pick_val:
            val += groups[key]
            need_val = need_val - vector
            room_val -= size
        else:
            train += groups[key]
            need_train = need_train - vector
            room_train -= size

    return sorted(train), sorted(val)


def _random_assign(groups: dict[str, list[str]], val_target: int, seed: int) -> tuple[list[str], list[str]]:
    rng = np.random.default_rng(seed)
    keys = list(groups)
    rng.shuffle(keys)

    val: list[str] = []
    for key in keys:
        size = len(groups[key])
        if val and len(val) + size / 2 > val_target:
            continue
        val += groups[key]
        if len(val) >= val_target:
            break

    chosen = set(val)
    train = [name for key in keys for name in groups[key] if name not in chosen]
    return sorted(train), sorted(val)


def plan_split(
    project: Project,
    val_ratio: float | None = 0.2,
    val_count: int | None = None,
    scope: str = "labeled",
    group_by: str = "series",
    stratify: bool = True,
    seed: int = 42,
) -> SplitPlan:
    samples = project.samples()
    if not samples:
        raise ValueError("В проекте нет изображений")

    if scope == "labeled":
        pool = [s for s in samples if s.get("boxes") or s.get("label")]
    else:
        pool = list(samples)
    if not pool:
        raise ValueError("Нет размеченных изображений — делить нечего")

    names = [s["name"] for s in pool]
    if val_count is not None:
        target = int(max(0, min(val_count, len(names))))
    else:
        target = int(round(len(names) * float(val_ratio or 0.0)))
    target = max(0, min(target, len(names)))

    groups: dict[str, list[str]] = defaultdict(list)
    for sample in pool:
        groups[group_key(sample, group_by)].append(sample["name"])

    nc = int(project.meta.get("nc") or 0)
    vectors = _class_vectors(project, names, nc) if stratify and nc else {}

    if stratify and vectors:
        train, val = _stratified_assign(dict(groups), vectors, target, seed)
    else:
        train, val = _random_assign(dict(groups), target, seed)

    class_names = list(project.meta.get("names") or [])
    balance = []
    if vectors:
        train_mass = np.sum([vectors[n] for n in train], axis=0) if train else np.zeros(max(nc, 1), np.int64)
        val_mass = np.sum([vectors[n] for n in val], axis=0) if val else np.zeros(max(nc, 1), np.int64)
        for cls in range(nc):
            total = int(train_mass[cls] + val_mass[cls])
            balance.append(
                {
                    "id": cls,
                    "name": class_names[cls] if cls < len(class_names) else f"class_{cls}",
                    "train": int(train_mass[cls]),
                    "val": int(val_mass[cls]),
                    "val_share": round(int(val_mass[cls]) / total, 3) if total else 0.0,
                }
            )

    # Одинаковый префикс по обе стороны — это утечка: почти такой же кадр
    # окажется и в обучении, и в валидации, и метрика окажется завышенной.
    if group_by == "none":
        train_series = {group_key({"name": n, "image": n}, "series") for n in train}
        shared = sum(1 for n in val if group_key({"name": n, "image": n}, "series") in train_series)
        leak = "none" if not shared else ("high" if shared > len(val) * 0.3 else "some")
    else:
        leak = "none"

    return SplitPlan(
        train=train,
        val=val,
        groups=len(groups),
        balance=balance,
        leak_risk=leak,
        params={
            "scope": scope,
            "group_by": group_by,
            "stratify": bool(stratify and vectors),
            "seed": seed,
            "val_ratio": round(len(val) / max(len(names), 1), 4),
            "requested": {"val_ratio": val_ratio, "val_count": val_count},
        },
    )


def _write_list(path: Path, samples: Sequence[dict]) -> None:
    path.write_text(
        "\n".join(str(Path(s["image"])).replace("\\", "/") for s in samples) + "\n", encoding="utf-8"
    )


def apply_split(project: Project, plan: SplitPlan) -> dict:
    """Проставляет split каждому кадру и перенастраивает data.yaml на списки файлов.

    Список путей вместо папок — единственный способ описать произвольное деление:
    после стратификации train и val перемешаны внутри одних и тех же каталогов, и
    никакая пара директорий их уже не разделит. Ultralytics читает такие списки
    наравне с папками.
    """
    assignment = {name: "val" for name in plan.val}
    assignment.update({name: "train" for name in plan.train})

    samples = [dict(s) for s in project.samples()]
    for sample in samples:
        split = assignment.get(sample["name"])
        if split:
            sample["split"] = split

    project.write_raw_index(samples)

    splits_dir = project.root / "splits"
    splits_dir.mkdir(parents=True, exist_ok=True)
    by_split = {"train": [], "val": []}
    for sample in samples:
        by_split.setdefault(sample.get("split", "train"), []).append(sample)

    _write_list(splits_dir / "train.txt", by_split.get("train", []))
    _write_list(splits_dir / "val.txt", by_split.get("val", []))

    data = yaml.safe_load(project.data_yaml.read_text(encoding="utf-8")) if project.data_yaml.exists() else {}
    data = data or {}
    data["path"] = str(project.root).replace("\\", "/")
    data["train"] = "splits/train.txt"
    data["val"] = "splits/val.txt"
    data["nc"] = int(project.meta.get("nc") or data.get("nc") or 0)
    data["names"] = list(project.meta.get("names") or data.get("names") or [])
    project.data_yaml.write_text(yaml.safe_dump(data, allow_unicode=True, sort_keys=False), encoding="utf-8")

    counts = dict(project.meta.get("counts") or {})
    counts["train"] = len(by_split.get("train", []))
    counts["val"] = len(by_split.get("val", []))
    project.meta["counts"] = counts
    project.meta["split"] = {
        "at": utc_now(),
        "train": len(plan.train),
        "val": len(plan.val),
        **plan.params,
    }
    project.save()

    (project.root / "splits" / "last_split.json").write_text(
        json.dumps({**plan.as_dict(), "at": utc_now()}, ensure_ascii=False), encoding="utf-8"
    )
    return project.meta["split"]


LINK_MODES = {"copy": "копировать", "hardlink": "жёсткие ссылки", "symlink": "символические ссылки"}


def _place(source: Path, target: Path, mode: str) -> None:
    if target.exists():
        return
    if mode == "hardlink":
        try:
            import os

            os.link(source, target)
            return
        except OSError:
            pass
    elif mode == "symlink":
        try:
            target.symlink_to(source)
            return
        except OSError:
            pass
    shutil.copyfile(source, target)


def export_dataset(
    project: Project,
    link_mode: str = "hardlink",
    include_unlabeled: bool = False,
    progress: ProgressFn | None = None,
) -> dict:
    """Выгружает проект в каноническую структуру YOLO.

    Жёсткие ссылки по умолчанию: копия датасета на 4 ГБ ради смены раскладки
    папок — это потерянное место и полчаса ожидания, а ссылка занимает байты и
    создаётся мгновенно. На другой том ссылка не ляжет — тогда происходит
    обычное копирование.
    """
    samples = project.samples()
    if not samples:
        raise ValueError("В проекте нет изображений")

    root = project.exports_dir / f"dataset_{utc_now().replace(':', '').replace('-', '')}"
    for split in ("train", "val"):
        (root / "images" / split).mkdir(parents=True, exist_ok=True)
        (root / "labels" / split).mkdir(parents=True, exist_ok=True)

    written = {"train": 0, "val": 0, "labels": 0, "skipped": 0}
    total = max(len(samples), 1)

    for index, sample in enumerate(samples):
        label = project.label_path(sample)
        if label is None and not include_unlabeled:
            written["skipped"] += 1
            continue

        split = sample.get("split", "train")
        if split not in ("train", "val"):
            split = "train"

        source = Path(sample["image"])
        if not source.exists():
            written["skipped"] += 1
            continue

        _place(source, root / "images" / split / source.name, link_mode)
        written[split] += 1

        if label is not None:
            shutil.copyfile(label, root / "labels" / split / f"{source.stem}.txt")
            written["labels"] += 1
        else:
            (root / "labels" / split / f"{source.stem}.txt").write_text("", encoding="utf-8")

        if progress and index % 200 == 0:
            progress(f"Выгрузка: {index + 1}/{total}", (index + 1) / total)

    names = list(project.meta.get("names") or [])
    (root / "data.yaml").write_text(
        yaml.safe_dump(
            {
                "path": str(root).replace("\\", "/"),
                "train": "images/train",
                "val": "images/val",
                "nc": len(names),
                "names": names,
            },
            allow_unicode=True,
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    if progress:
        progress("Готово", 1.0)
    return {"path": str(root), "link_mode": link_mode, **written}
