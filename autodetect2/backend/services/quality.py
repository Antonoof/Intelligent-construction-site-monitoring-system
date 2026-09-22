"""Аудит разметки: где она сломана, где просто подозрительна.

Модель для этого не нужна. Большая часть брака в YOLO-разметке видна из самих
файлов: бокс вылез за кадр, один объект обведён дважды, «башенный кран» размером
с колесо. Остальное ловится сравнением кадров между собой — если девять сотен
экскаваторов выглядят похоже, а один совсем нет, скорее всего это не экскаватор.
"""

from __future__ import annotations

import json
import math
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence

import numpy as np

from .analysis import load_ground_truth
from .dataset import iou_matrix
from .project import Project, utc_now

ProgressFn = Callable[[str, float], None]

QUALITY_VERSION = 2
CACHE_NAME = "quality.json"

# Порог по каждому виду брака. Значения подобраны так, чтобы срабатывание почти
# всегда означало ошибку, а не редкий, но законный кадр.
BOUNDS_SLACK = 0.004        # бокс считается вылезшим за кадр глубже этой доли
MIN_SIDE = 0.004            # сторона меньше — почти наверняка промах мышью
MIN_AREA = 2.5e-5           # площадь меньше — объект в несколько пикселей
MAX_AREA = 0.88             # бокс почти во весь кадр
MAX_ASPECT = 12.0           # вытянутость, за которой начинается брак
DUPLICATE_IOU = 0.80        # один объект обведён дважды одним классом
CONFLICT_IOU = 0.70         # один объект обведён двумя разными классами
CONTAIN_RATIO = 0.93        # меньший бокс внутри большего
SIZE_Z = 4.5                # робастный z-score по площади внутри класса
CLASS_MARGIN = 0.10         # насколько чужой класс должен быть «ближе» своего
MIN_CLASS_SAMPLES = 25      # меньше примеров — центроид классу не считаем
MIN_SEPARATION = 2.0        # во сколько раз попадание в свой класс должно бить случайное
MAX_CONFLICT_SHARE = 0.05   # доля боксов, которую проверка классов вправе пометить
MAX_CROPS = 30_000          # потолок на число вырезок для проверки классов
RARE_CLASS = 15             # класс с меньшим числом боксов помечается редким

ISSUE_KINDS: dict[str, tuple[str, str]] = {
    "degenerate": ("Нулевой или отрицательный бокс", "high"),
    "out_of_bounds": ("Бокс выходит за границы кадра", "high"),
    "bad_class": ("Класса нет в data.yaml", "high"),
    "duplicate": ("Объект обведён дважды", "high"),
    "class_conflict": ("Похоже, класс перепутан", "high"),
    "overlap": ("Два класса на одном объекте", "medium"),
    "nested": ("Бокс внутри другого бокса", "medium"),
    "tiny": ("Подозрительно маленький бокс", "medium"),
    "giant": ("Бокс почти во весь кадр", "medium"),
    "aspect": ("Неправдоподобная вытянутость", "medium"),
    "size_outlier": ("Размер выбивается из класса", "medium"),
    "empty": ("Файл разметки пуст", "low"),
    "rare_class": ("Очень редкий класс", "low"),
}
SEVERITY_WEIGHT = {"high": 3.0, "medium": 1.0, "low": 0.3}


class Separability(RuntimeError):
    """Вид объектов не разделяет классы — проверять по нему нечего."""


@dataclass(slots=True)
class Issue:
    name: str
    kind: str
    box: int
    detail: str

    @property
    def severity(self) -> str:
        return ISSUE_KINDS[self.kind][1]

    def as_dict(self) -> dict:
        return {
            "kind": self.kind,
            "box": self.box,
            "detail": self.detail,
            "severity": self.severity,
            "label": ISSUE_KINDS[self.kind][0],
        }


# ------------------------------------------------------- признаки вырезок
CROP_SIZE = 40
GRID = 6
CROP_DIM = GRID * GRID + 36 + 16


def _crop_descriptor(patch: np.ndarray) -> np.ndarray:
    """Тон, цвет и профиль краёв вырезки — дёшево и без модели."""
    gray = patch.mean(axis=2)
    step = CROP_SIZE // GRID
    tone = gray[: step * GRID, : step * GRID].reshape(GRID, step, GRID, step).mean(axis=(1, 3)).ravel()
    hist = np.concatenate(
        [np.histogram(patch[..., c], bins=12, range=(0.0, 1.0), density=True)[0] for c in range(3)]
    )
    gx = np.abs(np.diff(gray, axis=1)).mean(axis=1)
    gy = np.abs(np.diff(gray, axis=0)).mean(axis=0)
    edges = np.concatenate([gx.reshape(8, -1).mean(axis=1), gy.reshape(8, -1).mean(axis=1)])
    return np.nan_to_num(np.concatenate([tone, hist, edges]).astype(np.float32))


def _image_crops(job: tuple[str, np.ndarray]) -> np.ndarray:
    """Открывает кадр один раз и вырезает из него все боксы сразу."""
    path, boxes = job
    out = np.zeros((len(boxes), CROP_DIM), dtype=np.float32)
    try:
        from PIL import Image

        img = Image.open(path)
        img.draft("RGB", (1024, 1024))  # JPEG декодируется в уменьшенном масштабе
        img = img.convert("RGB")
    except (OSError, ValueError):
        return out

    width, height = img.size
    for i, (x1, y1, x2, y2) in enumerate(boxes):
        left = max(int(x1 * width), 0)
        top = max(int(y1 * height), 0)
        right = min(int(math.ceil(x2 * width)), width)
        bottom = min(int(math.ceil(y2 * height)), height)
        if right - left < 2 or bottom - top < 2:
            continue
        patch = img.crop((left, top, right, bottom)).resize((CROP_SIZE, CROP_SIZE))
        out[i] = _crop_descriptor(np.asarray(patch, dtype=np.float32) / 255.0)
    return out


def _class_consistency(
    paths: dict[str, str],
    offsets: dict[str, tuple[int, int]],
    boxes: np.ndarray,
    classes: np.ndarray,
    class_names: Sequence[str],
    progress: ProgressFn | None,
) -> list[tuple[int, int, float]]:
    """Ищет боксы, чей вид ближе к другому классу, чем к своему.

    Возвращает список `(индекс бокса, предлагаемый класс, запас уверенности)`.
    Центроид считается по медиане: один-два неверно размеченных объекта не должны
    утаскивать за собой эталон всего класса.
    """
    jobs = [(paths[name], boxes[start:end]) for name, (start, end) in offsets.items() if name in paths]
    if not jobs:
        return []

    total = len(boxes)
    feats = np.zeros((total, CROP_DIM), dtype=np.float32)
    done = 0
    with ThreadPoolExecutor(max_workers=8) as pool:
        for (name, _), chunk in zip(
            [(n, p) for n, p in offsets.items() if n in paths], pool.map(_image_crops, jobs)
        ):
            start, end = offsets[name]
            feats[start:end] = chunk
            done += 1
            if progress and done % 200 == 0:
                progress(f"Вырезки: {done}/{len(jobs)} кадров", 0.5 + 0.35 * done / len(jobs))

    alive = feats.any(axis=1)
    if alive.sum() < MIN_CLASS_SAMPLES * 2:
        return []

    from sklearn.decomposition import PCA

    normed = feats / (np.linalg.norm(feats, axis=1, keepdims=True) + 1e-8)
    dim = int(min(32, CROP_DIM, max(int(alive.sum()) - 1, 2)))
    z = np.zeros((total, dim), dtype=np.float32)
    z[alive] = PCA(n_components=dim, whiten=True, random_state=0).fit_transform(normed[alive])
    z /= np.linalg.norm(z, axis=1, keepdims=True) + 1e-8

    centroids: dict[int, np.ndarray] = {}
    for cls in np.unique(classes):
        if cls < 0 or cls >= len(class_names):
            continue
        rows = z[(classes == cls) & alive]
        if len(rows) >= MIN_CLASS_SAMPLES:
            centre = np.median(rows, axis=0)
            centroids[int(cls)] = centre / (np.linalg.norm(centre) + 1e-8)
    if len(centroids) < 2:
        return []

    keys = sorted(centroids)
    matrix = np.stack([centroids[k] for k in keys])
    similarity = z @ matrix.T
    position = {cls: i for i, cls in enumerate(keys)}

    rows = [i for i in np.where(alive)[0] if int(classes[i]) in position]
    if not rows:
        return []

    # Прежде чем верить этой проверке, надо убедиться, что дескриптор вообще
    # различает классы. Если доля боксов, чей ближайший центроид — свой же
    # класс, не отличается от случайного угадывания, сигнала нет, и «похоже на
    # другой класс» посыпется на половину датасета. Молчать честнее.
    own_is_nearest = sum(
        1 for i in rows if keys[int(np.argmax(similarity[i]))] == int(classes[i])
    )
    agreement = own_is_nearest / len(rows)
    chance = 1.0 / len(keys)
    if agreement < chance * MIN_SEPARATION:
        raise Separability(
            f"классы неразличимы по виду (совпадение {agreement:.0%} против случайных {chance:.0%})"
        )

    suspects: list[tuple[int, int, float]] = []
    for i in rows:
        cls = int(classes[i])
        own = float(similarity[i, position[cls]])
        ranked = np.argsort(-similarity[i])
        best = int(ranked[0]) if keys[int(ranked[0])] != cls else int(ranked[1])
        margin = float(similarity[i, best]) - own
        if margin > CLASS_MARGIN:
            suspects.append((int(i), keys[best], round(margin, 4)))

    # Проверка вероятностная, поэтому сообщается только её верхушка: разбирать
    # четверть датасета никто не станет, а первые полсотни — станет.
    suspects.sort(key=lambda item: -item[2])
    return suspects[: max(int(len(rows) * MAX_CONFLICT_SHARE), 10)]


# --------------------------------------------------------------- проверки
def _geometry_issues(name: str, boxes: np.ndarray, classes: np.ndarray, nc: int) -> list[Issue]:
    issues: list[Issue] = []
    for i, ((x1, y1, x2, y2), cls) in enumerate(zip(boxes, classes)):
        w, h = float(x2 - x1), float(y2 - y1)
        area = max(w, 0.0) * max(h, 0.0)

        if w <= 0 or h <= 0:
            issues.append(Issue(name, "degenerate", i, f"{w:.4f} на {h:.4f}"))
            continue
        if cls < 0 or (nc and cls >= nc):
            issues.append(Issue(name, "bad_class", i, f"class_id={int(cls)}, классов в проекте {nc}"))

        over = float(max(-x1, -y1, x2 - 1.0, y2 - 1.0))
        if over > BOUNDS_SLACK:
            issues.append(Issue(name, "out_of_bounds", i, f"вылет на {over * 100:.1f}% кадра"))
        if area < MIN_AREA or w < MIN_SIDE or h < MIN_SIDE:
            issues.append(Issue(name, "tiny", i, f"{w * 100:.1f}% на {h * 100:.1f}% кадра"))
        elif area > MAX_AREA:
            issues.append(Issue(name, "giant", i, f"{area * 100:.0f}% кадра"))

        aspect = w / max(h, 1e-6)
        if aspect > MAX_ASPECT or aspect < 1.0 / MAX_ASPECT:
            issues.append(Issue(name, "aspect", i, f"стороны как {aspect:.1f}:1"))
    return issues


def _pairwise_issues(
    name: str, boxes: np.ndarray, classes: np.ndarray, class_names: Sequence[str]
) -> list[Issue]:
    if len(boxes) < 2:
        return []

    def label(cls: int) -> str:
        return class_names[cls] if 0 <= cls < len(class_names) else f"class_{cls}"

    issues: list[Issue] = []
    iou = iou_matrix(boxes, boxes)
    areas = np.clip(boxes[:, 2] - boxes[:, 0], 0, None) * np.clip(boxes[:, 3] - boxes[:, 1], 0, None)

    for i in range(len(boxes)):
        for j in range(i + 1, len(boxes)):
            value = float(iou[i, j])
            if value <= 0.05:
                continue
            same = classes[i] == classes[j]
            if same and value > DUPLICATE_IOU:
                issues.append(Issue(name, "duplicate", j, f"IoU {value:.2f} с боксом №{i + 1}"))
                continue
            if not same and value > CONFLICT_IOU:
                issues.append(
                    Issue(
                        name,
                        "overlap",
                        j,
                        f"{label(int(classes[i]))} и {label(int(classes[j]))}, IoU {value:.2f}",
                    )
                )
                continue

            small, big = (i, j) if areas[i] <= areas[j] else (j, i)
            inter_w = min(boxes[i, 2], boxes[j, 2]) - max(boxes[i, 0], boxes[j, 0])
            inter_h = min(boxes[i, 3], boxes[j, 3]) - max(boxes[i, 1], boxes[j, 1])
            inter = max(float(inter_w), 0.0) * max(float(inter_h), 0.0)
            covered = inter / max(float(areas[small]), 1e-9)
            if covered > CONTAIN_RATIO and areas[small] / max(areas[big], 1e-9) < 0.6:
                issues.append(Issue(name, "nested", int(small), f"целиком внутри бокса №{big + 1}"))
    return issues


def _size_outliers(
    labelled: Sequence[str],
    offsets: dict[str, tuple[int, int]],
    boxes: np.ndarray,
    classes: np.ndarray,
    class_names: Sequence[str],
) -> tuple[list[Issue], dict[int, dict]]:
    """Робастный z-score по логарифму площади внутри каждого класса.

    Медиана и MAD вместо среднего и сигмы: выбросы — это ровно то, что мы ищем,
    и они не должны участвовать в определении нормы.
    """
    stats: dict[int, dict] = {}
    flags = np.zeros(len(boxes), dtype=bool)
    details: dict[int, str] = {}

    if not len(boxes):
        return [], stats

    widths = np.clip(boxes[:, 2] - boxes[:, 0], 1e-6, None)
    heights = np.clip(boxes[:, 3] - boxes[:, 1], 1e-6, None)
    log_area = np.log(widths * heights)
    aspect = widths / heights

    for cls in np.unique(classes):
        rows = np.where(classes == cls)[0]
        median = float(np.median(log_area[rows]))
        mad = float(np.median(np.abs(log_area[rows] - median)))
        stats[int(cls)] = {
            "boxes": int(len(rows)),
            "median_area": round(float(np.exp(median)), 6),
            "median_aspect": round(float(np.median(aspect[rows])), 3),
        }
        if len(rows) < MIN_CLASS_SAMPLES or mad <= 1e-6:
            continue

        z = 0.6745 * (log_area[rows] - median) / mad
        hit = rows[np.abs(z) > SIZE_Z]
        flags[hit] = True
        typical = math.exp(median)
        name = class_names[int(cls)] if 0 <= int(cls) < len(class_names) else f"class_{int(cls)}"
        for index in hit:
            ratio = float(np.exp(log_area[index])) / max(typical, 1e-9)
            word = f"в {ratio:.1f} раза больше" if ratio > 1 else f"в {1 / max(ratio, 1e-9):.1f} раза меньше"
            details[int(index)] = f"{word} типичного для «{name}»"

    issues: list[Issue] = []
    for name in labelled:
        start, end = offsets[name]
        for index in range(start, end):
            if flags[index]:
                issues.append(Issue(name, "size_outlier", index - start, details.get(index, "")))
    return issues, stats


# ------------------------------------------------------------------ аудит
def cache_path(project: Project) -> Path:
    return project.cache_dir / CACHE_NAME


def cached(project: Project) -> dict | None:
    path = cache_path(project)
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return payload if payload.get("version") == QUALITY_VERSION else None


def _stack(truth: dict, labelled: Sequence[str]) -> tuple[dict[str, tuple[int, int]], np.ndarray, np.ndarray]:
    offsets: dict[str, tuple[int, int]] = {}
    box_parts: list[np.ndarray] = []
    class_parts: list[np.ndarray] = []
    cursor = 0
    for name in labelled:
        entry = truth[name]
        offsets[name] = (cursor, cursor + len(entry.boxes))
        cursor += len(entry.boxes)
        box_parts.append(entry.boxes)
        class_parts.append(entry.classes)

    boxes = np.concatenate(box_parts) if box_parts else np.empty((0, 4), np.float32)
    classes = np.concatenate(class_parts).astype(np.int32) if class_parts else np.empty(0, np.int32)
    return offsets, boxes, classes


def audit(project: Project, check_classes: bool = True, progress: ProgressFn | None = None) -> dict:
    samples = {s["name"]: s for s in project.samples()}
    if not samples:
        raise ValueError("В проекте нет изображений")

    class_names = list(project.meta.get("names") or [])
    nc = len(class_names)

    if progress:
        progress("Чтение разметки", 0.1)
    truth = load_ground_truth(project)

    labelled: list[str] = []
    empty: list[str] = []
    unlabelled: list[str] = []
    for name in samples:
        entry = truth.get(name)
        if entry is None:
            unlabelled.append(name)
        elif len(entry.boxes):
            labelled.append(name)
        else:
            empty.append(name)

    offsets, boxes, classes = _stack(truth, labelled)

    if progress:
        progress("Геометрия боксов", 0.3)
    issues: list[Issue] = []
    for name in labelled:
        start, end = offsets[name]
        issues += _geometry_issues(name, boxes[start:end], classes[start:end], nc)
        issues += _pairwise_issues(name, boxes[start:end], classes[start:end], class_names)
    for name in empty:
        issues.append(Issue(name, "empty", -1, "ни одного бокса"))

    if progress:
        progress("Размеры внутри классов", 0.45)
    size_issues, class_stats = _size_outliers(labelled, offsets, boxes, classes, class_names)
    issues += size_issues

    for cls, stat in class_stats.items():
        if not 0 < stat["boxes"] < RARE_CLASS:
            continue
        name = class_names[cls] if 0 <= cls < nc else f"class_{cls}"
        for image in labelled:
            start, end = offsets[image]
            for index in range(start, end):
                if classes[index] == cls:
                    issues.append(
                        Issue(image, "rare_class", index - start, f"«{name}»: всего {stat['boxes']} боксов")
                    )

    backend = "geometry"
    if check_classes and len(boxes):
        if progress:
            progress("Сверка классов по виду объектов", 0.5)

        subset = offsets
        if len(boxes) > MAX_CROPS:
            subset, taken = {}, 0
            for name in labelled:
                start, end = offsets[name]
                if taken + (end - start) > MAX_CROPS:
                    break
                subset[name] = (start, end)
                taken += end - start

        try:
            suspects = _class_consistency(
                {name: samples[name]["image"] for name in subset},
                subset,
                boxes,
                classes,
                class_names,
                progress,
            )
            backend = "geometry+crops"
        except Separability as exc:
            suspects, backend = [], f"geometry (сверка классов пропущена: {exc})"
        except Exception as exc:  # noqa: BLE001 - сверка классов необязательна
            suspects, backend = [], f"geometry (сверка классов не выполнена: {exc})"

        owner: dict[int, str] = {}
        for name, (start, end) in offsets.items():
            for index in range(start, end):
                owner[index] = name

        for index, suggested, margin in suspects:
            name = owner.get(index)
            if name is None:
                continue
            label = class_names[suggested] if 0 <= suggested < nc else f"class_{suggested}"
            issues.append(
                Issue(name, "class_conflict", index - offsets[name][0], f"похоже на «{label}», запас {margin:.2f}")
            )

    if progress:
        progress("Сборка отчёта", 0.92)

    per_image: dict[str, list[Issue]] = {}
    for issue in issues:
        per_image.setdefault(issue.name, []).append(issue)

    rows = []
    for name, found in per_image.items():
        start, end = offsets.get(name, (0, 0))
        rows.append(
            {
                "name": name,
                "split": samples.get(name, {}).get("split", "train"),
                "boxes": end - start,
                "score": round(sum(SEVERITY_WEIGHT[i.severity] for i in found), 2),
                "kinds": sorted({i.kind for i in found}),
                "issues": [i.as_dict() for i in sorted(found, key=lambda i: -SEVERITY_WEIGHT[i.severity])],
            }
        )
    rows.sort(key=lambda row: (-row["score"], row["name"]))

    by_kind = []
    for kind, (label, severity) in ISSUE_KINDS.items():
        hits = [i for i in issues if i.kind == kind]
        if hits:
            by_kind.append(
                {
                    "kind": kind,
                    "label": label,
                    "severity": severity,
                    "count": len(hits),
                    "images": len({i.name for i in hits}),
                }
            )
    by_kind.sort(key=lambda item: (-SEVERITY_WEIGHT[item["severity"]], -item["count"]))

    total_boxes = int(len(boxes))
    highest = (int(classes.max()) + 1) if len(classes) else 0
    classes_payload = []
    for cls in range(max(nc, highest)):
        stat = class_stats.get(cls, {"boxes": 0, "median_area": 0.0, "median_aspect": 0.0})
        images_with = sum(
            1 for n in labelled if bool((classes[offsets[n][0] : offsets[n][1]] == cls).any())
        )
        classes_payload.append(
            {
                "id": cls,
                "name": class_names[cls] if cls < nc else f"class_{cls}",
                "known": cls < nc,
                "boxes": stat["boxes"],
                "images": images_with,
                "share": round(stat["boxes"] / max(total_boxes, 1), 4),
                "median_area": stat["median_area"],
                "median_aspect": stat["median_aspect"],
            }
        )

    payload = {
        "version": QUALITY_VERSION,
        "created_at": utc_now(),
        "backend": backend,
        "summary": {
            "images": len(samples),
            "labeled": len(labelled),
            "empty": len(empty),
            "unlabeled": len(unlabelled),
            "boxes": total_boxes,
            "issues": len(issues),
            "images_with_issues": len(per_image),
            "high": sum(1 for i in issues if i.severity == "high"),
            "medium": sum(1 for i in issues if i.severity == "medium"),
            "low": sum(1 for i in issues if i.severity == "low"),
        },
        "by_kind": by_kind,
        "classes": classes_payload,
        "images": rows[:4000],
        "unlabeled_sample": unlabelled[:4000],
    }

    project.cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path(project).write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    project.meta["quality"] = {"at": payload["created_at"], **payload["summary"]}
    project.save()

    if progress:
        progress("Готово", 1.0)
    return payload
