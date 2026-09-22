"""Задания: как отдать часть кадров напарнику и не разметить одно и то же дважды.

Кадр, ушедший в задание, переходит в состояние «в ожидании»: он исчезает из
отбора, из очереди разметки и из кандидатов для следующего задания — до тех пор,
пока не вернётся результат или задание не будет отозвано. Это и есть весь
механизм блокировки; ничего похожего на сервер с правами доступа здесь нет и не
нужно, обмен идёт файлами.
"""

from __future__ import annotations

import json
import zipfile
from pathlib import Path
from typing import Sequence

import numpy as np
import yaml

from ..config import INBOX_DIR, STATUS_ASSIGNED, STATUS_DONE, ensure_workspace
from . import quality as quality_service
from . import workflow
from .dataset import read_label
from .project import Project, utc_now

TASK_VERSION = 1
MANIFEST_NAME = "task.json"


def _stamp() -> str:
    return utc_now().replace(":", "").replace("-", "").replace("+0000", "")


def _record_path(project: Project, assignment_id: str) -> Path:
    return project.assignments_dir / f"{assignment_id}.json"


def _boxes_of(project: Project, sample: dict) -> list[list[float]]:
    label = project.label_path(sample)
    if label is None:
        return []
    rows = read_label(label)
    return [[int(r[0]), round(float(r[1]), 6), round(float(r[2]), 6), round(float(r[3]), 6), round(float(r[4]), 6)] for r in rows]


# ------------------------------------------------------------- кандидаты
def candidates(
    project: Project,
    count: int,
    source: str = "unlabeled",
    split: str | None = None,
) -> list[str]:
    """Кадры, которые имеет смысл отдать: непонятые моделью, битые или пустые.

    Уже занятые и уже закрытые кадры отсюда исключены — иначе напарник получил бы
    ровно то, что вы сейчас размечаете.
    """
    state = workflow.load(project)
    busy = state.busy()
    samples = [s for s in project.samples() if s["name"] not in busy]
    if split:
        samples = [s for s in samples if s.get("split") == split]

    if source == "issues":
        report = quality_service.cached(project)
        if not report:
            raise ValueError("Сначала запустите проверку разметки")
        ranked = [row["name"] for row in report["images"]]
        allowed = {s["name"] for s in samples}
        return [name for name in ranked if name in allowed][:count]

    if source == "selection":
        selections = sorted(project.selections_dir.glob("selection_*.json"), reverse=True)
        if not selections:
            raise ValueError("Сначала подберите изображения на странице разметки")
        payload = json.loads(selections[0].read_text(encoding="utf-8"))
        allowed = {s["name"] for s in samples}
        return [item["name"] for item in payload.get("items", []) if item["name"] in allowed][:count]

    if source == "all":
        pool = samples
    else:  # unlabeled
        pool = [s for s in samples if not s.get("boxes") and not s.get("label")]

    # Равномерно по всему списку, а не первые N подряд: подряд идут кадры одной
    # съёмки, и напарник получил бы один и тот же вид с разницей в минуту.
    if len(pool) <= count:
        return [s["name"] for s in pool]
    step = np.linspace(0, len(pool) - 1, count).round().astype(int)
    return [pool[i]["name"] for i in dict.fromkeys(step.tolist())]


# --------------------------------------------------------------- задания
def create(
    project: Project,
    images: Sequence[str],
    title: str = "",
    owner: str = "",
    note: str = "",
) -> dict:
    known = project.by_name()
    wanted = [name for name in dict.fromkeys(images) if name in known]
    if not wanted:
        raise ValueError("Ни одного из этих изображений нет в проекте")

    assignment_id = f"task-{_stamp()}"
    with workflow.transaction(project) as state:
        busy = state.busy()
        free = [name for name in wanted if name not in busy]
        if not free:
            raise ValueError("Все выбранные изображения уже заняты другим заданием")
        state.set(free, STATUS_ASSIGNED, assignment=assignment_id, owner=owner or None)

    record = {
        "id": assignment_id,
        "title": title or f"Задание {len(free)} кадров",
        "owner": owner,
        "note": note,
        "created_at": utc_now(),
        "status": "open",
        "count": len(free),
        "images": free,
        "skipped": len(wanted) - len(free),
        "exports": [],
    }
    project.assignments_dir.mkdir(parents=True, exist_ok=True)
    _record_path(project, assignment_id).write_text(
        json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return record


def load(project: Project, assignment_id: str) -> dict:
    path = _record_path(project, assignment_id)
    if not path.is_file():
        raise FileNotFoundError(f"Задание не найдено: {assignment_id}")
    return json.loads(path.read_text(encoding="utf-8"))


def save(project: Project, record: dict) -> dict:
    _record_path(project, record["id"]).write_text(
        json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return record


def listing(project: Project) -> list[dict]:
    out = []
    for path in sorted(project.assignments_dir.glob("task-*.json"), reverse=True):
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        out.append({k: v for k, v in record.items() if k != "images"})
    return out


def cancel(project: Project, assignment_id: str) -> dict:
    record = load(project, assignment_id)
    with workflow.transaction(project) as state:
        state.release(record.get("images", []))
    record["status"] = "cancelled"
    record["cancelled_at"] = utc_now()
    return save(project, record)


# ---------------------------------------------------------------- экспорт
def _manifest(project: Project, record: dict) -> dict:
    known = project.by_name()
    return {
        "version": TASK_VERSION,
        "kind": "autodetect2-task",
        "created_at": utc_now(),
        "project": {"id": project.id, "name": project.name},
        "assignment": {k: v for k, v in record.items() if k != "images"},
        "classes": list(project.meta.get("names") or []),
        "images": [
            {
                "name": name,
                "split": known[name].get("split", "train"),
                "boxes": _boxes_of(project, known[name]),
            }
            for name in record["images"]
            if name in known
        ],
    }


def _pack_readme(project: Project, record: dict) -> str:
    return f"""# Задание на разметку · {record.get('title')}

Из проекта «{project.name}», {record['count']} изображений, {utc_now()}.

## Что внутри
- `{MANIFEST_NAME}` — список кадров и уже имеющиеся боксы
- `images/` — сами изображения
- `labels/` — YOLO-разметка (пустой файл = кадр ещё не размечен)
- `data.yaml` — классы; порядок менять нельзя, номер строки это и есть class_id

## Как размечать в AutoDetect2
1. «Создать проект» → выбрать папку `images/` из этого архива;
2. на странице «Датасет» нажать «Принять задание» и выбрать `{MANIFEST_NAME}`
   из этого же архива — кадры встанут в очередь разметки;
3. разметить, затем «Собрать результат» и отправить `*.result.json` обратно.

## Как размечать в другом инструменте
Структура `images/` + `labels/` + `data.yaml` открывается CVAT, LabelImg и
Roboflow как обычный YOLO-датасет. Обратно достаточно прислать папку `labels/`.

## Классы
{chr(10).join(f'{i}. {name}' for i, name in enumerate(project.meta.get('names') or []))}
"""


def export(project: Project, assignment_id: str, mode: str = "manifest") -> Path:
    record = load(project, assignment_id)
    project.exports_dir.mkdir(parents=True, exist_ok=True)
    manifest = _manifest(project, record)

    if mode == "manifest":
        target = project.exports_dir / f"{assignment_id}.adtask.json"
        target.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    else:
        known = project.by_name()
        names = list(project.meta.get("names") or [])
        target = project.exports_dir / f"{assignment_id}.zip"
        with zipfile.ZipFile(target, "w", zipfile.ZIP_DEFLATED) as archive:
            archive.writestr(MANIFEST_NAME, json.dumps(manifest, ensure_ascii=False, indent=2))
            archive.writestr("README.md", _pack_readme(project, record))
            archive.writestr(
                "data.yaml",
                yaml.safe_dump(
                    {"path": ".", "train": "images", "val": "images", "nc": len(names), "names": names},
                    allow_unicode=True,
                    sort_keys=False,
                ),
            )
            for entry in manifest["images"]:
                sample = known.get(entry["name"])
                if sample is None:
                    continue
                source = Path(sample["image"])
                if source.is_file():
                    archive.write(source, f"images/{source.name}")
                lines = "\n".join(
                    f"{int(b[0])} {b[1]:.6f} {b[2]:.6f} {b[3]:.6f} {b[4]:.6f}" for b in entry["boxes"]
                )
                archive.writestr(f"labels/{source.stem}.txt", lines + ("\n" if lines else ""))

    record.setdefault("exports", [])
    if target.name not in record["exports"]:
        record["exports"].append(target.name)
    save(project, record)
    return target


# ------------------------------------------------------------ приём задания
def _read_task(path: Path) -> tuple[dict, Path | None]:
    """Возвращает манифест и папку, куда распакован архив, если это архив."""
    if path.suffix.lower() == ".zip":
        ensure_workspace()
        target = INBOX_DIR / path.stem
        target.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(path) as archive:
            archive.extractall(target)
        manifest_path = target / MANIFEST_NAME
        if not manifest_path.is_file():
            raise ValueError(f"В архиве нет {MANIFEST_NAME}")
        return json.loads(manifest_path.read_text(encoding="utf-8")), target

    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("kind") not in ("autodetect2-task", "autodetect2-result"):
        raise ValueError("Это не файл задания AutoDetect2")
    return payload, None


def accept(project: Project, path: Path) -> dict:
    """Принимает задание на своей стороне.

    Манифест ложится на уже имеющиеся кадры: если картинки те же самые (общий
    диск, одна и та же папка), разметчику достаточно списка имён. Архив
    распаковывается в `workspace/inbox`, и из этой папки создаётся отдельный
    проект — на чужой машине исходных путей всё равно не существует.
    """
    manifest, extracted = _read_task(Path(path))
    entries = manifest.get("images") or []
    known = project.by_name()
    matched = [e["name"] for e in entries if e["name"] in known]

    record = {
        "id": manifest.get("assignment", {}).get("id") or f"task-{_stamp()}",
        "title": manifest.get("assignment", {}).get("title") or "Принятое задание",
        "owner": manifest.get("assignment", {}).get("owner", ""),
        "note": manifest.get("assignment", {}).get("note", ""),
        "created_at": utc_now(),
        "status": "accepted",
        "incoming": True,
        "source_project": manifest.get("project", {}),
        "count": len(matched),
        "images": matched,
        "extracted_to": str(extracted) if extracted else None,
        "exports": [],
    }

    if matched:
        with workflow.transaction(project) as state:
            state.set(matched, STATUS_ASSIGNED, assignment=record["id"], owner=record["owner"] or "я")
        project.assignments_dir.mkdir(parents=True, exist_ok=True)
        save(project, record)

        # Очередь разметки читает выборки — задание становится обычной очередью.
        project.selections_dir.mkdir(parents=True, exist_ok=True)
        (project.selections_dir / f"selection_{_stamp()}.json").write_text(
            json.dumps(
                {
                    "items": [
                        {
                            "name": name,
                            "cluster": 0,
                            "interest": 0.0,
                            "reason": "assignment",
                            "reason_label": "Из задания",
                            "split": known[name].get("split", "train"),
                            "n_gt": known[name].get("boxes", 0),
                            "n_pred": 0,
                            "ghost": 0,
                            "missed": 0,
                            "rank": i + 1,
                        }
                        for i, name in enumerate(matched)
                    ],
                    "clusters": [],
                    "signals": {"with_submission": False, "clusters": 1, "source": "assignment"},
                    "backend": "assignment",
                    "created_at": utc_now(),
                    "budget": len(matched),
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

    return {
        "assignment": {k: v for k, v in record.items() if k != "images"},
        "matched": len(matched),
        "missing": len(entries) - len(matched),
        "extracted_to": record["extracted_to"],
        "classes": manifest.get("classes", []),
    }


def export_result(project: Project, assignment_id: str) -> Path:
    """Собирает текущую разметку по кадрам задания в один файл для отправки."""
    record = load(project, assignment_id)
    known = project.by_name()
    images = []
    labelled = 0
    boxes = 0
    for name in record["images"]:
        sample = known.get(name)
        if sample is None:
            continue
        found = _boxes_of(project, sample)
        labelled += 1 if found else 0
        boxes += len(found)
        images.append({"name": name, "boxes": found})

    payload = {
        "version": TASK_VERSION,
        "kind": "autodetect2-result",
        "created_at": utc_now(),
        "assignment": {k: v for k, v in record.items() if k != "images"},
        "project": {"id": project.id, "name": project.name},
        "classes": list(project.meta.get("names") or []),
        "stats": {"images": len(images), "labeled": labelled, "boxes": boxes},
        "images": images,
    }

    project.exports_dir.mkdir(parents=True, exist_ok=True)
    target = project.exports_dir / f"{assignment_id}.result.json"
    target.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return target


# ------------------------------------------------------- импорт результата
def _labels_from_folder(folder: Path) -> dict[str, list[list[float]]]:
    out: dict[str, list[list[float]]] = {}
    for path in folder.rglob("*.txt"):
        rows = read_label(path)
        out[path.stem] = [[int(r[0]), *(round(float(v), 6) for v in r[1:5])] for r in rows]
    return out


def import_result(project: Project, path: str | Path, assignment_id: str | None = None) -> dict:
    """Вливает чужую разметку в проект и закрывает задание.

    Принимается `*.result.json`, zip с папкой `labels/` или просто папка с
    `*.txt`. Во всех трёх случаях сопоставление идёт по имени файла без
    расширения — ровно так же, как AutoDetect2 ищет разметку на диске.
    """
    source = Path(path)
    incoming: dict[str, list[list[float]]] = {}

    if source.is_dir():
        incoming = _labels_from_folder(source)
    elif source.suffix.lower() == ".zip":
        ensure_workspace()
        target = INBOX_DIR / f"{source.stem}-result"
        target.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(source) as archive:
            archive.extractall(target)
        incoming = _labels_from_folder(target)
    else:
        payload = json.loads(source.read_text(encoding="utf-8"))
        if payload.get("kind") not in ("autodetect2-result", "autodetect2-task"):
            raise ValueError("Это не результат задания AutoDetect2")
        assignment_id = assignment_id or payload.get("assignment", {}).get("id")
        for entry in payload.get("images", []):
            incoming[Path(entry["name"]).stem] = entry.get("boxes") or []

    known = {Path(name).stem: name for name in project.by_name()}
    project.annotations_dir.mkdir(parents=True, exist_ok=True)

    written, boxes, unknown = 0, 0, 0
    touched: list[str] = []
    for stem, rows in incoming.items():
        name = known.get(stem)
        if name is None:
            unknown += 1
            continue
        body = "\n".join(
            f"{int(r[0])} {float(r[1]):.6f} {float(r[2]):.6f} {float(r[3]):.6f} {float(r[4]):.6f}"
            for r in rows
            if len(r) >= 5 and float(r[3]) > 0 and float(r[4]) > 0
        )
        (project.annotations_dir / f"{stem}.txt").write_text(body + ("\n" if body else ""), encoding="utf-8")
        written += 1
        boxes += len(rows)
        touched.append(name)

    with workflow.transaction(project) as state:
        state.set(touched, STATUS_DONE, assignment=assignment_id, source="import")

    record = None
    if assignment_id:
        try:
            record = load(project, assignment_id)
            record["status"] = "imported"
            record["imported_at"] = utc_now()
            record["result"] = {"images": written, "boxes": boxes}
            save(project, record)
        except FileNotFoundError:
            record = None

    for cache in ("dashboard.json", "heatmap.json", "quality.json", "ground_truth.npz"):
        (project.cache_dir / cache).unlink(missing_ok=True)

    return {
        "images": written,
        "boxes": boxes,
        "unknown": unknown,
        "assignment": {k: v for k, v in record.items() if k != "images"} if record else None,
    }
