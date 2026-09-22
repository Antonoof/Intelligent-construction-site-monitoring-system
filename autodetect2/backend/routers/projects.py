from __future__ import annotations

import shutil
from pathlib import Path

from fastapi import APIRouter, File, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse

from ..config import STATUS_DONE, UPLOADS_DIR, ensure_workspace
from ..schemas import ClassesIn, LabelsIn, OpenProjectIn, PathIn, ProjectIn
from ..services import project as project_service
from ..services import proposals as proposals_service
from ..services import submission as submission_service
from ..services import workflow
from ..services.dataset import read_label
from ..services.tasks import registry

MEDIA_TYPES = {".zip": "application/zip", ".json": "application/json", ".csv": "text/csv"}


def write_label_file(
    project: project_service.Project,
    sample: dict,
    boxes: list[dict],
    write_source: bool = False,
) -> Path:
    """Пишет YOLO-файл в папку правок проекта, не трогая исходные данные.

    Исходная папка переписывается только по явному флажку: чужой датасет на
    сетевом диске должен оставаться ровно таким, каким его прислали, пока
    разметчик сам не решит иначе.
    """
    lines = [
        f"{int(box['cls'])} {float(box['x']):.6f} {float(box['y']):.6f} {float(box['w']):.6f} {float(box['h']):.6f}"
        for box in boxes
        if float(box.get("w", 0)) > 0 and float(box.get("h", 0)) > 0
    ]
    body = "".join(f"{line}\n" for line in lines)

    project.annotations_dir.mkdir(parents=True, exist_ok=True)
    target = project.annotations_dir / f"{Path(sample['name']).stem}.txt"
    target.write_text(body, encoding="utf-8")

    if write_source and sample.get("label"):
        Path(sample["label"]).write_text(body, encoding="utf-8")
    return target

router = APIRouter(prefix="/api/projects", tags=["projects"])


def get_project(project_id: str) -> project_service.Project:
    try:
        return project_service.load_project(project_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.get("")
def list_projects() -> dict:
    return {"projects": project_service.list_projects()}


@router.post("")
def create_project(payload: ProjectIn) -> dict:
    sources = [s.model_dump() for s in payload.sources]
    if not sources:
        raise HTTPException(status_code=400, detail="Не выбраны папки с данными")

    def job(ctx):
        project = project_service.create_project(
            name=payload.name,
            sources=sources,
            class_labels=payload.classes,
            progress=lambda message, value: ctx.progress(message, value),
        )
        return project.summary()

    task = registry.submit("project.create", f"Создание проекта «{payload.name}»", job)
    return {"task": task.as_dict()}


@router.post("/open")
def open_project(payload: OpenProjectIn) -> dict:
    try:
        project = project_service.load_project_from_path(payload.path)
    except (FileNotFoundError, OSError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"project": project.summary()}


@router.get("/{project_id}")
def project_detail(project_id: str) -> dict:
    return {"project": get_project(project_id).summary()}


@router.delete("/{project_id}")
def remove_project(project_id: str) -> dict:
    project_service.delete_project(project_id)
    return {"ok": True}


@router.put("/{project_id}/classes")
def update_classes(project_id: str, payload: ClassesIn) -> dict:
    project = get_project(project_id)
    names = project_service.update_classes(project, payload.names)
    return {"names": names, "nc": len(names)}


@router.get("/{project_id}/samples")
def samples(
    project_id: str,
    offset: int = 0,
    limit: int = Query(default=60, le=500),
    split: str | None = None,
    q: str | None = None,
    labeled: bool | None = None,
    status: str | None = None,
) -> dict:
    project = get_project(project_id)
    state = workflow.load(project)
    rows = [{**row, "status": state.status(row["name"])} for row in project.samples()]

    if split:
        rows = [s for s in rows if s.get("split") == split]
    if labeled is not None:
        rows = [s for s in rows if bool(s.get("label")) == labeled]
    if q:
        needle = q.lower()
        rows = [s for s in rows if needle in s["name"].lower()]
    if status:
        rows = [s for s in rows if s["status"] == status]

    return {
        "total": len(rows),
        "offset": offset,
        "items": rows[offset : offset + limit],
    }


@router.get("/{project_id}/image/{name}")
def image(project_id: str, name: str):
    project = get_project(project_id)
    sample = project.sample_by_name(name)
    if sample is None:
        raise HTTPException(status_code=404, detail="Изображение не найдено в проекте")
    path = Path(sample["image"])
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"Файл отсутствует: {path}")
    return FileResponse(path)


@router.get("/{project_id}/boxes/{name}")
def boxes(project_id: str, name: str) -> dict:
    project = get_project(project_id)
    sample = project.sample_by_name(name)
    if sample is None:
        raise HTTPException(status_code=404, detail="Изображение не найдено в проекте")

    label = project.label_path(sample)
    gt = read_label(label) if label else []
    ground_truth = [
        {"cls": int(row[0]), "x": float(row[1]), "y": float(row[2]), "w": float(row[3]), "h": float(row[4])}
        for row in gt
    ]

    predictions = proposals_service_boxes(project, name)
    sub = submission_service.load(project)
    if sub is not None and name in sub.by_image:
        prediction = sub.by_image[name]
        for box, score, cls in zip(prediction.boxes, prediction.scores, prediction.classes):
            x1, y1, x2, y2 = (float(v) for v in box)
            predictions.append(
                {
                    "cls": int(cls),
                    "conf": round(float(score), 4),
                    "x": (x1 + x2) / 2,
                    "y": (y1 + y2) / 2,
                    "w": x2 - x1,
                    "h": y2 - y1,
                }
            )

    state = workflow.load(project)
    return {
        "name": name,
        "split": sample.get("split"),
        "edited": bool(label and label.parent == project.annotations_dir),
        "status": state.status(name),
        "note": state.entry(name).get("note", ""),
        "gt": ground_truth,
        "pred": sorted(predictions, key=lambda p: -p["conf"]),
        "classes": project.meta.get("names", []),
    }


def proposals_service_boxes(project: project_service.Project, name: str) -> list[dict]:
    store = proposals_service.load(project)
    return store.for_image(name) if store is not None else []


@router.post("/{project_id}/labels/{name}")
def save_labels(project_id: str, name: str, payload: LabelsIn) -> dict:
    project = get_project(project_id)
    sample = project.sample_by_name(name)
    if sample is None:
        raise HTTPException(status_code=404, detail="Изображение не найдено в проекте")

    boxes = [box.model_dump() for box in payload.boxes]
    target = write_label_file(project, sample, boxes, write_source=payload.write_source)

    with workflow.transaction(project) as state:
        state.set([name], STATUS_DONE, boxes=len(boxes), source="manual")

    reviewed = set(project.meta.get("reviewed", []))
    reviewed.add(name)
    project.meta["reviewed"] = sorted(reviewed)
    project.save()

    for cache in ("dashboard.json", "heatmap.json", "quality.json", "ground_truth.npz"):
        (project.cache_dir / cache).unlink(missing_ok=True)

    return {"ok": True, "boxes": len(boxes), "path": str(target)}


@router.post("/{project_id}/submission/upload")
async def upload_submission(project_id: str, file: UploadFile = File(...)) -> dict:
    project = get_project(project_id)
    project.submissions_dir.mkdir(parents=True, exist_ok=True)
    target = project.submissions_dir / (file.filename or "submission.csv")

    with open(target, "wb") as handle:
        shutil.copyfileobj(file.file, handle)

    try:
        submission = submission_service.register(project, target)
    except ValueError as exc:
        target.unlink(missing_ok=True)
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return {
        "submission": project.meta["submission"],
        "unmatched": submission.unmatched,
        "boxes": submission.boxes,
    }


@router.post("/{project_id}/submission/path")
def register_submission(project_id: str, payload: PathIn) -> dict:
    project = get_project(project_id)
    source = Path(payload.path)
    if not source.is_file():
        raise HTTPException(status_code=404, detail=f"Файл не найден: {source}")

    project.submissions_dir.mkdir(parents=True, exist_ok=True)
    target = project.submissions_dir / source.name
    if source.resolve() != target.resolve():
        shutil.copyfile(source, target)

    try:
        submission = submission_service.register(project, target)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return {
        "submission": project.meta["submission"],
        "unmatched": submission.unmatched,
        "boxes": submission.boxes,
    }


@router.delete("/{project_id}/submission")
def drop_submission(project_id: str) -> dict:
    project = get_project(project_id)
    project.meta["submission"] = None
    project.save()
    submission_service.forget(project)
    for cache in ("dashboard.json", "heatmap.json"):
        (project.cache_dir / cache).unlink(missing_ok=True)
    return {"ok": True}


@router.get("/{project_id}/download/{filename}")
def download(project_id: str, filename: str):
    project = get_project(project_id)
    path = project.exports_dir / Path(filename).name
    if not path.is_file():
        raise HTTPException(status_code=404, detail="Файл не найден")
    media = MEDIA_TYPES.get(path.suffix.lower(), "application/octet-stream")
    return FileResponse(path, filename=path.name, media_type=media)


@router.post("/{project_id}/models/upload")
async def upload_model(project_id: str, file: UploadFile = File(...)) -> dict:
    project = get_project(project_id)
    project.models_dir.mkdir(parents=True, exist_ok=True)
    name = Path(file.filename or "model.pt").name
    if not name.endswith(".pt"):
        raise HTTPException(status_code=400, detail="Ожидается файл *.pt")

    target = project.models_dir / name
    with open(target, "wb") as handle:
        shutil.copyfileobj(file.file, handle)
    return {"model": {"name": name, "path": str(target), "size": target.stat().st_size}}


@router.post("/{project_id}/models/link")
def link_model(project_id: str, payload: PathIn) -> dict:
    project = get_project(project_id)
    source = Path(payload.path)
    if not source.is_file() or source.suffix != ".pt":
        raise HTTPException(status_code=400, detail="Нужен существующий файл *.pt")

    project.models_dir.mkdir(parents=True, exist_ok=True)
    target = project.models_dir / source.name
    if source.resolve() != target.resolve():
        shutil.copyfile(source, target)
    return {"model": {"name": target.name, "path": str(target), "size": target.stat().st_size}}


@router.get("/{project_id}/models")
def list_models(project_id: str) -> dict:
    project = get_project(project_id)
    return {
        "models": [
            {"name": path.name, "path": str(path), "size": path.stat().st_size}
            for path in sorted(project.models_dir.glob("*.pt"))
        ]
    }


@router.delete("/{project_id}/models/{name}")
def delete_model(project_id: str, name: str) -> dict:
    project = get_project(project_id)
    target = project.models_dir / Path(name).name
    target.unlink(missing_ok=True)
    return {"ok": True}


@router.post("/uploads")
async def stage_upload(file: UploadFile = File(...)) -> dict:
    """Generic staging area used by drag & drop before a project exists."""
    ensure_workspace()
    target = UPLOADS_DIR / Path(file.filename or "upload.bin").name
    with open(target, "wb") as handle:
        shutil.copyfileobj(file.file, handle)
    return {"path": str(target), "name": target.name}
