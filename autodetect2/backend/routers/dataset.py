from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

from ..config import STATUS_ASSIGNED, STATUS_DONE, STATUS_REVIEW
from ..schemas import (
    AssignmentExportIn,
    AssignmentIn,
    CandidatesIn,
    ExportDatasetIn,
    ImportIn,
    SplitIn,
)
from ..services import assignments as assignment_service
from ..services import splitting
from ..services import workflow
from ..services.tasks import registry
from .projects import get_project

router = APIRouter(prefix="/api/projects", tags=["dataset"])


# ----------------------------------------------------------------- статусы
@router.get("/{project_id}/status")
def status(project_id: str) -> dict:
    project = get_project(project_id)
    state = workflow.load(project)
    counts = state.counts()
    samples = project.samples()
    labeled = sum(1 for s in samples if s.get("boxes") or s.get("label"))

    return {
        "counts": counts,
        "labeled": labeled,
        "unlabeled": len(samples) - labeled,
        "pending": [
            {
                "name": name,
                "owner": state.owner(name),
                "assignment": state.assignment(name),
                "at": state.entry(name).get("at"),
            }
            for name in state.names_with(STATUS_ASSIGNED)[:200]
        ],
        "review": [
            {"name": name, "note": state.entry(name).get("note", "")}
            for name in state.names_with(STATUS_REVIEW)[:200]
        ],
        "done": counts.get(STATUS_DONE, 0),
    }


# ------------------------------------------------------------- train / val
@router.post("/{project_id}/split/plan")
def plan_split(project_id: str, payload: SplitIn) -> dict:
    project = get_project(project_id)
    plan = splitting.plan_split(
        project,
        val_ratio=payload.val_ratio,
        val_count=payload.val_count,
        scope=payload.scope,
        group_by=payload.group_by,
        stratify=payload.stratify,
        seed=payload.seed,
    )
    return {"plan": plan.as_dict(), "modes": splitting.GROUP_MODES}


@router.post("/{project_id}/split/apply")
def apply_split(project_id: str, payload: SplitIn) -> dict:
    project = get_project(project_id)
    plan = splitting.plan_split(
        project,
        val_ratio=payload.val_ratio,
        val_count=payload.val_count,
        scope=payload.scope,
        group_by=payload.group_by,
        stratify=payload.stratify,
        seed=payload.seed,
    )
    applied = splitting.apply_split(project, plan)
    return {"split": applied, "plan": plan.as_dict()}


@router.post("/{project_id}/export/dataset")
def export_dataset(project_id: str, payload: ExportDatasetIn) -> dict:
    project = get_project(project_id)

    def job(ctx):
        return splitting.export_dataset(
            project,
            link_mode=payload.link_mode,
            include_unlabeled=payload.include_unlabeled,
            progress=ctx.progress,
        )

    return {"task": registry.submit("export", "Выгрузка датасета", job).as_dict()}


# ---------------------------------------------------------------- задания
@router.post("/{project_id}/candidates")
def candidates(project_id: str, payload: CandidatesIn) -> dict:
    project = get_project(project_id)
    names = assignment_service.candidates(
        project, count=payload.count, source=payload.source, split=payload.split
    )
    return {"names": names, "count": len(names)}


@router.get("/{project_id}/assignments")
def list_assignments(project_id: str) -> dict:
    project = get_project(project_id)
    return {"assignments": assignment_service.listing(project)}


@router.post("/{project_id}/assignments")
def create_assignment(project_id: str, payload: AssignmentIn) -> dict:
    project = get_project(project_id)
    images = payload.images
    if not images:
        images = assignment_service.candidates(
            project, count=payload.count, source=payload.source, split=payload.split
        )
    record = assignment_service.create(
        project, images=images, title=payload.title, owner=payload.owner, note=payload.note
    )
    return {"assignment": {k: v for k, v in record.items() if k != "images"}}


@router.get("/{project_id}/assignments/{assignment_id}")
def assignment_detail(project_id: str, assignment_id: str) -> dict:
    project = get_project(project_id)
    try:
        return {"assignment": assignment_service.load(project, assignment_id)}
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.delete("/{project_id}/assignments/{assignment_id}")
def cancel_assignment(project_id: str, assignment_id: str) -> dict:
    project = get_project(project_id)
    try:
        return {"assignment": assignment_service.cancel(project, assignment_id)}
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.post("/{project_id}/assignments/{assignment_id}/export")
def export_assignment(project_id: str, assignment_id: str, payload: AssignmentExportIn) -> dict:
    project = get_project(project_id)

    def job(ctx):
        ctx.progress("Сборка задания", 0.2)
        target = assignment_service.export(project, assignment_id, mode=payload.mode)
        ctx.progress("Готово", 1.0)
        return {
            "file": target.name,
            "size": target.stat().st_size,
            "url": f"/api/projects/{project.id}/download/{target.name}",
        }

    label = "архива" if payload.mode == "pack" else "манифеста"
    return {"task": registry.submit("assignment", f"Сборка {label} задания", job).as_dict()}


@router.post("/{project_id}/assignments/{assignment_id}/result")
def export_assignment_result(project_id: str, assignment_id: str) -> dict:
    project = get_project(project_id)
    try:
        target = assignment_service.export_result(project, assignment_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return {
        "file": target.name,
        "size": target.stat().st_size,
        "url": f"/api/projects/{project.id}/download/{target.name}",
    }


@router.post("/{project_id}/assignments/accept")
def accept_assignment(project_id: str, payload: ImportIn) -> dict:
    project = get_project(project_id)
    source = Path(payload.path)
    if not source.exists():
        raise HTTPException(status_code=404, detail=f"Файл не найден: {source}")
    return assignment_service.accept(project, source)


@router.post("/{project_id}/assignments/import")
def import_assignment_result(project_id: str, payload: ImportIn) -> dict:
    project = get_project(project_id)
    source = Path(payload.path)
    if not source.exists():
        raise HTTPException(status_code=404, detail=f"Файл не найден: {source}")
    return assignment_service.import_result(project, source, assignment_id=payload.assignment)


@router.get("/{project_id}/exports/{filename}")
def download_export(project_id: str, filename: str):
    project = get_project(project_id)
    path = project.exports_dir / Path(filename).name
    if not path.is_file():
        raise HTTPException(status_code=404, detail="Файл не найден")
    return FileResponse(path, filename=path.name)
