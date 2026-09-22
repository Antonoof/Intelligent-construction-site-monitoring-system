from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query

from ..schemas import QualityIn
from ..services import quality as quality_service
from ..services.tasks import registry
from .projects import get_project

router = APIRouter(prefix="/api/projects", tags=["quality"])


@router.get("/{project_id}/quality")
def cached_quality(project_id: str, limit: int = Query(default=400, ge=1, le=4000)) -> dict:
    project = get_project(project_id)
    payload = quality_service.cached(project)
    if payload is None:
        raise HTTPException(status_code=404, detail="Проверка разметки ещё не запускалась")
    return {**payload, "images": payload["images"][:limit], "total_images": len(payload["images"])}


@router.post("/{project_id}/quality")
def build_quality(project_id: str, payload: QualityIn) -> dict:
    project = get_project(project_id)

    def job(ctx):
        report = quality_service.audit(
            project, check_classes=payload.check_classes, progress=ctx.progress
        )
        # Список кадров бывает на тысячи строк — в результат задачи кладётся
        # только сводка, полный отчёт страница забирает отдельным запросом.
        return {k: v for k, v in report.items() if k not in ("images", "unlabeled_sample")}

    title = "Проверка разметки"
    return {"task": registry.submit("quality", title, job).as_dict()}


@router.get("/{project_id}/quality/kinds")
def kinds() -> dict:
    return {
        "kinds": [
            {"kind": kind, "label": label, "severity": severity}
            for kind, (label, severity) in quality_service.ISSUE_KINDS.items()
        ]
    }
