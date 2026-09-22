from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query

from ..services import analysis as analysis_service
from ..services import selection as selection_service
from ..services import submission as submission_service
from ..services.tasks import registry
from .projects import get_project

router = APIRouter(prefix="/api/projects", tags=["stats"])


@router.get("/{project_id}/stats")
def cached_stats(project_id: str) -> dict:
    project = get_project(project_id)
    payload = analysis_service.cached_dashboard(project)
    if payload is None:
        raise HTTPException(status_code=404, detail="Статистика ещё не рассчитана")
    return payload


@router.post("/{project_id}/stats")
def build_stats(
    project_id: str,
    conf: float = Query(default=0.25, ge=0.0, le=1.0),
    iou: float = Query(default=0.5, ge=0.05, le=0.95),
    clusters: int = Query(default=12, ge=2, le=200),
) -> dict:
    project = get_project(project_id)
    if project.submission_path is None:
        raise HTTPException(status_code=400, detail="Сначала загрузите submission.csv")

    def job(ctx):
        ctx.progress("Чтение submission.csv", 0.15)
        submission = submission_service.load(project)
        if submission is None:
            raise ValueError("submission.csv не найден")

        ctx.progress("Кластеризация", 0.35)
        cluster_map = None
        try:
            names, labels = selection_service.assign_clusters(project, clusters)
            cluster_map = {name: int(label) for name, label in zip(names, labels)}
        except Exception as exc:  # noqa: BLE001 - clusters are optional context
            ctx.log(f"Кластеры недоступны: {exc}")

        payload = analysis_service.build_dashboard(
            project,
            submission,
            conf_eval=conf,
            iou_thr=iou,
            clusters=cluster_map,
            progress=lambda message, value=None: ctx.progress(message, value),
        )
        payload["submission"] = project.meta.get("submission")
        analysis_service.store_dashboard(project, payload)
        ctx.progress("Готово", 1.0)
        return payload

    return {"task": registry.submit("stats", "Расчёт статистики", job).as_dict()}
