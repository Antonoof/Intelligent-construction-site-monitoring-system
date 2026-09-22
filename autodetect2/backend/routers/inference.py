from __future__ import annotations

import json
from pathlib import Path

from fastapi import APIRouter, HTTPException

from ..schemas import InferBundleIn, TuneIn
from ..services import packaging
from ..services import tuning as tuning_service
from ..services.tasks import registry
from .projects import get_project

router = APIRouter(prefix="/api/projects", tags=["inference"])


def _params_path(project) -> Path:
    return project.cache_dir / "wbf_params.json"


@router.post("/{project_id}/tune")
def tune(project_id: str, payload: TuneIn) -> dict:
    project = get_project(project_id)
    model_paths = [project.models_dir / Path(name).name for name in payload.models]
    missing = [p.name for p in model_paths if not p.is_file()]
    if missing:
        raise HTTPException(status_code=400, detail=f"Нет моделей: {', '.join(missing)}")
    if not model_paths:
        raise HTTPException(status_code=400, detail="Не выбрано ни одной модели")

    def job(ctx):
        ctx.progress("Отбор разнородных изображений", 0.05)
        val_images = tuning_service.diverse_validation_images(project, payload.k, payload.split)
        ctx.log(f"Валидация: {len(val_images)} изображений")

        params = tuning_service.tune(
            project,
            model_paths=model_paths,
            val_images=val_images,
            trials=payload.trials,
            conf_range=tuple(payload.conf_range),
            iou_range=tuple(payload.iou_range),
            skip_range=tuple(payload.skip_range),
            conf_type=payload.conf_type,
            device=payload.device,
            ctx=ctx,
        )
        params["models"] = [p.name for p in model_paths]
        project.cache_dir.mkdir(parents=True, exist_ok=True)
        _params_path(project).write_text(json.dumps(params, indent=2), encoding="utf-8")
        project.meta["wbf"] = params
        project.save()
        return params

    title = f"Optuna WBF · {len(model_paths)} моделей"
    return {"task": registry.submit("tune", title, job).as_dict()}


@router.get("/{project_id}/wbf")
def wbf_params(project_id: str) -> dict:
    project = get_project(project_id)
    path = _params_path(project)
    if not path.exists():
        raise HTTPException(status_code=404, detail="Параметры WBF ещё не найдены")
    return json.loads(path.read_text(encoding="utf-8"))


@router.post("/{project_id}/infer-bundle")
def infer_bundle(project_id: str, payload: InferBundleIn) -> dict:
    project = get_project(project_id)
    path = _params_path(project)
    if not path.exists():
        raise HTTPException(status_code=400, detail="Сначала подберите параметры WBF")

    params = json.loads(path.read_text(encoding="utf-8"))
    model_paths = [project.models_dir / name for name in params.get("models", [])]
    model_paths = [p for p in model_paths if p.is_file()]

    target = packaging.build_infer_bundle(
        project, params, model_paths, include_weights=payload.include_weights
    )
    return {
        "file": target.name,
        "size": target.stat().st_size,
        "url": f"/api/projects/{project.id}/download/{target.name}",
    }
