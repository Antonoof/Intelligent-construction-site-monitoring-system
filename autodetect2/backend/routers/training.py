from __future__ import annotations

from fastapi import APIRouter

from ..config import MODEL_FAMILIES, MODEL_SIZES
from ..schemas import BundleIn, TrainConfigIn
from ..services import packaging
from ..services import project as project_service
from .projects import get_project

router = APIRouter(prefix="/api/projects", tags=["training"])


@router.get("/{project_id}/train-config")
def train_config(project_id: str) -> dict:
    project = get_project(project_id)
    return {
        "config": project_service.read_train_yaml(project),
        "families": MODEL_FAMILIES,
        "sizes": list(MODEL_SIZES),
        "yaml_path": str(project.train_yaml),
        "data_yaml": project.data_yaml.read_text(encoding="utf-8"),
    }


@router.put("/{project_id}/train-config")
def update_train_config(project_id: str, payload: TrainConfigIn) -> dict:
    project = get_project(project_id)
    config = project_service.write_train_yaml(project, payload.config)
    return {"config": config, "yaml_path": str(project.train_yaml)}


@router.post("/{project_id}/train-bundle")
def build_bundle(project_id: str, payload: BundleIn) -> dict:
    project = get_project(project_id)
    config = payload.config or project_service.read_train_yaml(project)
    if payload.config:
        project_service.write_train_yaml(project, payload.config)

    target = packaging.build_train_bundle(project, config)
    return {
        "file": target.name,
        "size": target.stat().st_size,
        "url": f"/api/projects/{project.id}/download/{target.name}",
    }
