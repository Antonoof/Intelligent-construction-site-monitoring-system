from __future__ import annotations

from fastapi import APIRouter, HTTPException

from ..services.tasks import registry

router = APIRouter(prefix="/api/tasks", tags=["tasks"])


@router.get("")
def list_tasks(kind: str | None = None) -> dict:
    return {"tasks": registry.list(kind)}


@router.get("/{task_id}")
def get_task(task_id: str) -> dict:
    task = registry.get(task_id)
    if task is None:
        raise HTTPException(status_code=404, detail="Задача не найдена")
    return task.as_dict()
