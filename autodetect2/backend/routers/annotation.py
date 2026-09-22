from __future__ import annotations

import json

from fastapi import APIRouter, HTTPException, Query

from pathlib import Path

from ..config import STATUS_NEW
from ..schemas import ApplyProposalsIn, FlagIn, LabelsBatchIn, PrelabelIn, SelectionIn
from ..services import embeddings as embeddings_service
from ..services import heatmap as heatmap_service
from ..services import proposals as proposals_service
from ..services import selection as selection_service
from ..services import workflow
from ..services.tasks import registry
from .projects import get_project, write_label_file

router = APIRouter(prefix="/api/projects", tags=["annotation"])


@router.post("/{project_id}/embeddings")
def rebuild_embeddings(project_id: str, force: bool = False) -> dict:
    project = get_project(project_id)

    def job(ctx):
        emb = embeddings_service.compute(project, progress=ctx.progress, force=force)
        return {"count": len(emb.names), "backend": emb.backend, "dim": int(emb.z.shape[1])}

    return {"task": registry.submit("embeddings", "Расчёт эмбеддингов", job).as_dict()}


@router.post("/{project_id}/selection")
def create_selection(project_id: str, payload: SelectionIn) -> dict:
    project = get_project(project_id)

    def job(ctx):
        selection = selection_service.build_selection(
            project,
            budget=payload.budget,
            n_clusters=payload.clusters,
            alpha=payload.alpha,
            weights=payload.weights,
            split=payload.split,
            scope=payload.scope,
            include_assigned=payload.include_assigned,
            progress=ctx.progress,
        )
        filename = selection_service.save_selection(project, selection)
        return {**selection.as_dict(), "file": filename}

    title = f"Отбор {payload.budget} изображений"
    return {"task": registry.submit("selection", title, job).as_dict()}


@router.get("/{project_id}/selections")
def selections(project_id: str) -> dict:
    project = get_project(project_id)
    return {"selections": selection_service.list_selections(project)}


@router.get("/{project_id}/selections/{filename}")
def selection_detail(project_id: str, filename: str) -> dict:
    project = get_project(project_id)
    path = project.selections_dir / filename
    if not path.is_file():
        raise HTTPException(status_code=404, detail="Выборка не найдена")
    return json.loads(path.read_text(encoding="utf-8"))


@router.post("/{project_id}/heatmap")
def build_heatmap(project_id: str, clusters: int = Query(default=24, ge=2, le=200), force: bool = False) -> dict:
    project = get_project(project_id)

    def job(ctx):
        ctx.progress("Подготовка признаков", 0.2)
        payload = heatmap_service.build(project, n_clusters=clusters, force=force)
        ctx.progress("Карта построена", 1.0)
        return payload

    return {"task": registry.submit("heatmap", "Построение карты", job).as_dict()}


@router.get("/{project_id}/heatmap")
def cached_heatmap(project_id: str, clusters: int = Query(default=24, ge=2, le=200)) -> dict:
    project = get_project(project_id)
    cache = project.cache_dir / "heatmap.json"
    if not cache.exists():
        raise HTTPException(status_code=404, detail="Карта ещё не построена")
    payload = json.loads(cache.read_text(encoding="utf-8"))
    if payload.get("clusters_k") != clusters:
        raise HTTPException(status_code=409, detail="Карта построена с другим числом кластеров")
    return payload


# ------------------------------------------------------- ускорители разметки
@router.post("/{project_id}/labels-batch")
def save_labels_batch(project_id: str, payload: LabelsBatchIn) -> dict:
    """Одинаковая разметка на пачку кадров.

    Нужно ровно для двух вещей, которые в ручном режиме съедают часы: пометить
    сотню пустых кадров как проверенные и размножить боксы на соседние кадры той
    же съёмки, где техника не сдвинулась.
    """
    project = get_project(project_id)
    known = project.by_name()
    boxes = [box.model_dump() for box in payload.boxes]

    written = []
    for name in payload.names:
        sample = known.get(name)
        if sample is None:
            continue
        write_label_file(project, sample, boxes, write_source=payload.write_source)
        written.append(name)

    if payload.mark_done and written:
        with workflow.transaction(project) as state:
            state.set(written, "done", source="batch")

    for cache in ("dashboard.json", "heatmap.json", "quality.json", "ground_truth.npz"):
        (project.cache_dir / cache).unlink(missing_ok=True)

    return {"images": len(written), "boxes": len(boxes)}


@router.post("/{project_id}/flag/{name}")
def flag_image(project_id: str, name: str, payload: FlagIn) -> dict:
    """Пометка кадра: спорный, вернуть в работу, закрыть."""
    project = get_project(project_id)
    if project.sample_by_name(name) is None:
        raise HTTPException(status_code=404, detail="Изображение не найдено в проекте")

    with workflow.transaction(project) as state:
        if payload.status == STATUS_NEW:
            state.release([name])
        elif payload.status == "review":
            state.flag(name, payload.note)
        else:
            state.set([name], payload.status, note=payload.note)
    return {"ok": True, "status": payload.status}


@router.get("/{project_id}/neighbors/{name}")
def neighbors(project_id: str, name: str, k: int = Query(default=12, ge=1, le=200)) -> dict:
    """Ближайшие кадры в пространстве признаков.

    На стройплощадке камера стоит месяцами: соседний по признакам кадр — это,
    как правило, та же сцена десять минут спустя, и разметка с неё переносится
    почти без правок.
    """
    project = get_project(project_id)
    emb = embeddings_service.compute(project)
    index = emb.index_of()
    position = index.get(name)
    if position is None:
        raise HTTPException(status_code=404, detail="Для этого кадра нет эмбеддинга")

    import numpy as np

    distances = np.linalg.norm(emb.z - emb.z[position], axis=1)
    order = np.argsort(distances)[: k + 1]
    samples = project.by_name()
    state = workflow.load(project)

    out = []
    for i in order:
        if int(i) == position:
            continue
        neighbour = emb.names[int(i)]
        sample = samples.get(neighbour, {})
        out.append(
            {
                "name": neighbour,
                "distance": round(float(distances[i]), 4),
                "boxes": sample.get("boxes", 0),
                "split": sample.get("split"),
                "status": state.status(neighbour),
            }
        )
    return {"name": name, "neighbors": out}


# ------------------------------------------------------------ предразметка
@router.post("/{project_id}/prelabel")
def prelabel(project_id: str, payload: PrelabelIn) -> dict:
    project = get_project(project_id)
    model_path = project.models_dir / Path(payload.model).name
    if not model_path.is_file():
        raise HTTPException(status_code=400, detail=f"Модель не найдена: {payload.model}")

    def job(ctx):
        ctx.progress("Отбор кадров", 0.02)
        samples = project.samples()
        names = [s["name"] for s in samples]
        mask = selection_service.eligible(project, names, split=payload.split, scope=payload.scope)
        chosen = [s for s, keep in zip(samples, mask) if keep][: payload.limit]
        if not chosen:
            raise ValueError("Не нашлось кадров под эти условия")
        ctx.log(f"Кадров в работе: {len(chosen)}")

        result = proposals_service.run(
            project,
            model_path=model_path,
            images=chosen,
            conf=payload.conf,
            iou=payload.iou,
            imgsz=payload.imgsz,
            device=payload.device,
            progress=ctx.progress,
        )
        return {
            "model": result.model,
            "images": len(result.names),
            "boxes": int(len(result.boxes)),
            "created_at": result.created_at,
        }

    return {"task": registry.submit("prelabel", f"Предразметка · {model_path.name}", job).as_dict()}


@router.get("/{project_id}/proposals")
def proposals_info(project_id: str) -> dict:
    project = get_project(project_id)
    store = proposals_service.load(project)
    if store is None:
        raise HTTPException(status_code=404, detail="Предразметка ещё не выполнялась")
    return {
        "model": store.model,
        "created_at": store.created_at,
        "images": len(store.names),
        "boxes": int(len(store.boxes)),
    }


@router.post("/{project_id}/proposals/apply")
def apply_proposals(project_id: str, payload: ApplyProposalsIn) -> dict:
    project = get_project(project_id)
    return proposals_service.apply(
        project,
        names=payload.names,
        conf=payload.conf,
        only_unlabeled=payload.only_unlabeled,
        mark_done=payload.mark_done,
    )


@router.delete("/{project_id}/proposals")
def drop_proposals(project_id: str) -> dict:
    project = get_project(project_id)
    proposals_service.forget(project)
    return {"ok": True}
