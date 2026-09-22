from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

from ..assets import asset
from ..config import APP_NAME, APP_VERSION, CLASS_NAMES, MODEL_FAMILIES, MODEL_SIZES, TEMPLATES_DIR
from ..services import project as project_service

router = APIRouter(tags=["pages"])
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))


templates.env.globals.update(app_name=APP_NAME, app_version=APP_VERSION, asset=asset)

SECTIONS = {
    "annotate": "Разметка",
    "quality": "Проверка разметки",
    "dataset": "Датасет",
    "train": "Обучение",
    "inference": "Инференс",
    "stats": "Статистика",
    "heatmap": "Карта данных",
}


def render(request: Request, template: str, **context) -> HTMLResponse:
    return templates.TemplateResponse(request=request, name=template, context=context)


def project_context(project_id: str) -> dict:
    try:
        project = project_service.load_project(project_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return {"project": project.summary()}


@router.get("/", response_class=HTMLResponse)
def index(request: Request) -> HTMLResponse:
    return render(request, "index.html", projects=project_service.list_projects())


@router.get("/create", response_class=HTMLResponse)
def create(request: Request) -> HTMLResponse:
    return render(request, "create.html", default_classes=CLASS_NAMES)


@router.get("/p/{project_id}", response_class=HTMLResponse)
def hub(request: Request, project_id: str) -> HTMLResponse:
    return render(request, "project.html", sections=SECTIONS, **project_context(project_id))


@router.get("/p/{project_id}/annotate", response_class=HTMLResponse)
def annotate(request: Request, project_id: str) -> HTMLResponse:
    return render(request, "annotate.html", **project_context(project_id))


@router.get("/p/{project_id}/quality", response_class=HTMLResponse)
def quality(request: Request, project_id: str) -> HTMLResponse:
    return render(request, "quality.html", **project_context(project_id))


@router.get("/p/{project_id}/dataset", response_class=HTMLResponse)
def dataset(request: Request, project_id: str) -> HTMLResponse:
    return render(request, "dataset.html", **project_context(project_id))


@router.get("/p/{project_id}/heatmap", response_class=HTMLResponse)
def heatmap(request: Request, project_id: str) -> HTMLResponse:
    return render(request, "heatmap.html", **project_context(project_id))


@router.get("/p/{project_id}/train", response_class=HTMLResponse)
def train(request: Request, project_id: str) -> HTMLResponse:
    return render(
        request,
        "train.html",
        families=MODEL_FAMILIES,
        sizes=MODEL_SIZES,
        **project_context(project_id),
    )


@router.get("/p/{project_id}/inference", response_class=HTMLResponse)
def inference(request: Request, project_id: str) -> HTMLResponse:
    return render(request, "inference.html", **project_context(project_id))


@router.get("/p/{project_id}/stats", response_class=HTMLResponse)
def stats(request: Request, project_id: str) -> HTMLResponse:
    return render(request, "stats.html", **project_context(project_id))
