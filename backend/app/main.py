"""ОКО — сервис поиска нарушений на строительных площадках (FastAPI).

Запуск на ноутбуке:  cd backend && uvicorn app.main:app --port 8000   →  http://localhost:8000
Документация API (Swagger): http://localhost:8000/docs
"""
from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles
from sqlalchemy import func, select

from . import models as M
from .ai import jobs as ai_jobs
from .api import ai, analyze, deviations, projects, reference, snapshots
from .config import settings
from .db import SessionLocal, init_db, session_scope
from .detection import get_detector
from .methodology import get_methodology, sync_to_db

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("oko")

def startup() -> None:
    """Схема БД, справочники методики, детектор, демо-проект при первом запуске."""
    init_db()
    m = get_methodology()
    with session_scope() as s:
        sync_to_db(s, m)
    det = get_detector()
    log.info("методика %s, правила %s, детектор %s (%d классов)", m.version, m.rules_version, det.name,
             len(det.classes))
    with session_scope() as s:
        if ai_jobs.recover(s):
            log.info("незавершённые ИИ-анализы помечены как прерванные")
    ai_st = _ai_brief()
    log.info("ИИ-анализ: VLM %s, LLM %s", ai_st["vlm"] or "выключена", ai_st["llm"] or "выключена")
    if settings.seed_demo:
        with session_scope() as s:
            if not s.scalar(select(func.count(M.Project.id))):
                from .demo import seed_demo
                seed_demo(s)


@asynccontextmanager
async def lifespan(_app: FastAPI):
    startup()
    yield


app = FastAPI(
    lifespan=lifespan,
    title="ОКО — поиск нарушений на стройплощадках",
    version="2.0.0",
    description="Снимки с камер → техника (RF-DETR) → этапы календарного графика (методика «этап → техника») "
                "→ отклонения с зоной и снимками-доказательствами → проверка инженером → нарушения.",
)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
for r in (reference.router, projects.router, snapshots.router, deviations.router, analyze.router, ai.router):
    app.include_router(r)


@app.get("/api/health", tags=["Служебное"])
def health():
    m = get_methodology()
    det = get_detector()
    with SessionLocal() as s:
        projects_n = s.scalar(select(func.count(M.Project.id)))
    return {"status": "ok", "methodology_version": m.version, "rules_version": m.rules_version,
            "detector": det.name, "detector_classes": sorted(det.classes),
            "database": settings.db_url().split(":")[0], "projects": projects_n, "ai": _ai_brief()}


def _ai_brief() -> dict:
    """Включённые ИИ-слои: VLM (какая модель будет загружена под доступную память) и LLM."""
    st = ai_jobs.status()
    return {"vlm": st["vlm"].get("model") if st["vlm"].get("enabled") else None,
            "llm": f'{st["llm"]["provider"]}:{st["llm"]["model"]}' if st["llm"]["provider"] != "off" else None}


if settings.frontend_dir.exists():
    app.mount("/app", StaticFiles(directory=settings.frontend_dir, html=True), name="frontend")

    @app.get("/", include_in_schema=False)
    def root():
        return RedirectResponse("/app/")
