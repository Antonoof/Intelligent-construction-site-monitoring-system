from __future__ import annotations

import logging

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from .assets import VersionedStatic
from .config import APP_NAME, APP_VERSION, STATIC_DIR, ensure_workspace
from .routers import annotation, dataset, fs, inference, pages, projects, quality, stats, tasks, training

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s %(name)s  %(message)s",
    datefmt="%H:%M:%S",
)


def create_app() -> FastAPI:
    ensure_workspace()
    app = FastAPI(title=APP_NAME, version=APP_VERSION, docs_url="/api/docs", redoc_url=None)

    app.mount("/static", VersionedStatic(directory=str(STATIC_DIR)), name="static")

    app.include_router(pages.router)
    app.include_router(projects.router)
    app.include_router(annotation.router)
    app.include_router(quality.router)
    app.include_router(dataset.router)
    app.include_router(training.router)
    app.include_router(inference.router)
    app.include_router(stats.router)
    app.include_router(fs.router)
    app.include_router(tasks.router)

    @app.exception_handler(ValueError)
    async def value_error_handler(_: Request, exc: ValueError) -> JSONResponse:
        return JSONResponse(status_code=400, content={"detail": str(exc)})

    @app.get("/api/health")
    def health() -> dict:
        return {"status": "ok", "app": APP_NAME, "version": APP_VERSION}

    return app


app = create_app()
