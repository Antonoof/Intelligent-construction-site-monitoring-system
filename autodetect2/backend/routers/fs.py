from __future__ import annotations

import shutil
import string
from pathlib import Path

from fastapi import APIRouter, File, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse

from ..config import IMAGE_SUFFIXES, UPLOADS_DIR, ensure_workspace
from ..schemas import PathIn
from ..services import filesystem

router = APIRouter(prefix="/api/fs", tags=["fs"])


def _drives() -> list[str]:
    return [f"{letter}:\\" for letter in string.ascii_uppercase if Path(f"{letter}:\\").exists()]


@router.get("/roots")
def roots() -> dict:
    entries = [{"name": d, "path": d} for d in _drives()]
    home = Path.home()
    for candidate in (home, home / "Desktop", home / "Downloads", Path.cwd()):
        if candidate.is_dir():
            entries.append({"name": candidate.name or str(candidate), "path": str(candidate)})
    return {"roots": entries}


@router.get("/list")
def list_dir(
    path: str = Query(default=""),
    show_files: bool = False,
    preview: int = Query(default=8, ge=0, le=24),
) -> dict:
    if not path:
        return {"path": "", "parent": None, "dirs": [], "files": [], "roots": _drives()}
    try:
        return filesystem.list_dir(path, show_files=show_files, preview=preview)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except PermissionError as exc:
        raise HTTPException(status_code=403, detail="Нет доступа к папке") from exc


@router.get("/inspect")
def inspect(path: str) -> dict:
    try:
        return filesystem.inspect(path)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.get("/search")
def search(root: str, pattern: str = "*.pt", limit: int = Query(default=200, le=1000)) -> dict:
    try:
        return {"root": root, "pattern": pattern, "matches": filesystem.search(root, pattern, limit)}
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.get("/thumb")
def thumb(path: str):
    """Preview of an image that is not part of a project yet."""
    target = Path(path)
    if target.suffix.lower() not in IMAGE_SUFFIXES or not target.is_file():
        raise HTTPException(status_code=404, detail="Изображение не найдено")
    return FileResponse(target)


@router.post("/yaml")
def read_yaml(payload: PathIn) -> dict:
    try:
        return filesystem.parse_data_yaml(payload.path)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/yaml/upload")
async def upload_yaml(file: UploadFile = File(...)) -> dict:
    ensure_workspace()
    target = UPLOADS_DIR / Path(file.filename or "data.yaml").name
    with open(target, "wb") as handle:
        shutil.copyfileobj(file.file, handle)
    try:
        return filesystem.parse_data_yaml(str(target))
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
