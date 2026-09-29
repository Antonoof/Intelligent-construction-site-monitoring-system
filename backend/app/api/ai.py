"""ИИ-анализ: локальная VLM и LLM (Claude / ChatGPT) поверх детектора, модели готовности и правил."""
from __future__ import annotations

import datetime as dt

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.orm import Session

from .. import models as M
from ..ai import jobs
from ..ai.fusion import apply_corrections, revert_corrections
from ..db import get_session
from .projects import get_project

router = APIRouter(prefix="/api", tags=["ИИ-анализ"])


def review_view(r: M.AIReview, full: bool = False) -> dict:
    out = {"id": r.id, "kind": r.kind, "status": r.status, "step": r.step, "project_id": r.project_id,
           "snapshot_id": r.snapshot_id, "day": r.day.isoformat() if r.day else None,
           "vlm": {"model": r.vlm_model, "output": r.vlm_output} if r.vlm_model or r.vlm_output else None,
           "llm": {"provider": r.llm_provider, "model": r.llm_model, "tokens_in": r.tokens_in,
                   "tokens_out": r.tokens_out} if r.llm_provider else None,
           "final": r.final, "applied": r.applied, "error": r.error, "duration_ms": r.duration_ms,
           "created_at": r.created_at.isoformat(timespec="seconds") if r.created_at else None,
           "updated_at": r.updated_at.isoformat(timespec="seconds") if r.updated_at else None}
    if full:
        out["context"] = r.context
        out["llm"] = {**(out["llm"] or {}), "output": r.llm_output}
    return out


def _review(s: Session, rid: int) -> M.AIReview:
    r = s.get(M.AIReview, rid)
    if r is None:
        raise HTTPException(404, "анализ не найден")
    return r


@router.get("/ai/status", summary="Какие ИИ-слои включены: VLM (модель под доступную память) и LLM")
def ai_status():
    return jobs.status()


@router.post("/snapshots/{sid}/ai-review", status_code=202,
             summary="Запустить ИИ-анализ снимка: VLM описывает кадр, LLM проверяет все слои")
def start_snapshot_review(sid: int, s: Session = Depends(get_session)):
    snap = s.get(M.Snapshot, sid)
    if snap is None:
        raise HTTPException(404, "снимок не найден")
    try:
        r = jobs.submit_snapshot(s, snap)
    except jobs.AIDisabled as e:
        raise HTTPException(409, str(e))
    s.refresh(r)
    return review_view(r)


@router.get("/snapshots/{sid}/ai-review", summary="Последний ИИ-анализ снимка (null, если не запускался)")
def last_snapshot_review(sid: int, s: Session = Depends(get_session)):
    r = s.scalar(select(M.AIReview).where(M.AIReview.snapshot_id == sid, M.AIReview.kind == "snapshot")
                 .order_by(M.AIReview.id.desc()))
    return review_view(r) if r else None


@router.get("/ai/reviews/{rid}", summary="ИИ-анализ; full=1 — с контекстом, отправленным в LLM, и её ответом")
def get_review(rid: int, full: int = 0, s: Session = Depends(get_session)):
    return review_view(_review(s, rid), bool(full))


@router.post("/ai/reviews/{rid}/apply", summary="Применить исправления LLM и пересчитать отклонения")
def apply_review(rid: int, s: Session = Depends(get_session)):
    r = _review(s, rid)
    try:
        apply_corrections(s, r)
    except ValueError as e:
        raise HTTPException(409, str(e))
    s.commit()
    return review_view(r)


@router.post("/ai/reviews/{rid}/revert", summary="Отменить применённые исправления")
def revert_review(rid: int, s: Session = Depends(get_session)):
    r = _review(s, rid)
    try:
        revert_corrections(s, r)
    except ValueError as e:
        raise HTTPException(409, str(e))
    s.commit()
    return review_view(r)


@router.post("/projects/{pid}/ai-summary", status_code=202, summary="Запустить ИИ-анализ площадки за день")
def start_day_review(pid: int, day: dt.date, s: Session = Depends(get_session)):
    p = get_project(s, pid)
    try:
        r = jobs.submit_day(s, p, day)
    except jobs.AIDisabled as e:
        raise HTTPException(409, str(e))
    s.refresh(r)
    return review_view(r)


@router.get("/projects/{pid}/ai-summary", summary="Последний ИИ-анализ площадки за день (null, если не было)")
def last_day_review(pid: int, day: dt.date, s: Session = Depends(get_session)):
    r = s.scalar(select(M.AIReview).where(M.AIReview.project_id == pid, M.AIReview.kind == "day",
                                          M.AIReview.day == day).order_by(M.AIReview.id.desc()))
    return review_view(r) if r else None
