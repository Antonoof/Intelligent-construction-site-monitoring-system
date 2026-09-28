"""Отклонения, проверка инженером, реестр нарушений."""
from __future__ import annotations

import datetime as dt

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.orm import Session

from .. import models as M
from .. import services as S
from ..db import get_session
from ..methodology import get_methodology
from .projects import get_project

router = APIRouter(prefix="/api", tags=["Отклонения и нарушения"])
SEV_ORDER = {"critical": 0, "warning": 1, "info": 2}


class ReviewIn(BaseModel):
    verdict: str = Field(..., description="accepted — подтвердить и зарегистрировать нарушение; "
                                          "rejected — ложное срабатывание; force_majeure — форс-мажор")
    comment: str = ""
    reviewer: str = "инженер"
    contractor: str = ""
    due_days: int = 3


@router.get("/projects/{pid}/deviations", summary="Отклонения с зоной, этапом и снимками-доказательствами")
def list_deviations(pid: int, day: dt.date | None = None, severity: str | None = None, status: str | None = None,
                    zone: str | None = None, include_rejected: bool = False, s: Session = Depends(get_session)):
    p = get_project(s, pid)
    m = get_methodology()
    q = select(M.Deviation).where(M.Deviation.project_id == pid)
    if day:
        q = q.where(M.Deviation.day == day)
    if severity:
        q = q.where(M.Deviation.severity == severity)
    if status:
        q = q.where(M.Deviation.status == status)
    if zone:
        z = S.zone_by_key(s, p, zone)
        q = q.where(M.Deviation.zone_id == (z.id if z else -1))
    devs = [d for d in s.scalars(q) if include_rejected or d.review_status != "rejected"]
    devs.sort(key=lambda d: (d.day, SEV_ORDER[d.severity], d.status != "confirmed", d.id))
    return [S.deviation_view(s, d, m, p) for d in devs]


@router.get("/deviations/{did}")
def get_deviation(did: int, s: Session = Depends(get_session)):
    d = s.get(M.Deviation, did)
    if d is None:
        raise HTTPException(404, "отклонение не найдено")
    out = S.deviation_view(s, d, get_methodology())
    out["reviews"] = [{"verdict": r.verdict, "comment": r.comment, "reviewer": r.reviewer,
                       "at": r.created_at.isoformat()} for r in
                      s.scalars(select(M.DeviationReview).where(M.DeviationReview.deviation_id == did)
                                .order_by(M.DeviationReview.id))]
    return out


@router.post("/deviations/{did}/review", summary="Вердикт инженера: подтверждённое отклонение становится нарушением")
def review(did: int, body: ReviewIn, s: Session = Depends(get_session)):
    d = s.get(M.Deviation, did)
    if d is None:
        raise HTTPException(404, "отклонение не найдено")
    try:
        S.review_deviation(s, d, body.verdict, body.comment, body.reviewer, body.contractor, body.due_days)
    except ValueError as e:
        raise HTTPException(422, str(e))
    s.commit()
    return S.deviation_view(s, d, get_methodology())


@router.get("/projects/{pid}/violations", summary="Реестр нарушений")
def violations(pid: int, s: Session = Depends(get_session)):
    p = get_project(s, pid)
    zones = {z.id: z for z in p.zones}
    out = []
    for v in s.scalars(select(M.Violation).where(M.Violation.project_id == pid).order_by(M.Violation.id)):
        t = s.get(M.ScheduleTask, v.task_id) if v.task_id else None
        z = zones.get(v.zone_id)
        out.append({"id": v.id, "number": v.number, "title": v.title, "description": v.description,
                    "zone": z.name if z else "Вся площадка", "task": t.name if t else None,
                    "contractor": v.contractor, "severity": v.severity, "status": v.status,
                    "registered_at": v.registered_at.isoformat(),
                    "due_date": v.due_date.isoformat() if v.due_date else None, "deviation_id": v.deviation_id})
    return out
