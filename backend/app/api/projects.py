"""Проекты, календарный график, камеры и зоны, сводка на день, отчёт."""
from __future__ import annotations

import csv
import datetime as dt
import io

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.orm import Session

from .. import models as M
from .. import services as S
from ..db import get_session
from ..methodology import get_methodology

router = APIRouter(prefix="/api", tags=["Проекты и график"])


def get_project(s: Session, pid: int) -> M.Project:
    p = s.get(M.Project, pid)
    if p is None:
        raise HTTPException(404, f"проект {pid} не найден")
    return p


class ProjectIn(BaseModel):
    name: str = Field(..., examples=["Жилой дом, корпус 1"])
    object_type: str = Field("housing", description="тип объекта из справочника: housing, education, roads …")
    address: str = ""
    description: str = ""


class SiteConfigIn(BaseModel):
    zones: list[dict] = []
    cameras: list[dict] = []
    site_equipment: list[dict] = []


class TaskPatch(BaseModel):
    work_type_id: str | None = Field(None, description="вид работ справочника, «» — снять привязку")
    zone_key: str | None = None


def project_view(s: Session, p: M.Project) -> dict:
    zkeys = {z.id: z.key for z in p.zones}
    ver = S.current_schedule_version(s, p.id)
    return {"id": p.id, "name": p.name, "object_type": p.object_type, "address": p.address,
            "description": p.description, "is_demo": p.is_demo,
            "zones": [{"key": z.key, "name": z.name, "color": z.color} for z in p.zones],
            "cameras": [{"key": c.key, "name": c.name, "overview": c.is_overview, "tiles": c.tiles,
                         "default_zone": zkeys.get(c.default_zone_id), "note": c.note,
                         "zones": {zkeys[cz.zone_id]: cz.polygon for cz in c.zone_polygons}} for c in p.cameras],
            "site_equipment": [{"class": se.equipment_class, "name": se.name, "zones": se.zone_keys}
                               for se in s.scalars(select(M.SiteEquipment).where(M.SiteEquipment.project_id == p.id))],
            "schedule": {"version": ver.version, "file": ver.source_filename, "rows": ver.rows_total,
                         "matched": ver.rows_matched} if ver else None,
            "days": S.project_days(s, p.id)}


@router.get("/projects")
def list_projects(s: Session = Depends(get_session)):
    return [{"id": p.id, "name": p.name, "object_type": p.object_type, "is_demo": p.is_demo}
            for p in s.scalars(select(M.Project).order_by(M.Project.id))]


@router.post("/projects", status_code=201)
def create_project(body: ProjectIn, s: Session = Depends(get_session)):
    p = S.create_project(s, body.name, body.object_type, body.address, body.description)
    s.commit()
    return project_view(s, p)


@router.get("/projects/{pid}")
def get_project_view(pid: int, s: Session = Depends(get_session)):
    return project_view(s, get_project(s, pid))


@router.put("/projects/{pid}/site", summary="Зоны, камеры с полигонами зон, заявленная техника")
def put_site(pid: int, body: SiteConfigIn, s: Session = Depends(get_session)):
    p = get_project(s, pid)
    S.apply_site_config(s, p, body.model_dump())
    S.recompute_project(s, p)
    s.commit()
    return project_view(s, p)


@router.post("/projects/{pid}/schedule", summary="Загрузить календарный график (XLSX/CSV)")
async def upload_schedule(pid: int, file: UploadFile = File(...), s: Session = Depends(get_session)):
    p = get_project(s, pid)
    data = await file.read()
    try:
        res = S.import_schedule(s, p, file.filename or "schedule.xlsx", data)
    except ValueError as e:
        raise HTTPException(422, str(e))
    S.recompute_project(s, p)
    s.commit()
    return res


@router.get("/projects/{pid}/schedule", summary="Этапы графика с привязкой к справочнику и технике")
def get_schedule(pid: int, day: dt.date | None = None, s: Session = Depends(get_session)):
    p = get_project(s, pid)
    m = get_methodology()
    zones = {z.id: z for z in p.zones}
    out = []
    for t in S.schedule_tasks(s, pid):
        wt = m.work_types.get(t.work_type_id) if t.work_type_id else None
        prof = m.profiles.get(t.profile_id) if t.profile_id else None
        z = zones.get(t.zone_id)
        out.append({
            "id": t.id, "row": t.row_no, "wbs": t.wbs, "name": t.name, "parent_id": t.parent_task_id,
            "summary": t.is_summary, "start": t.start_date.isoformat(), "end": t.end_date.isoformat(),
            "zone": {"key": z.key, "name": z.name, "color": z.color} if z else None,
            "work_type": {"id": wt.id, "name": wt.name, "stage": wt.stage, "observability": wt.observability,
                          "expert_note": wt.expert_note} if wt else None,
            "match": {"method": t.match_method, "score": t.match_score},
            "profile": {"id": prof.id, "name": prof.name,
                        "required": [{"role": g.role, "any_of": list(g.any_of), "min": g.min, "window": g.window}
                                     for g in prof.required],
                        "companions": [{"lead": list(c.lead), "partner": list(c.partner), "message": c.message}
                                       for c in prof.companions],
                        "allowed": sorted(prof.allowed), "not_detected": sorted(prof.not_detected)} if prof else None,
            "planned": t.planned_equipment, "contractor": t.contractor,
            "active": bool(day and t.start_date <= day <= t.end_date)})
    return out


@router.patch("/tasks/{tid}", summary="Вручную привязать этап к виду работ или зоне")
def patch_task(tid: int, body: TaskPatch, s: Session = Depends(get_session)):
    t = s.get(M.ScheduleTask, tid)
    if t is None:
        raise HTTPException(404, "этап не найден")
    try:
        S.set_task_work_type(s, t, body.work_type_id, body.zone_key)
    except S.NotFound as e:
        raise HTTPException(404, str(e))
    S.recompute_project(s, s.get(M.Project, t.project_id))
    s.commit()
    return {"ok": True}


@router.get("/projects/{pid}/summary", summary="Сводка площадки на день: зоны, KPI, отклонения")
def summary(pid: int, day: dt.date | None = None, s: Session = Depends(get_session)):
    p = get_project(s, pid)
    if day is None:
        days = S.project_days(s, pid)
        day = dt.date.fromisoformat(days[-1]["day"]) if days else dt.date.today()
    return S.project_summary(s, p, day)


@router.post("/projects/{pid}/recompute", summary="Пересчитать все выводы проекта")
def recompute(pid: int, s: Session = Depends(get_session)):
    p = get_project(s, pid)
    n = S.recompute_project(s, p)
    s.commit()
    return {"days": n}


@router.get("/projects/{pid}/report", summary="Отчёт об отклонениях (JSON или CSV)")
def report(pid: int, day: dt.date | None = None, format: str = "json", s: Session = Depends(get_session)):
    p = get_project(s, pid)
    m = get_methodology()
    q = select(M.Deviation).where(M.Deviation.project_id == pid)
    if day:
        q = q.where(M.Deviation.day == day)
    devs = [S.deviation_view(s, d, m, p) for d in s.scalars(q.order_by(M.Deviation.day, M.Deviation.id))]
    if format != "csv":
        return {"project": p.name, "day": day.isoformat() if day else None, "deviations": devs}
    buf = io.StringIO()
    w = csv.writer(buf, delimiter=";")
    w.writerow(["id", "дата", "зона", "этап", "правило", "отклонение", "важность", "статус", "проверка",
                "техника", "снимков", "первый снимок", "последний снимок", "описание", "рекомендация", "снимки"])
    for d in devs:
        w.writerow([d["id"], d["day"], d["zone"]["name"], (d["task"] or {}).get("name", ""), d["rule"], d["title"],
                    d["severity"], d["status"], d["review_status"] or "", d["equipment_name"] or "",
                    d["snapshots_count"], d["first_seen"], d["last_seen"], d["message"], d["recommendation"],
                    " ".join(str(e["snapshot_id"]) for e in d["evidence"])])
    data = buf.getvalue().encode("utf-8-sig")
    name = f"oko_report_{pid}_{day or 'all'}.csv"
    return StreamingResponse(io.BytesIO(data), media_type="text/csv",
                             headers={"Content-Disposition": f'attachment; filename="{name}"'})
