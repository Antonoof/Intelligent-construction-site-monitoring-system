"""Снимки: загрузка, список, карточка с объяснением сопоставления, изображение с разметкой."""
from __future__ import annotations

import datetime as dt
from pathlib import Path

from fastapi import APIRouter, Depends, File, Form, HTTPException, Response, UploadFile
from sqlalchemy import select
from sqlalchemy.orm import Session

from .. import models as M
from .. import services as S
from ..ai import jobs as ai_jobs
from ..config import settings
from ..db import get_session
from ..imaging import load_image, render_annotated, thumbnail
from ..methodology import get_methodology
from .projects import get_project

router = APIRouter(prefix="/api", tags=["Снимки"])


def _parse_dt(v: str | None) -> dt.datetime | None:
    if not v:
        return None
    try:
        return dt.datetime.fromisoformat(v.replace("Z", "")).replace(tzinfo=None)
    except ValueError:
        raise HTTPException(422, f"не удалось разобрать дату-время «{v}», нужен ISO 8601: 2026-09-24T10:30")


@router.post("/projects/{pid}/snapshots", summary="Загрузить снимки с камеры и сразу проверить их")
async def upload_snapshots(pid: int, camera: str = Form(..., description="ключ камеры, например CAM-01"),
                           taken_at: str | None = Form(None, description="время съёмки, если его нет в EXIF и имени файла"),
                           files: list[UploadFile] = File(...), s: Session = Depends(get_session)):
    p = get_project(s, pid)
    try:
        cam = S.camera_by_key(s, p, camera)
    except S.NotFound as e:
        raise HTTPException(404, str(e))
    when = _parse_dt(taken_at)
    m = get_methodology()
    out = []
    for f in files:
        data = await f.read()
        if len(data) > settings.max_upload_mb * 1024 * 1024:
            raise HTTPException(413, f"{f.filename}: файл больше {settings.max_upload_mb} МБ")
        try:
            snap, info = S.ingest_snapshot(s, p, cam, data, f.filename or "snapshot.jpg", when)
        except OSError:
            raise HTTPException(422, f"{f.filename}: не изображение")
        s.commit()
        devs = s.scalars(select(M.DeviationEvidence.deviation_id).where(M.DeviationEvidence.snapshot_id == snap.id)).all()
        row = {**S.snapshot_brief(snap, cam, m), **info, "deviation_ids": sorted(set(devs))}
        if settings.ai_auto and not info.get("duplicate") and snap.quality_ok:
            # OKO_AI_AUTO=1: каждый новый годный снимок сразу уходит на ИИ-анализ (в фоне, ответ не ждёт)
            try:
                row["ai_review_id"] = ai_jobs.submit_snapshot(s, snap).id
            except ai_jobs.AIDisabled:
                pass
        elif cam.is_overview and not info.get("duplicate") and snap.quality_ok:
            # кадр общего плана: стадия и готовность по модели готовности — в фоне (DINOv2 на CPU ≈ 30 с)
            ai_jobs.submit_readiness(snap.id)
        out.append(row)
    return out


@router.get("/projects/{pid}/snapshots")
def list_snapshots(pid: int, day: dt.date | None = None, camera: str | None = None, s: Session = Depends(get_session)):
    p = get_project(s, pid)
    m = get_methodology()
    cams = {c.id: c for c in p.cameras}
    q = select(M.Snapshot).where(M.Snapshot.project_id == pid)
    if day:
        start = dt.datetime.combine(day, dt.time.min)
        q = q.where(M.Snapshot.taken_at >= start, M.Snapshot.taken_at < start + dt.timedelta(days=1))
    if camera:
        q = q.where(M.Snapshot.camera_id == S.camera_by_key(s, p, camera).id)
    out = []
    for sn in s.scalars(q.order_by(M.Snapshot.taken_at, M.Snapshot.camera_id)):
        b = S.snapshot_brief(sn, cams[sn.camera_id], m)
        checks = s.scalars(select(M.SnapshotCheck).where(M.SnapshotCheck.snapshot_id == sn.id)).all()
        sev = [f["severity"] for c in checks for f in c.payload.get("findings", [])]
        b["findings"] = {k: sev.count(k) for k in ("critical", "warning", "info") if sev.count(k)}
        b["zones"] = [c.zone_key for c in checks]
        out.append(b)
    return out


@router.get("/snapshots/{sid}", summary="Карточка снимка: рамки, зоны, этапы графика, проверки, отклонения")
def snapshot_detail(sid: int, s: Session = Depends(get_session)):
    sn = s.get(M.Snapshot, sid)
    if sn is None:
        raise HTTPException(404, "снимок не найден")
    p = s.get(M.Project, sn.project_id)
    m = get_methodology()
    cam = s.get(M.Camera, sn.camera_id)
    zkeys = {z.id: z.key for z in p.zones}
    zmeta = {z.key: {"key": z.key, "name": z.name, "color": z.color} for z in p.zones}
    checks = s.scalars(select(M.SnapshotCheck).where(M.SnapshotCheck.snapshot_id == sid)).all()
    dev_ids = sorted(set(s.scalars(select(M.DeviationEvidence.deviation_id)
                                   .where(M.DeviationEvidence.snapshot_id == sid)).all()))
    devs = [S.deviation_view(s, d, m, p, with_evidence=False) for d in
            s.scalars(select(M.Deviation).where(M.Deviation.id.in_(dev_ids or [-1])))]
    return {**S.snapshot_brief(sn, cam, m),
            "brightness": sn.brightness, "contrast": sn.contrast, "sharpness": sn.sharpness,
            "assessment": sn.assessment, "original_name": sn.original_name,
            "boxes": [S.box_view(d, m, zkeys) for d in sn.detections],
            "zones": [{**zmeta[zkeys[cz.zone_id]], "polygon": cz.polygon} for cz in cam.zone_polygons
                      if cz.zone_id in zkeys],
            "checks": [c.payload for c in checks], "deviations": devs}


@router.get("/snapshots/{sid}/image", summary="Снимок: оригинал, превью или с разметкой")
def snapshot_image(sid: int, w: int = 0, annotate: int = 0, highlight: str = "", s: Session = Depends(get_session)):
    sn = s.get(M.Snapshot, sid)
    if sn is None or not Path(sn.file_path).exists():
        raise HTTPException(404, "снимок не найден")
    headers = {"Cache-Control": "public, max-age=86400"}
    if not annotate and not w:
        return Response(Path(sn.file_path).read_bytes(), media_type="image/jpeg", headers=headers)
    img = load_image(Path(sn.file_path).read_bytes())
    if annotate:
        m = get_methodology()
        p = s.get(M.Project, sn.project_id)
        cam = s.get(M.Camera, sn.camera_id)
        zmeta = {z.id: z for z in p.zones}
        hl = {int(x) for x in highlight.split(",") if x.strip().isdigit()}
        boxes = [{"id": d.id, "label": m.class_name(d.equipment_class), "conf": d.confidence,
                  "color": m.classes.get(d.equipment_class, {}).get("color", "#E1A21C"),
                  "x1": d.x1, "y1": d.y1, "x2": d.x2, "y2": d.y2,
                  "dashed": d.confidence < m.presence_threshold(d.equipment_class)}
                 for d in sn.detections if not d.is_rejected and (not hl or d.id in hl)]
        zones = [{"name": zmeta[cz.zone_id].name, "color": zmeta[cz.zone_id].color, "polygon": cz.polygon}
                 for cz in cam.zone_polygons if cz.zone_id in zmeta]
        return Response(render_annotated(img, boxes, zones, hl, width=w or None), media_type="image/jpeg",
                        headers=headers)
    return Response(thumbnail(img, w), media_type="image/jpeg", headers=headers)
