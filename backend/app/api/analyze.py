"""Быстрая проверка одного снимка без проекта и контракт AI-сервиса.

POST /api/analyze — снимок + вид(ы) работ справочника → техника на снимке, проверки методики,
отклонения. Удобно для жюри: любой снимок, любой этап, результат за один запрос.
POST /internal/detect, GET /internal/info — контракт AI-сервиса (backend/readme.md): тот же код,
запущенный на машине с GPU и весами RF-DETR, обслуживает основной сервис на CPU.
"""
from __future__ import annotations

import datetime as dt
import time

from fastapi import APIRouter, File, Form, HTTPException, Query, UploadFile

from ..detection import get_detector
from ..engine import DetInfo, EngineContext, Obs, TaskInfo, ZoneWindow, evaluate_window
from ..imaging import assess_quality, load_image, resolve_time, sha256
from ..methodology import get_methodology
from ..schedule_io import parse_equipment

router = APIRouter(tags=["Быстрая проверка и AI-сервис"])


@router.post("/api/analyze", summary="Проверить снимок против этапа работ (без проекта)")
async def analyze(file: UploadFile = File(...),
                  work_types: str = Form(..., description="id видов работ через запятую, например 12.3.7-1"),
                  planned: str = Form("", description="техника по графику: «экскаватор ×1; самосвал ×3»"),
                  tiles: int = Form(0, description="детекция по фрагментам 3×3 для общих планов"),
                  taken_at: str | None = Form(None)):
    m = get_methodology()
    ids = [w.strip() for w in work_types.split(",") if w.strip()]
    unknown = [w for w in ids if w not in m.work_types]
    if not ids or unknown:
        raise HTTPException(422, f"неизвестные виды работ: {unknown or 'не указаны'}")
    data = await file.read()
    t0 = time.perf_counter()
    try:
        img = load_image(data)
    except OSError:
        raise HTTPException(422, "файл не является изображением")
    when, source = resolve_time(data, file.filename or "", dt.datetime.fromisoformat(taken_at) if taken_at else None)
    q = assess_quality(img, m.config.get("quality", {}))
    det = get_detector()
    res = det.detect(img, tiles=tiles, sha256=sha256(data)) if q.ok else None
    t_det = time.perf_counter()
    valid, reason = q.ok, q.reason
    if res is not None and not res.recognized:          # детектор не подключён — «нет данных», а не «нет техники»
        valid, reason = False, "детектор не подключён: снимка нет в демо-разметке"
    dets = res.dets if res else []
    plan = parse_equipment(planned, m)
    tasks = []
    for i, w in enumerate(ids, 1):
        wt = m.work_types[w]
        tasks.append(TaskInfo(id=i, wbs=wt.id, name=wt.name, work_type_id=wt.id, profile=m.profiles[wt.profile],
                              observability=wt.observability, zone_key="FRAME", start=when.date(), end=when.date(),
                              planned=plan if i == 1 else {}))
    obs = Obs(snapshot_ids=[0], taken_at=when, valid=valid, reason=reason,
              dets=[DetInfo(k, d.cls, d.conf) for k, d in enumerate(dets)])
    zw = ZoneWindow(zone_key="FRAME", zone_name="Кадр", day=when.date(), obs=[obs], active=tasks)
    findings, check = evaluate_window(zw, EngineContext(m, frozenset(det.classes)))
    return {
        "detector": res.model if res else det.name, "detector_note": res.note if res else "",
        "taken_at": when.isoformat(), "time_source": source,
        "quality": {"ok": valid, "reason": reason, "brightness": q.brightness, "contrast": q.contrast,
                    "sharpness": q.sharpness},
        "width": img.width, "height": img.height,
        "boxes": [{"id": k, "cls": d.cls, "label": m.class_name(d.cls), "conf": round(d.conf, 3),
                   "xyxy": [round(v, 1) for v in d.box], "color": m.classes[d.cls].get("color"),
                   "strong": d.conf >= m.presence_threshold(d.cls)} for k, d in enumerate(dets)],
        "check": check,
        "findings": [{"rule": f.rule, "title": f.title, "severity": f.severity, "status": f.status,
                      "message": f.message, "recommendation": f.recommendation, "cls": f.cls, "role": f.role,
                      "metrics": f.metrics} for f in findings],
        "timing_ms": {"detect": round((t_det - t0) * 1000), "total": round((time.perf_counter() - t0) * 1000)},
    }


@router.get("/internal/info", summary="AI-сервис: версия модели и классы")
def internal_info():
    return get_detector().info()


@router.post("/internal/detect", summary="AI-сервис: детекция техники на одном кадре")
async def internal_detect(file: UploadFile = File(...), tiles: int = 0,
                          sha256_hint: str = Query("", alias="sha256", description="SHA-256 исходного файла")):
    data = await file.read()
    img = load_image(data)
    res = get_detector().detect(img, tiles=tiles, sha256=sha256_hint or sha256(data))
    return {"model_version": res.model, "note": res.note, "assessment": res.assessment, "recognized": res.recognized,
            "boxes": [{"cls": d.cls, "conf": round(d.conf, 4), "xyxy": [round(v, 1) for v in d.box]} for d in res.dets]}
