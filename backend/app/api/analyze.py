"""Быстрая проверка одного снимка без проекта и контракт AI-сервиса.

POST /api/analyze — снимок + вид(ы) работ справочника → техника на снимке, проверки методики,
отклонения. Удобно для жюри: любой снимок, любой этап, результат за один запрос. С ai=1 в том же запросе
ставится ИИ-анализ снимка — без повторной загрузки файла и повторной детекции.
POST /internal/detect, GET /internal/info — контракт AI-сервиса (backend/readme.md): тот же код,
запущенный на машине с GPU и весами RF-DETR, обслуживает основной сервис на CPU.
"""
from __future__ import annotations

import datetime as dt
import threading
import time
from collections import OrderedDict

from fastapi import APIRouter, File, Form, HTTPException, Query, UploadFile

from ..ai import jobs as ai_jobs
from ..detection import get_detector
from ..engine import DetInfo, EngineContext, Obs, TaskInfo, ZoneWindow, evaluate_window
from ..imaging import assess_quality, load_image, resolve_time, sha256
from ..methodology import get_methodology
from ..schedule_io import parse_equipment

router = APIRouter(tags=["Быстрая проверка и AI-сервис"])

# Детекция по содержимому файла: «Повторить анализ» и повторная проверка того же снимка не запускают детектор
# заново (RF-DETR на CPU — секунды, по фрагментам 3×3 — десятки секунд). Детектор детерминирован.
_det_cache: OrderedDict = OrderedDict()
_det_lock = threading.Lock()
DET_CACHE_SIZE = 32


def _detect(det, img, tiles: int, digest: str):
    key = (id(det), det.name, getattr(det, "threshold", None), tiles, digest)
    with _det_lock:
        if key in _det_cache:
            _det_cache.move_to_end(key)
            return _det_cache[key]
    res = det.detect(img, tiles=tiles, sha256=digest)
    with _det_lock:
        _det_cache[key] = res
        while len(_det_cache) > DET_CACHE_SIZE:
            _det_cache.popitem(last=False)
    return res


# обычные (не async) обработчики: детекция на CPU идёт в пуле потоков и не останавливает остальные запросы
@router.post("/api/analyze", summary="Проверить снимок против этапа работ (без проекта)")
def analyze(file: UploadFile = File(...),
            work_types: str = Form(..., description="id видов работ через запятую, например 12.3.7-1"),
            planned: str = Form("", description="техника по графику: «экскаватор ×1; самосвал ×3»"),
            tiles: int = Form(0, description="детекция по фрагментам 3×3 для общих планов"),
            taken_at: str | None = Form(None),
            ai: int = Form(0, description="1 — сразу поставить ИИ-анализ снимка (ответ: ai_job)")):
    res, img, ids = quick_check(file.file.read(), file.filename or "", work_types, planned, tiles, taken_at)
    if ai:
        if not res["quality"]["ok"]:
            res["ai_note"] = "снимок не прошёл контроль качества — анализ ИИ не запущен автоматически"
        else:
            try:
                res["ai_job"] = ai_jobs.submit_quick(img, dict(res), ids, file.filename or "")
            except ai_jobs.AIDisabled as e:
                res["ai_note"] = str(e)
    return res


@router.post("/api/analyze/ai", status_code=202,
             summary="Быстрая проверка + ИИ-анализ: модель готовности, VLM и LLM (YandexGPT) по снимку без проекта")
def analyze_ai(file: UploadFile = File(...), work_types: str = Form(...), planned: str = Form(""),
               tiles: int = Form(0), taken_at: str | None = Form(None)):
    res, img, ids = quick_check(file.file.read(), file.filename or "", work_types, planned, tiles, taken_at)
    try:
        return ai_jobs.submit_quick(img, res, ids, file.filename or "")
    except ai_jobs.AIDisabled as e:
        raise HTTPException(409, str(e))


@router.get("/api/analyze/ai/{jid}", summary="Результат ИИ-анализа быстрой проверки; full=1 — с контекстом для LLM")
def analyze_ai_result(jid: str, full: int = 0):
    v = ai_jobs.quick_view(jid, bool(full))
    if v is None:
        raise HTTPException(404, "анализ не найден (результаты быстрых проверок хранятся до перезапуска сервиса)")
    return v


def quick_check(data: bytes, filename: str, work_types: str, planned: str, tiles: int, taken_at: str | None):
    """Снимок + виды работ → техника, проверки методики, предупреждения. Возвращает (ответ, кадр, виды работ)."""
    m = get_methodology()
    ids = [w.strip() for w in work_types.split(",") if w.strip()]
    unknown = [w for w in ids if w not in m.work_types]
    if not ids or unknown:
        raise HTTPException(422, f"неизвестные виды работ: {unknown or 'не указаны'}")
    t0 = time.perf_counter()
    try:
        img = load_image(data)
    except OSError:
        raise HTTPException(422, "файл не является изображением")
    when, source = resolve_time(data, filename, dt.datetime.fromisoformat(taken_at) if taken_at else None)
    q = assess_quality(img, m.config.get("quality", {}))
    det = get_detector()
    digest = sha256(data)
    res = _detect(det, img, tiles, digest) if q.ok else None
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
        "width": img.width, "height": img.height, "sha256": digest,
        "boxes": [{"id": k, "cls": d.cls, "label": m.class_name(d.cls), "conf": round(d.conf, 3),
                   "xyxy": [round(v, 1) for v in d.box], "color": m.classes[d.cls].get("color"),
                   "strong": d.conf >= m.presence_threshold(d.cls)} for k, d in enumerate(dets)],
        "check": check,
        "findings": [{"rule": f.rule, "title": f.title, "severity": f.severity, "status": f.status,
                      "message": f.message, "recommendation": f.recommendation, "cls": f.cls, "role": f.role,
                      "metrics": f.metrics} for f in findings],
        "timing_ms": {"detect": round((t_det - t0) * 1000), "total": round((time.perf_counter() - t0) * 1000)},
    }, img, ids


@router.get("/internal/info", summary="AI-сервис: версия модели и классы")
def internal_info():
    return get_detector().info()


@router.post("/internal/detect", summary="AI-сервис: детекция техники на одном кадре")
def internal_detect(file: UploadFile = File(...), tiles: int = 0,
                    sha256_hint: str = Query("", alias="sha256", description="SHA-256 исходного файла")):
    data = file.file.read()
    img = load_image(data)
    res = get_detector().detect(img, tiles=tiles, sha256=sha256_hint or sha256(data))
    return {"model_version": res.model, "note": res.note, "assessment": res.assessment, "recognized": res.recognized,
            "boxes": [{"cls": d.cls, "conf": round(d.conf, 4), "xyxy": [round(v, 1) for v in d.box]} for d in res.dets]}
