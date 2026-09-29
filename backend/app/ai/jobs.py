"""Фоновые задачи ИИ-анализа: модель готовности и VLM на CPU и вызов LLM занимают от секунд до минут, поэтому
запрос сразу возвращает запись ai_reviews со статусом «в очереди», а интерфейс опрашивает её до «готово».

Конвейер снимка (run_snapshot):
    1. модель готовности (DINOv2 + голова): стадия, готовность, план по графику → snapshots.assessment,
       пересчёт отклонений дня (STAGE_MISMATCH);
    2. сбор выводов детектора, модели готовности и правил (context.snapshot_context);
    3. локальная VLM описывает кадр и отмечает технику рамками; сверка рамок VLM и детектора (cross_check);
    4. LLM (YandexGPT / Claude / ChatGPT) проверяет все слои и исправляет ошибки моделей;
    5. сведение (fusion.fuse_snapshot) и, при OKO_AI_APPLY=auto, применение исправлений.

Один рабочий поток: модели держатся в памяти и обрабатывают снимки по очереди. Каждый шаг фиксируется в БД
(status, step), поэтому видно, на каком слое идёт анализ. После перезапуска незавершённые задачи помечаются
ошибкой (recover).
"""
from __future__ import annotations

import datetime as dt
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import httpx
from sqlalchemy import select
from sqlalchemy.orm import Session

from .. import models as M
from ..config import settings
from ..db import SessionLocal
from ..imaging import load_image
from ..methodology import get_methodology
from . import context as C
from .fusion import apply_corrections, clean_day_output, cross_check, fuse_snapshot
from .llm import DAY_SCHEMA, LLMError, get_llm, snapshot_schema
from .readiness import assess_snapshot, get_readiness
from .vlm import get_vlm

log = logging.getLogger("oko.ai")
_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="oko-ai")
SYNC = False            # тесты: выполнять задачу сразу в вызывающем потоке


class AIDisabled(RuntimeError):
    pass


def status() -> dict:
    rd, vlm, llm = get_readiness(), get_vlm(), get_llm()
    return {"readiness": rd.plan(), "vlm": vlm.plan(), "llm": llm.info(), "auto": settings.ai_auto,
            "apply": settings.ai_apply, "min_conf": settings.ai_apply_min_conf,
            "enabled": rd.enabled or vlm.enabled or llm.enabled}


def _submit(fn, arg) -> None:
    if SYNC:
        fn(arg)
    else:
        _executor.submit(_safe, fn, arg)


def _safe(fn, arg) -> None:
    try:
        fn(arg)
    except Exception:                       # ошибка уже записана в ai_reviews; здесь — только в журнал
        log.exception("ИИ-задача %s(%s) завершилась ошибкой", fn.__name__, arg)


def _active(s: Session, **where) -> M.AIReview | None:
    q = select(M.AIReview).where(M.AIReview.status.in_(("queued", "running")))
    for k, v in where.items():
        q = q.where(getattr(M.AIReview, k) == v)
    return s.scalar(q.order_by(M.AIReview.id.desc()))


def submit_snapshot(s: Session, snap: M.Snapshot) -> M.AIReview:
    if not status()["enabled"]:
        raise AIDisabled("ИИ-анализ выключен: задайте OKO_LLM_PROVIDER=yandex, OKO_VLM=auto и/или положите "
                         "веса модели готовности в weights/readiness/")
    r = _active(s, snapshot_id=snap.id, kind="snapshot")
    if r:
        return r
    r = M.AIReview(project_id=snap.project_id, snapshot_id=snap.id, day=snap.taken_at.date(), kind="snapshot",
                   status="queued", step="в очереди")
    s.add(r)
    s.commit()
    _submit(run_snapshot, r.id)
    s.refresh(r)
    return r


def submit_day(s: Session, p: M.Project, day: dt.date) -> M.AIReview:
    if not get_llm().enabled:
        raise AIDisabled("Анализ дня делает LLM: задайте OKO_LLM_PROVIDER=yandex и OKO_YC_FOLDER_ID")
    r = _active(s, project_id=p.id, kind="day", day=day)
    if r:
        return r
    r = M.AIReview(project_id=p.id, day=day, kind="day", status="queued", step="в очереди")
    s.add(r)
    s.commit()
    _submit(run_day, r.id)
    s.refresh(r)
    return r


def submit_readiness(snapshot_id: int) -> None:
    """Оценка модели готовности для нового снимка общего плана — в фоне, загрузка снимка её не ждёт."""
    if get_readiness().enabled:
        _submit(run_readiness, snapshot_id)


def run_readiness(snapshot_id: int) -> None:
    s = SessionLocal()
    try:
        snap = s.get(M.Snapshot, snapshot_id)
        if snap is not None and snap.quality_ok:
            assess_snapshot(s, snap)
            s.commit()
    except Exception:
        s.rollback()
        raise
    finally:
        s.close()


def _step(s: Session, r: M.AIReview, text: str) -> None:
    r.status, r.step = "running", text
    s.commit()


def run_snapshot(rid: int) -> None:
    t0 = time.perf_counter()
    s = SessionLocal()
    r = s.get(M.AIReview, rid)
    try:
        snap = s.get(M.Snapshot, r.snapshot_id)
        m = get_methodology()
        rd, vlm, llm = get_readiness(), get_vlm(), get_llm()
        errors, layers = [], {}

        if rd.enabled and snap.quality_ok:
            _step(s, r, "модель готовности (DINOv2 + голова): стадия и готовность")
            try:
                a = assess_snapshot(s, snap)
                s.commit()
                layers["readiness_model"] = (a or {}).get("model")
            except Exception as e:           # нет памяти или весов DINOv2 — анализ продолжается без слоя
                s.rollback()
                log.warning("модель готовности: %s", e)
                errors.append(f"модель готовности: {e}")

        _step(s, r, "сбор выводов детектора, модели готовности и правил")
        ctx = C.snapshot_context(s, snap)

        vlm_out = None
        if vlm.enabled:
            plan = vlm.plan()
            _step(s, r, " ".join(x for x in ("локальная VLM", plan.get("model") or "", "описывает кадр") if x))
            try:
                zones = ", ".join(z["name"] for z in ctx["zones"]) or "вся площадка"
                res = vlm.describe(load_image(Path(snap.file_path).read_bytes()),
                                   hint=f"камера {ctx['camera']['name']}, зоны: {zones}")
                r.vlm_model = res.model
                vlm_out = {**res.output, "_seconds": round(res.seconds, 1), "_device": res.device}
            except Exception as e:           # нет памяти, нет transformers — анализ продолжается без VLM
                log.warning("VLM: %s", e)
                vlm_out = {"error": str(e)[:500]}
                errors.append(f"VLM: {e}")
            r.vlm_output = vlm_out
            ctx["vlm"] = {k: v for k, v in vlm_out.items() if not k.startswith("_")} | {"model": r.vlm_model}
            if "error" not in vlm_out:
                ctx["cross_check"] = cross_check(m, ctx["detections"], vlm_out)
            s.commit()

        llm_out = None
        if llm.enabled:
            images = C.llm_images(s, snap) if llm.vision else []   # до commit: транзакция не держится во время запроса
            _step(s, r, f"{llm.name} ({llm.model}) сверяет все слои")
            try:
                res = llm.ask(C.snapshot_instructions(llm.vision), ctx, snapshot_schema(sorted(m.classes)),
                              "snapshot_review", images)
                llm_out = res.output
                r.llm_provider, r.llm_model = res.provider, res.model
                r.tokens_in, r.tokens_out = res.tokens_in, res.tokens_out
                r.llm_output = llm_out
                if "raw" in llm_out:
                    errors.append(f"{llm.name} ответила не JSON")
            except (LLMError, httpx.HTTPError, ValueError) as e:
                log.warning("LLM: %s", e)
                errors.append(f"{llm.name}: {e}")
                r.llm_provider, r.llm_model = llm.provider, llm.model

        _step(s, r, "сведение результатов")
        r.context = ctx
        layers.update({"vlm_model": r.vlm_model, "llm_model": r.llm_model, "llm_provider": r.llm_provider,
                       "llm_name": llm.name if llm.enabled else "", "llm_vision": llm.vision if llm.enabled else None})
        r.final = fuse_snapshot(ctx, vlm_out, llm_out, settings.ai_apply_min_conf, meta=layers)
        ok = (bool(llm_out) and "raw" not in llm_out) or (vlm_out is not None and "error" not in vlm_out) \
            or bool(layers.get("readiness_model"))
        r.status, r.step, r.error = ("done" if ok else "error"), "", "; ".join(errors)[:2000]
        r.duration_ms = round((time.perf_counter() - t0) * 1000)
        s.commit()
        if ok and llm_out and settings.ai_apply == "auto" and any(c.get("apply") for c in r.final["corrections"]):
            try:                                 # анализ уже сохранён; сбой применения его не портит
                apply_corrections(s, r)
                s.commit()
            except Exception as e:
                s.rollback()
                log.warning("автоприменение исправлений: %s", e)
                r = s.get(M.AIReview, rid)
                r.error = (r.error + "; " if r.error else "") + f"исправления не применены: {e}"[:500]
                s.commit()
    except Exception as e:
        s.rollback()
        r = s.get(M.AIReview, rid)
        r.status, r.step, r.error = "error", "", str(e)[:2000]
        r.duration_ms = round((time.perf_counter() - t0) * 1000)
        s.commit()
        raise
    finally:
        s.close()


def run_day(rid: int) -> None:
    t0 = time.perf_counter()
    s = SessionLocal()
    r = s.get(M.AIReview, rid)
    try:
        p = s.get(M.Project, r.project_id)
        llm = get_llm()
        _step(s, r, "сбор сводки дня, отклонений и анализа снимков")
        ctx = C.day_context(s, p, r.day)
        images = C.day_images(s, p, r.day) if llm.vision else []
        r.context = ctx
        _step(s, r, f"{llm.name} ({llm.model}) анализирует день")
        res = llm.ask(C.day_instructions(llm.vision), ctx, DAY_SCHEMA, "day_review", images)
        r.llm_provider, r.llm_model = res.provider, res.model
        r.tokens_in, r.tokens_out = res.tokens_in, res.tokens_out
        r.llm_output = res.output
        bad = "raw" in res.output
        r.final = {**(res.output if bad else clean_day_output(res.output)),
                   "layers": {"llm_provider": res.provider, "llm_model": res.model, "llm_name": llm.name,
                              "snapshots_reviewed": len(ctx["ai_snapshots"]),
                              "readiness_frames": len(ctx["stage"]["readiness_model"])}}
        r.status, r.step, r.error = ("error" if bad else "done"), "", (f"{llm.name} ответила не JSON" if bad else "")
        r.duration_ms = round((time.perf_counter() - t0) * 1000)
        s.commit()
    except Exception as e:
        s.rollback()
        r = s.get(M.AIReview, rid)
        r.status, r.step, r.error = "error", "", str(e)[:2000]
        r.duration_ms = round((time.perf_counter() - t0) * 1000)
        s.commit()
        if not isinstance(e, (LLMError, httpx.HTTPError)):
            raise
    finally:
        s.close()


def recover(s: Session) -> int:
    """После перезапуска: задачи «в очереди» и «выполняется» больше никто не выполнит."""
    n = 0
    for r in s.scalars(select(M.AIReview).where(M.AIReview.status.in_(("queued", "running")))):
        r.status, r.step, r.error = "error", "", "прервано перезапуском сервиса — запустите анализ снова"
        n += 1
    return n
