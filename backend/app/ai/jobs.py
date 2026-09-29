"""Фоновые задачи ИИ-анализа: модели на CPU и вызов LLM занимают от секунд до минут, поэтому запрос сразу
возвращает запись ai_reviews со статусом «в очереди», а интерфейс опрашивает её до «готово».

Конвейер снимка:
    1. модель готовности (DINOv2 + голова): стадия, готовность, план по графику → snapshots.assessment,
       пересчёт отклонений дня (STAGE_MISMATCH);
    2. сбор выводов детектора, модели готовности и правил (context.snapshot_context);
    3. Grounding DINO: прочие объекты по подсказкам (опалубка, леса, рабочие…) и проверка техники RF-DETR;
    4. локальная VLM описывает кадр и отмечает технику рамками; сверка рамок VLM и детектора (cross_check);
    5. LLM (YandexGPT / Claude / ChatGPT) проверяет все слои и исправляет ошибки моделей;
    6. сведение (fusion.fuse_snapshot) и, при OKO_AI_APPLY=auto, применение исправлений.

Скорость без потери качества:
  • шаги 1–4 (модели на CPU) идут в одном рабочем потоке, шаги 5–6 (сеть) — в отдельном пуле: пока LLM отвечает по
    одному кадру, модели уже считают следующий;
  • выводы детерминированных слоёв кешируются по содержимому кадра (cache.py): повторный анализ того же кадра и
    анализ кадра общего плана, оценённого в фоне при загрузке, заново вызывают только LLM;
  • модели загружаются в память сразу после старта сервиса (warmup), а не при первом анализе;
  • время каждого слоя пишется в итог (layers.timings) — видно, что занимает время.
Каждый шаг фиксируется в БД (status, step). После перезапуска незавершённые задачи помечаются ошибкой (recover).
"""
from __future__ import annotations

import datetime as dt
import logging
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import httpx
from sqlalchemy import select
from sqlalchemy.orm import Session

from .. import models as M
from .. import services as S
from ..config import settings
from ..db import SessionLocal
from ..imaging import load_image
from ..methodology import get_methodology
from . import cache
from . import context as C
from .fusion import apply_corrections, clean_day_output, cross_check, fuse_snapshot
from .llm import DAY_SCHEMA, LLMError, get_llm, snapshot_schema
from .openvocab import get_openvocab, summarize
from .readiness import assess_snapshot, assess_snapshot_ex, cached_predict, get_readiness
from .vlm import get_vlm

log = logging.getLogger("oko.ai")
_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="oko-ai")       # модели на CPU — по очереди
_llm_executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="oko-llm")  # запросы к LLM — параллельно
SYNC = False            # тесты: выполнять задачу сразу в вызывающем потоке


class AIDisabled(RuntimeError):
    pass


def not_json(llm, res) -> str:
    """Понятная причина, почему ответ LLM не разобрался как JSON."""
    raw = res.raw or {}
    if raw.get("finish_reason") == "length":
        why = f"ответ обрезан на лимите {raw.get('max_tokens')} токенов"
        if raw.get("reasoning"):
            why += " (модель рассуждала)"
        return f"{llm.name} ({llm.model}): {why} — увеличьте OKO_LLM_MAX_TOKENS или выберите модель без рассуждений"
    head = str(res.output.get("raw", ""))[:200].replace("\n", " ")
    return f"{llm.name} ({llm.model}) ответила не JSON: «{head}…»"


def status() -> dict:
    rd, vlm, llm, ov = get_readiness(), get_vlm(), get_llm(), get_openvocab()
    return {"readiness": rd.plan(), "open_vocab": ov.plan(), "vlm": vlm.plan(), "llm": llm.info(),
            "auto": settings.ai_auto, "apply": settings.ai_apply, "min_conf": settings.ai_apply_min_conf,
            "enabled": rd.enabled or ov.enabled or vlm.enabled or llm.enabled}


def _submit(fn, *args) -> None:
    if SYNC:
        fn(*args)
    else:
        _executor.submit(_safe, fn, *args)


def _submit_llm(fn, *args) -> None:
    if SYNC:
        fn(*args)
    else:
        _llm_executor.submit(_safe, fn, *args)


def _safe(fn, *args) -> None:
    try:
        fn(*args)
    except Exception:                       # ошибка уже записана в ai_reviews; здесь — только в журнал
        log.exception("ИИ-задача %s завершилась ошибкой", fn.__name__)


def warmup() -> None:
    """Загрузить модели ИИ-слоя в память после старта: первый анализ не ждёт загрузки весов.
    Порядок важен: VLM выбирается под память, оставшуюся после модели готовности и Grounding DINO."""
    for name, obj in (("модель готовности", get_readiness()), ("Grounding DINO", get_openvocab()),
                      ("VLM", get_vlm())):
        if getattr(obj, "enabled", False) and hasattr(obj, "warm"):
            t0 = time.perf_counter()
            try:
                obj.warm()
                log.info("прогрев: %s в памяти (%.0f с)", name, time.perf_counter() - t0)
            except Exception as e:           # нет памяти или сети — модель загрузится при первом анализе
                log.warning("прогрев: %s не загружена: %s", name, e)


def submit_warmup() -> bool:
    if not settings.ai_warmup:
        return False
    if not any(getattr(o, "enabled", False) for o in (get_readiness(), get_openvocab(), get_vlm())):
        return False
    _executor.submit(_safe, warmup)
    return True


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
    _submit_llm(run_day, r.id)                   # только LLM — не ждёт очереди моделей
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


def _timed(layers: dict, name: str, t0: float, hit: bool = False) -> None:
    layers.setdefault("timings", {})[name] = round(time.perf_counter() - t0, 1)
    if hit:
        layers.setdefault("cached", []).append(name)


def _open_vocab(ctx: dict, img_fn, sha: str, step, errors: list, layers: dict, zone_of=None) -> dict | None:
    """Слой Grounding DINO: прочие объекты и техника по подсказкам → ctx["open_vocab"] (без пиксельных рамок).
    Возвращает полный итог слоя (с рамками в пикселях — для интерфейса) или None."""
    ov = get_openvocab()
    if not ov.enabled or not ctx["quality"]["ok"]:
        return None
    step(f"Grounding DINO ({ov.model_name.split('/')[-1]}): опалубка, леса, рабочие и техника по подсказкам")
    t0 = time.perf_counter()
    try:
        W, H = ctx["image"]["width"], ctx["image"]["height"]
        eq_px = [[d["box"][0] * W, d["box"][1] * H, d["box"][2] * W, d["box"][3] * H] for d in ctx["detections"]]
        if hasattr(ov, "cache_key"):
            raw, hit = cache.cached("open_vocab", ov.cache_key(sha, eq_px), lambda: ov.detect(img_fn(), eq_px))
        else:
            raw, hit = ov.detect(img_fn(), eq_px), False
        full = summarize(raw, ctx["detections"], zone_of)
    except Exception as e:                   # нет памяти или весов — анализ продолжается без слоя
        log.warning("Grounding DINO: %s", e)
        errors.append(f"Grounding DINO: {e}")
        return None
    _timed(layers, "open_vocab", t0, hit)
    strip = lambda items: [{k: v for k, v in x.items() if k != "xyxy"} for x in items]  # noqa: E731
    ctx["open_vocab"] = {**full, "objects": strip(full["objects"]), "only_open_vocab": strip(full["only_open_vocab"])}
    layers["open_vocab_model"] = full["model"]
    return full


def _vlm(ctx: dict, img_fn, sha: str, step, errors: list, layers: dict, hint: str = "") -> tuple[str, dict | None]:
    """Слой VLM и сверка её рамок с детектором. Дополняет ctx (vlm, cross_check); возвращает (модель, ответ)."""
    vlm = get_vlm()
    if not vlm.enabled:
        return "", None
    plan = vlm.plan()
    step(" ".join(x for x in ("локальная VLM", plan.get("model") or "", "описывает кадр") if x))
    t0 = time.perf_counter()

    def compute() -> dict:
        res = vlm.describe(img_fn(), hint=hint)
        return {"model": res.model, "device": res.device, "output": res.output, "seconds": res.seconds}
    try:
        if hasattr(vlm, "cache_key"):
            v, hit = cache.cached("vlm", vlm.cache_key(sha, hint), compute)
        else:
            v, hit = compute(), False
        vlm_model = v["model"]
        vlm_out = {**v["output"], "_seconds": round(v["seconds"], 1), "_device": v["device"]}
        _timed(layers, "vlm", t0, hit)
        if plan.get("reason"):                # почему выбрана именно эта модель (память, OKO_VLM)
            layers["vlm_reason"] = plan["reason"]
    except Exception as e:                   # нет памяти, нет transformers — анализ продолжается без VLM
        log.warning("VLM: %s", e)
        vlm_model, vlm_out = "", {"error": str(e)[:500]}
        errors.append(f"VLM: {e}")
    ctx["vlm"] = {k: v for k, v in vlm_out.items() if not k.startswith("_")} | {"model": vlm_model}
    if "error" not in vlm_out:
        ctx["cross_check"] = cross_check(get_methodology(), ctx["detections"], vlm_out)
    return vlm_model, vlm_out


def _ask_llm(ctx: dict, images: list[bytes], errors: list, layers: dict):
    llm = get_llm()
    if not llm.enabled:
        return None
    t0 = time.perf_counter()
    try:
        res = llm.ask(C.snapshot_instructions(llm.vision), ctx, snapshot_schema(sorted(get_methodology().classes)),
                      "snapshot_review", images)
        if "raw" in res.output:
            errors.append(not_json(llm, res))
        return res
    except (LLMError, httpx.HTTPError, ValueError) as e:
        log.warning("LLM: %s", e)
        errors.append(f"{llm.name}: {e}")
        return None
    finally:
        _timed(layers, "llm", t0)


def _layers_meta(layers: dict, vlm_model: str, llm_res) -> dict:
    llm = get_llm()
    return {**layers, "vlm_model": vlm_model, "llm_model": llm_res.model if llm_res else (llm.model if llm.enabled else ""),
            "llm_provider": llm_res.provider if llm_res else (llm.provider if llm.enabled else ""),
            "llm_name": llm.name if llm.enabled else "", "llm_vision": llm.vision if llm.enabled else None}


def _ok(llm_out, vlm_out, layers) -> bool:
    return (bool(llm_out) and "raw" not in llm_out) or (vlm_out is not None and "error" not in vlm_out) \
        or bool(layers.get("readiness_model")) or bool(layers.get("open_vocab_model"))


def _fail(rid: int, s: Session, t0: float, e: Exception) -> None:
    s.rollback()
    r = s.get(M.AIReview, rid)
    r.status, r.step, r.error = "error", "", str(e)[:2000]
    r.duration_ms = round((time.perf_counter() - t0) * 1000)
    s.commit()


def run_snapshot(rid: int) -> None:
    """Шаги 1–4 на CPU; шаги 5–6 уходят в пул LLM (finish_snapshot), модели берутся за следующий кадр."""
    t0 = time.perf_counter()
    s = SessionLocal()
    r = s.get(M.AIReview, rid)
    payload = None
    try:
        snap = s.get(M.Snapshot, r.snapshot_id)
        rd, llm = get_readiness(), get_llm()
        errors, layers = [], {"timings": {}, "cached": []}
        step = lambda text: _step(s, r, text)  # noqa: E731

        if rd.enabled and snap.quality_ok:
            step("модель готовности (DINOv2 + голова): стадия и готовность")
            t1 = time.perf_counter()
            try:
                a, hit = assess_snapshot_ex(s, snap)
                s.commit()
                layers["readiness_model"] = (a or {}).get("model")
                _timed(layers, "readiness", t1, hit)
            except Exception as e:           # нет памяти или весов DINOv2 — анализ продолжается без слоя
                s.rollback()
                log.warning("модель готовности: %s", e)
                errors.append(f"модель готовности: {e}")

        step("сбор выводов детектора, модели готовности и правил")
        ctx = C.snapshot_context(s, snap)
        loaded = {}

        def img_fn():                        # кадр читается с диска, только если хоть один слой не из кеша
            if "img" not in loaded:
                loaded["img"] = load_image(Path(snap.file_path).read_bytes())
            return loaded["img"]
        p = s.get(M.Project, snap.project_id)
        zkeys = {z.id: z.key for z in p.zones}
        assign = S.zone_assigner(s.get(M.Camera, snap.camera_id), zkeys, snap.frame_zone, snap.width, snap.height,
                                 get_methodology())
        ov_full = _open_vocab(ctx, img_fn, snap.sha256, step, errors, layers,
                              zone_of=lambda xyxy: zkeys.get(assign("", xyxy)))
        hint = f"камера {ctx['camera']['name']}, зоны: {', '.join(z['name'] for z in ctx['zones']) or 'вся площадка'}"
        vlm_model, vlm_out = _vlm(ctx, img_fn, snap.sha256, step, errors, layers, hint)
        r.vlm_model, r.vlm_output = vlm_model, vlm_out
        r.context = ctx
        images = C.llm_images(s, snap) if llm.enabled and llm.vision else []
        if llm.enabled:
            step(f"{llm.name} ({llm.model}) сверяет все слои")
        else:
            s.commit()
        payload = {"ctx": ctx, "images": images, "vlm_model": vlm_model, "vlm_out": vlm_out, "ov_full": ov_full,
                   "errors": errors, "layers": layers, "t0": t0}
    except Exception as e:
        _fail(rid, s, t0, e)
        raise
    finally:
        s.close()
    _submit_llm(finish_snapshot, rid, payload)


def finish_snapshot(rid: int, payload: dict) -> None:
    """Шаги 5–6: LLM сверяет слои, сведение, сохранение, при OKO_AI_APPLY=auto — применение исправлений."""
    t0 = payload["t0"]
    s = SessionLocal()
    try:
        r = s.get(M.AIReview, rid)
        ctx, errors, layers = payload["ctx"], payload["errors"], payload["layers"]
        vlm_out = payload["vlm_out"]
        llm = get_llm()
        llm_res = _ask_llm(ctx, payload["images"], errors, layers)
        llm_out = llm_res.output if llm_res else None
        if llm_res:
            r.llm_provider, r.llm_model = llm_res.provider, llm_res.model
            r.tokens_in, r.tokens_out = llm_res.tokens_in, llm_res.tokens_out
            r.llm_output = llm_out
        elif llm.enabled:
            r.llm_provider, r.llm_model = llm.provider, llm.model
        r.context = ctx
        layers = _layers_meta(layers, payload["vlm_model"], llm_res)
        r.final = fuse_snapshot(ctx, vlm_out, llm_out, settings.ai_apply_min_conf, meta=layers,
                                open_vocab=payload["ov_full"])
        ok = _ok(llm_out, vlm_out, layers)
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
        _fail(rid, s, t0, e)
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
        r.status, r.step, r.error = ("error" if bad else "done"), "", (not_json(llm, res) if bad else "")
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


# ------------------------------------------------------------------ быстрая проверка без проекта

_quick: dict[str, dict] = {}
_quick_lock = threading.Lock()
QUICK_KEEP = 30                 # результаты быстрых проверок хранятся в памяти, последние 30


def _qset(jid: str, **kw) -> None:
    with _quick_lock:
        if jid in _quick:
            _quick[jid].update(kw)


def quick_view(jid: str, full: bool = False) -> dict | None:
    with _quick_lock:
        j = _quick.get(jid)
        if j is None:
            return None
        return {k: v for k, v in j.items() if not k.startswith("_") and (full or k != "context")}


def submit_quick(img, res: dict, work_type_ids: list[str], filename: str) -> dict:
    """Снимок из «Проверить снимок»: тот же конвейер ИИ, результат — в памяти (в БД снимок не попадает)."""
    if not status()["enabled"]:
        raise AIDisabled("ИИ-анализ выключен: задайте OKO_LLM_PROVIDER=yandex, OKO_VLM=auto и/или положите "
                         "веса модели готовности в weights/readiness/")
    jid = "q" + uuid.uuid4().hex[:12]
    with _quick_lock:
        for old in sorted(_quick, key=lambda k: _quick[k]["created"])[:-QUICK_KEEP + 1 or None]:
            _quick.pop(old, None)
        _quick[jid] = {"id": jid, "kind": "quick", "status": "queued", "step": "в очереди", "created": time.time(),
                       "final": None, "error": "", "applied": None, "_img": img, "_res": res, "_wts": work_type_ids,
                       "_name": filename}
    _submit(run_quick, jid)
    return quick_view(jid)


def run_quick(jid: str) -> None:
    t0 = time.perf_counter()
    with _quick_lock:
        job = _quick.get(jid)
        if job is None:
            return
        img, res, wts, name = job.pop("_img"), job.pop("_res"), job.pop("_wts"), job.pop("_name")
    try:
        m = get_methodology()
        rd, llm = get_readiness(), get_llm()
        errors, layers = [], {"timings": {}, "cached": []}
        ctx = C.quick_context(res, wts, name)
        sha = res.get("sha256") or ""
        step = lambda text: _qset(jid, status="running", step=text)  # noqa: E731
        assessment = None
        if rd.enabled and res["quality"]["ok"]:
            step("модель готовности (DINOv2 + голова): стадия и готовность")
            t1 = time.perf_counter()
            try:
                dets = [(b["cls"], b["conf"], tuple(b["xyxy"])) for b in res["boxes"] if b["conf"] >= 0.3]
                assessment, hit = cached_predict(rd, lambda: img, dets, m.normalize_class, sha)
                assessment["note"] = "модель обучена на общих планах площадки; на крупном плане оценка ориентировочная"
                ctx["stage"]["readiness_model"] = C.assessment_brief(assessment)
                layers["readiness_model"] = assessment.get("model")
                _timed(layers, "readiness", t1, hit)
            except Exception as e:
                log.warning("модель готовности: %s", e)
                errors.append(f"модель готовности: {e}")
        boxes = [{"id": b["id"], "label": f"#{b['id']} {b['label']}", "conf": b["conf"], "color": b.get("color") or "#E1A21C",
                  "x1": b["xyxy"][0], "y1": b["xyxy"][1], "x2": b["xyxy"][2], "y2": b["xyxy"][3],
                  "dashed": not b["strong"]} for b in res["boxes"]]
        ov_full = _open_vocab(ctx, lambda: img, sha, step, errors, layers, zone_of=lambda xyxy: "кадр")
        vlm_model, vlm_out = _vlm(ctx, lambda: img, sha, step, errors, layers, "быстрая проверка снимка стройплощадки")
        images = C.frame_images(img, boxes) if llm.enabled and llm.vision else []
        if llm.enabled:
            step(f"{llm.name} ({llm.model}) сверяет все слои")
        payload = {"ctx": ctx, "images": images, "vlm_model": vlm_model, "vlm_out": vlm_out, "ov_full": ov_full,
                   "errors": errors, "layers": layers, "t0": t0, "assessment": assessment}
    except Exception as e:
        _qset(jid, status="error", step="", error=str(e)[:2000], duration_ms=round((time.perf_counter() - t0) * 1000))
        raise
    _submit_llm(finish_quick, jid, payload)


def finish_quick(jid: str, payload: dict) -> None:
    t0 = payload["t0"]
    try:
        ctx, errors, layers, vlm_out = payload["ctx"], payload["errors"], payload["layers"], payload["vlm_out"]
        llm_res = _ask_llm(ctx, payload["images"], errors, layers)
        llm_out = llm_res.output if llm_res else None
        layers = _layers_meta(layers, payload["vlm_model"], llm_res)
        final = fuse_snapshot(ctx, vlm_out, llm_out, settings.ai_apply_min_conf, meta=layers,
                              open_vocab=payload["ov_full"])
        final["assessment"] = payload["assessment"]
        ok = _ok(llm_out, vlm_out, layers)
        _qset(jid, status="done" if ok else "error", step="", final=final, error="; ".join(errors)[:2000],
              vlm={"model": payload["vlm_model"], "output": vlm_out} if vlm_out else None,
              llm={"provider": llm_res.provider, "model": llm_res.model, "tokens_in": llm_res.tokens_in,
                   "tokens_out": llm_res.tokens_out, "output": llm_out} if llm_res else None,
              context=ctx, duration_ms=round((time.perf_counter() - t0) * 1000))
    except Exception as e:
        _qset(jid, status="error", step="", error=str(e)[:2000], duration_ms=round((time.perf_counter() - t0) * 1000))
        raise
