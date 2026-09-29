"""Сценарии сервиса поверх БД: проект, график, камеры, приём снимка, пересчёт отклонений.

Поток одного снимка (ingest_snapshot):
  файл → SHA-256 (повторная загрузка не дублирует данные) → время съёмки → контроль качества →
  детекция (RF-DETR / AI-сервис / демо) → зоны по точке опоры → объяснение по снимку →
  пересчёт отклонений зоны за день → ответ за доли секунды после детекции.
"""
from __future__ import annotations

import datetime as dt
import logging
import time
from collections import defaultdict
from pathlib import Path

from sqlalchemy import delete, func, select, text
from sqlalchemy.orm import Session

from . import models as M
from .config import settings
from .detection import get_detector
from .engine import (SITE_ZONE, DetInfo, EngineContext, Finding, Obs, TaskInfo, ZoneWindow, evaluate_window,
                     tasks_for_zone)
from .imaging import assess_quality, load_image, resolve_time, sha256
from .methodology import Methodology, get_methodology
from .schedule_io import parse_schedule
from .zones import assign_zone

log = logging.getLogger("oko.services")
ZONE_COLORS = ["#2A78D6", "#EB6834", "#1BAF7A", "#EDA100", "#E87BA4", "#008300", "#7A5CC2", "#B83C8C", "#2F8FB5"]
STAGE_ORDER = {"S1": 1, "S2": 2, "S3": 3, "S4": 4, "S5": 5}


class NotFound(Exception):
    pass


# ============================================================ проект, зоны, камеры

def create_project(s: Session, name: str, object_type: str = "housing", address: str = "", description: str = "",
                   is_demo: bool = False) -> M.Project:
    p = M.Project(name=name, object_type=object_type, address=address, description=description, is_demo=is_demo)
    s.add(p)
    s.flush()
    return p


def zone_by_key(s: Session, project: M.Project, key: str) -> M.Zone | None:
    return s.scalar(select(M.Zone).where(M.Zone.project_id == project.id, M.Zone.key == key))


def ensure_zone(s: Session, project: M.Project, name_or_key: str) -> M.Zone:
    """Зона по ключу или названию; неизвестная зона из графика создаётся (без полигонов)."""
    val = name_or_key.strip()
    for z in project.zones:
        if z.key.lower() == val.lower() or z.name.lower() == val.lower():
            return z
    n = len(project.zones)
    key = f"Z{n + 1}"
    while zone_by_key(s, project, key):
        n += 1
        key = f"Z{n + 1}"
    z = M.Zone(project_id=project.id, key=key, name=val, color=ZONE_COLORS[n % len(ZONE_COLORS)])
    s.add(z)
    s.flush()
    s.refresh(project)
    return z


def apply_site_config(s: Session, project: M.Project, cfg: dict) -> None:
    """Зоны, камеры с полигонами и заявленная общеплощадочная техника из JSON-описания.

    {"zones": [{"key","name","color"}], "cameras": [{"key","name","overview","tiles","default_zone",
      "zones": {"Z1": [[x,y],...]}}], "site_equipment": [{"class","name","zones":[...]}]}
    """
    for i, zc in enumerate(cfg.get("zones", [])):
        z = zone_by_key(s, project, zc["key"])
        if z is None:
            z = M.Zone(project_id=project.id, key=zc["key"], name=zc["name"],
                       color=zc.get("color", ZONE_COLORS[i % len(ZONE_COLORS)]))
            s.add(z)
        else:
            z.name, z.color = zc["name"], zc.get("color", z.color)
    s.flush()
    s.refresh(project)
    zmap = {z.key: z for z in project.zones}
    for cc in cfg.get("cameras", []):
        cam = s.scalar(select(M.Camera).where(M.Camera.project_id == project.id, M.Camera.key == cc["key"]))
        if cam is None:
            cam = M.Camera(project_id=project.id, key=cc["key"], name=cc.get("name", cc["key"]))
            s.add(cam)
        cam.name = cc.get("name", cam.name)
        cam.source_url = cc.get("url", "")
        cam.is_overview = bool(cc.get("overview", False))
        cam.tiles = int(cc.get("tiles", 0))
        cam.note = cc.get("note", "")
        dz = cc.get("default_zone")
        cam.default_zone_id = zmap[dz].id if dz in zmap else None
        s.flush()
        s.execute(delete(M.CameraZone).where(M.CameraZone.camera_id == cam.id))
        for zk, poly in (cc.get("zones") or {}).items():
            if zk in zmap:
                s.add(M.CameraZone(camera_id=cam.id, zone_id=zmap[zk].id, polygon=poly))
    for se in cfg.get("site_equipment", []):
        s.add(M.SiteEquipment(project_id=project.id, equipment_class=se["class"], name=se.get("name", se["class"]),
                              zone_keys=se.get("zones", []),
                              active_from=dt.date.fromisoformat(se["from"]) if se.get("from") else None,
                              active_to=dt.date.fromisoformat(se["to"]) if se.get("to") else None))
    s.flush()
    s.refresh(project)


def camera_by_key(s: Session, project: M.Project, key: str) -> M.Camera:
    cam = s.scalar(select(M.Camera).where(M.Camera.project_id == project.id, M.Camera.key == key))
    if cam is None:
        raise NotFound(f"камера {key} не найдена")
    return cam


# ============================================================ календарный график

def import_schedule(s: Session, project: M.Project, filename: str, data: bytes) -> dict:
    m = get_methodology()
    res = parse_schedule(data, filename, m, project.object_type)
    last = s.scalar(select(func.max(M.ScheduleVersion.version)).where(M.ScheduleVersion.project_id == project.id))
    ver = M.ScheduleVersion(project_id=project.id, version=(last or 0) + 1, source_filename=filename,
                            is_baseline=last is None, rows_total=len(res.tasks),
                            rows_matched=sum(1 for t in res.tasks if t.work_type_id))
    s.add(ver)
    s.flush()
    row_to_task: dict[int, M.ScheduleTask] = {}
    for t in res.tasks:
        zone = ensure_zone(s, project, t.zone) if t.zone else None
        wt = m.work_types.get(t.work_type_id) if t.work_type_id else None
        planned = {k: v for k, v in t.planned.items() if k in m.classes}
        task = M.ScheduleTask(schedule_version_id=ver.id, project_id=project.id, row_no=t.row_no, wbs=t.wbs,
                              name=t.name, work_type_id=t.work_type_id, match_method=t.match_method,
                              match_score=t.match_score, profile_id=wt.profile if wt else None,
                              zone_id=zone.id if zone else None, start_date=t.start, end_date=t.end,
                              is_summary=t.is_summary, planned_equipment=planned, contractor=t.contractor)
        s.add(task)
        s.flush()
        row_to_task[t.row_no] = task
    for t in res.tasks:
        if t.parent_row and t.parent_row in row_to_task:
            row_to_task[t.row_no].parent_task_id = row_to_task[t.parent_row].id
    s.flush()
    return {"version": ver.version, "sheet": res.sheet, "columns": res.columns, "rows": len(res.tasks),
            "matched": ver.rows_matched, "warnings": res.warnings}


def current_schedule_version(s: Session, project_id: int) -> M.ScheduleVersion | None:
    return s.scalar(select(M.ScheduleVersion).where(M.ScheduleVersion.project_id == project_id)
                    .order_by(M.ScheduleVersion.version.desc()).limit(1))


def schedule_tasks(s: Session, project_id: int) -> list[M.ScheduleTask]:
    ver = current_schedule_version(s, project_id)
    if ver is None:
        return []
    return list(s.scalars(select(M.ScheduleTask).where(M.ScheduleTask.schedule_version_id == ver.id)
                          .order_by(M.ScheduleTask.row_no)))


def task_infos(s: Session, project: M.Project, m: Methodology) -> list[TaskInfo]:
    zkeys = {z.id: z.key for z in project.zones}
    out = []
    for t in schedule_tasks(s, project.id):
        wt = m.work_types.get(t.work_type_id) if t.work_type_id else None
        prof = m.profiles.get(t.profile_id) if t.profile_id else None
        out.append(TaskInfo(id=t.id, wbs=t.wbs, name=t.name, work_type_id=t.work_type_id, profile=prof,
                            observability=wt.observability if wt else (prof.observability if prof else "none"),
                            zone_key=zkeys.get(t.zone_id), start=t.start_date, end=t.end_date,
                            is_summary=t.is_summary, planned=dict(t.planned_equipment or {})))
    return out


def set_task_work_type(s: Session, task: M.ScheduleTask, work_type_id: str | None, zone_key: str | None = None) -> None:
    m = get_methodology()
    if work_type_id is not None:
        if work_type_id and work_type_id not in m.work_types:
            raise NotFound(f"вид работ {work_type_id} не найден в справочнике")
        task.work_type_id = work_type_id or None
        task.profile_id = m.work_types[work_type_id].profile if work_type_id else None
        task.match_method, task.match_score = ("manual", 1.0) if work_type_id else ("none", 0.0)
    if zone_key is not None:
        project = s.get(M.Project, task.project_id)
        task.zone_id = zone_by_key(s, project, zone_key).id if zone_key else None
    s.flush()


# ============================================================ приём снимка

def _camera_zone_polys(cam: M.Camera, zkeys: dict[int, str]) -> list[tuple[str, list]]:
    return [(zkeys[cz.zone_id], cz.polygon) for cz in cam.zone_polygons if cz.zone_id in zkeys]


def camera_zone_keys(cam: M.Camera, zkeys: dict[int, str]) -> list[str]:
    keys = [zkeys[cz.zone_id] for cz in cam.zone_polygons if cz.zone_id in zkeys]
    if cam.default_zone_id and zkeys.get(cam.default_zone_id) and zkeys[cam.default_zone_id] not in keys:
        keys.append(zkeys[cam.default_zone_id])
    return keys


def ingest_snapshot(s: Session, project: M.Project, cam: M.Camera, data: bytes, filename: str,
                    taken_at: dt.datetime | None = None, recompute: bool = True,
                    detector=None) -> tuple[M.Snapshot, dict]:
    """Принять снимок: сохранить, распознать, привязать к зонам, пересчитать отклонения.

    detector — подмена детектора (демо-проекты всегда загружаются с эталонной разметкой).
    """
    detector = detector or get_detector()
    t0 = time.perf_counter()
    m = get_methodology()
    digest = sha256(data)
    existing = s.scalar(select(M.Snapshot).where(M.Snapshot.project_id == project.id, M.Snapshot.sha256 == digest))
    if existing:
        return existing, {"duplicate": True, "processing_ms": 0}
    img = load_image(data)
    when, source = resolve_time(data, filename, taken_at)
    q = assess_quality(img, m.config.get("quality", {}))
    timings = {"quality_ms": round((time.perf_counter() - t0) * 1000)}

    det_result = None
    if q.ok:
        t1 = time.perf_counter()
        det_result = detector.detect(img, tiles=cam.tiles if cam.is_overview else 0, sha256=digest)
        timings["detect_ms"] = round((time.perf_counter() - t1) * 1000)

    day_dir = settings.snapshots_dir / str(project.id) / cam.key / when.strftime("%Y-%m-%d")
    day_dir.mkdir(parents=True, exist_ok=True)
    path = day_dir / f"{when:%H%M%S}_{digest[:10]}.jpg"
    img.save(path, "JPEG", quality=90)

    quality_ok, quality_reason = q.ok, q.reason
    if det_result is not None and not det_result.recognized:
        # детектор не подключён (демо-детектор не знает снимок): «нет данных», а не «работы не ведутся»
        quality_ok, quality_reason = False, "детектор не подключён: снимка нет в демо-разметке"
    snap = M.Snapshot(project_id=project.id, camera_id=cam.id, taken_at=when, time_source=source,
                      original_name=Path(filename).name, file_path=str(path), sha256=digest, width=img.width,
                      height=img.height, brightness=q.brightness, contrast=q.contrast, sharpness=q.sharpness,
                      quality_ok=quality_ok, quality_reason=quality_reason,
                      detector=det_result.model if det_result else detector.name,
                      assessment=det_result.assessment if det_result else None)
    s.add(snap)
    s.flush()
    zkeys = {z.id: z.key for z in project.zones}
    zids = {z.key: z.id for z in project.zones}
    polys = _camera_zone_polys(cam, zkeys)
    default = zkeys.get(cam.default_zone_id) if cam.default_zone_id else None
    for d in (det_result.dets if det_result else []):
        zk = None if d.cls in m.site_wide else assign_zone(d.box, img.width, img.height, polys, default)
        s.add(M.Detection(snapshot_id=snap.id, equipment_class=d.cls, confidence=round(d.conf, 3),
                          x1=d.box[0], y1=d.box[1], x2=d.box[2], y2=d.box[3], zone_id=zids.get(zk) if zk else None,
                          source=d.source))
    s.flush()
    s.refresh(snap)
    t2 = time.perf_counter()
    if recompute:
        recompute_snapshot(s, project, snap)
        recompute_day(s, project, when.date())
    timings["rules_ms"] = round((time.perf_counter() - t2) * 1000)
    snap.processing_ms = round((time.perf_counter() - t0) * 1000)
    s.flush()
    info = {"duplicate": False, "processing_ms": snap.processing_ms, **timings,
            "detector_note": det_result.note if det_result else "детекция пропущена: снимок не прошёл контроль качества"}
    return snap, info


# ============================================================ пересчёт

def _ctx(m: Methodology) -> EngineContext:
    return EngineContext(m, frozenset(get_detector().classes))


def _site_equipment(s: Session, project: M.Project, day: dt.date, zone_key: str) -> set[str]:
    out = set()
    for se in s.scalars(select(M.SiteEquipment).where(M.SiteEquipment.project_id == project.id)):
        if se.active_from and day < se.active_from or se.active_to and day > se.active_to:
            continue
        if not se.zone_keys or zone_key in se.zone_keys:
            out.add(se.equipment_class)
    return out


def _obs_for(snaps: list[M.Snapshot], zone_id: int | None, site_wide: set[str]) -> list[Obs]:
    """Наблюдения зоны: рамки этой зоны на снимках камер, которые её видят; zone_id=None — вся площадка.

    Общеплощадочная техника (башенный кран) не привязана к зоне и попадает в наблюдения всех зон
    своей камеры: кран обслуживает всю площадку, но «рабочей» зону сам по себе не делает (engine).
    """
    out = []
    for sn in snaps:
        dets = [DetInfo(d.id, d.equipment_class, d.confidence) for d in sn.detections
                if not d.is_rejected and (zone_id is None or d.zone_id == zone_id
                                          or (d.zone_id is None and d.equipment_class in site_wide))]
        out.append(Obs(snapshot_ids=[sn.id], taken_at=sn.taken_at, valid=sn.quality_ok, reason=sn.quality_reason,
                       dets=dets))
    return out


def _windows_for_day(s: Session, project: M.Project, day: dt.date, snaps: list[M.Snapshot],
                     m: Methodology) -> list[ZoneWindow]:
    tasks = task_infos(s, project, m)
    win = m.config.get("windows", {})
    horizon, lookback = int(win.get("early_start_horizon_days", 21)), int(win.get("overrun_lookback_days", 90))
    zkeys = {z.id: z.key for z in project.zones}
    cams = {c.id: c for c in project.cameras}
    site_a, site_u, site_r = tasks_for_zone(tasks, SITE_ZONE, day, horizon, lookback)
    windows = []
    for z in project.zones:
        zsnaps = [sn for sn in snaps if z.key in camera_zone_keys(cams[sn.camera_id], zkeys)]
        if not zsnaps:
            continue
        a, u, r = tasks_for_zone(tasks, z.key, day, horizon, lookback)
        windows.append(ZoneWindow(zone_key=z.key, zone_name=z.name, day=day, obs=_obs_for(zsnaps, z.id, m.site_wide), active=a,
                                  upcoming=u + site_u, recent=r + site_r,
                                  site_equipment=_site_equipment(s, project, day, z.key), site_active=site_a))
    if any(not t.is_summary for t in site_a) and snaps:      # сводные этапы без зоны не проверяются
        windows.append(ZoneWindow(zone_key=SITE_ZONE, zone_name="Вся площадка", day=day, obs=_obs_for(snaps, None, m.site_wide),
                                  active=site_a, site_equipment=_site_equipment(s, project, day, SITE_ZONE),
                                  check_unexpected=False))
    return windows


def _stage_mismatch(project: M.Project, day: dt.date, snaps: list[M.Snapshot], tasks: list[TaskInfo],
                    m: Methodology) -> list[Finding]:
    """Сверка стадии по кадру общего плана (модель готовности ML-части) со стадией графика."""
    cams = {c.id: c for c in project.cameras}
    assessed = [sn for sn in snaps if sn.assessment and sn.quality_ok and cams[sn.camera_id].is_overview]
    if not assessed:
        return []
    sn = max(assessed, key=lambda x: x.taken_at)
    obs_stage = sn.assessment.get("stage")
    active_stages = [m.work_types[t.work_type_id].stage for t in tasks
                     if t.start <= day <= t.end and not t.is_summary and t.work_type_id
                     and m.work_types[t.work_type_id].stage]
    if not obs_stage or not active_stages:
        return []
    plan_stage = min(active_stages, key=lambda x: STAGE_ORDER[x])
    if STAGE_ORDER.get(obs_stage, 0) >= STAGE_ORDER[plan_stage]:
        return []
    r = m.rules["STAGE_MISMATCH"]
    return [Finding(rule="STAGE_MISMATCH", zone_key=SITE_ZONE, severity=r.severity, status="preliminary",
                    title=r.title, message=r.message.format(observed_stage=obs_stage, planned_stage=plan_stage),
                    recommendation=r.recommendation, metrics={"assessment": sn.assessment, "planned_stage": plan_stage},
                    evidence=[(sn.id, [], "кадр общего плана")], first_seen=sn.taken_at, last_seen=sn.taken_at,
                    snapshots_count=1)]


def recompute_day(s: Session, project: M.Project, day: dt.date) -> list[M.Deviation]:
    """Пересчитать отклонения проекта за день по всем снимкам дня (идемпотентно)."""
    if s.get_bind().dialect.name == "postgresql":
        # загрузка снимков и фоновые задачи ИИ пересчитывают один и тот же день — по очереди, до конца транзакции
        s.execute(text("SELECT pg_advisory_xact_lock(:k)"), {"k": 0x4F4B4F00000 + project.id})
    m = get_methodology()
    ctx = _ctx(m)
    start = dt.datetime.combine(day, dt.time.min)
    snaps = list(s.scalars(select(M.Snapshot).where(M.Snapshot.project_id == project.id,
                                                    M.Snapshot.taken_at >= start,
                                                    M.Snapshot.taken_at < start + dt.timedelta(days=1))
                           .order_by(M.Snapshot.taken_at)))
    findings: list[Finding] = []
    for zw in _windows_for_day(s, project, day, snaps, m):
        f, _ = evaluate_window(zw, ctx)
        findings.extend(f)
    findings.extend(_stage_mismatch(project, day, snaps, task_infos(s, project, m), m))
    return _upsert_deviations(s, project, day, findings, m)


def _upsert_deviations(s: Session, project: M.Project, day: dt.date, findings: list[Finding],
                       m: Methodology) -> list[M.Deviation]:
    zids = {z.key: z.id for z in project.zones}
    existing = {d.key: d for d in s.scalars(select(M.Deviation).where(M.Deviation.project_id == project.id,
                                                                      M.Deviation.day == day))}
    seen = set()
    result = []
    for f in findings:
        key = f.key(day)
        if key in seen:
            continue
        seen.add(key)
        d = existing.get(key)
        if d is None:
            d = M.Deviation(project_id=project.id, key=key, day=day, rule_code=f.rule, rules_version=m.rules_version)
            s.add(d)
        d.zone_id = zids.get(f.zone_key)
        d.task_id = f.task_id
        d.equipment_class = f.cls
        d.severity, d.status, d.title = f.severity, f.status, f.title
        d.message, d.recommendation = f.message, f.recommendation
        d.metrics = {**f.metrics, "zone_key": f.zone_key, "role": f.role}
        d.first_seen, d.last_seen = f.first_seen, f.last_seen
        d.snapshots_count = f.snapshots_count
        d.rules_version = m.rules_version
        s.flush()
        s.execute(delete(M.DeviationEvidence).where(M.DeviationEvidence.deviation_id == d.id))
        added = set()
        for sid, det_ids, note in f.evidence:
            if sid in added:
                continue
            added.add(sid)
            s.add(M.DeviationEvidence(deviation_id=d.id, snapshot_id=sid, detection_ids=list(det_ids), note=note))
        result.append(d)
    # отклонения, которых больше нет (снимков стало больше, данные исправлены), удаляем,
    # если инженер их ещё не рассматривал
    for key, d in existing.items():
        if key not in seen and d.review_status is None:
            s.delete(d)
    s.flush()
    return result


def recompute_snapshot(s: Session, project: M.Project, snap: M.Snapshot) -> list[dict]:
    """Объяснение по одному снимку: как он сопоставлен с этапами в каждой зоне камеры."""
    m = get_methodology()
    ctx = _ctx(m)
    day = snap.taken_at.date()
    windows = _windows_for_day(s, project, day, [snap], m)
    s.execute(delete(M.SnapshotCheck).where(M.SnapshotCheck.snapshot_id == snap.id))
    out = []
    for zw in windows:
        findings, check = evaluate_window(zw, ctx)
        check["findings"] = [{"rule": f.rule, "title": f.title, "severity": f.severity, "message": f.message,
                              "cls": f.cls, "role": f.role, "task_id": f.task_id} for f in findings]
        s.add(M.SnapshotCheck(snapshot_id=snap.id, zone_key=zw.zone_key, payload=check))
        out.append(check)
    s.flush()
    return out


def recompute_project(s: Session, project: M.Project) -> int:
    """Полный пересчёт (после загрузки нового графика или изменения зон)."""
    days = sorted({d.date() for d in s.scalars(select(M.Snapshot.taken_at).where(M.Snapshot.project_id == project.id))})
    zkeys = {z.id: z.key for z in project.zones}
    polys = {c.id: (_camera_zone_polys(c, zkeys), zkeys.get(c.default_zone_id)) for c in project.cameras}
    zids = {z.key: z.id for z in project.zones}
    m = get_methodology()
    for snap in s.scalars(select(M.Snapshot).where(M.Snapshot.project_id == project.id)):
        pz, default = polys[snap.camera_id]
        for d in snap.detections:
            zk = None if d.equipment_class in m.site_wide else assign_zone((d.x1, d.y1, d.x2, d.y2), snap.width,
                                                                           snap.height, pz, default)
            d.zone_id = zids.get(zk) if zk else None
        s.flush()
        recompute_snapshot(s, project, snap)
    for day in days:
        recompute_day(s, project, day)
    return len(days)


# ============================================================ проверка инженером и нарушения

def review_deviation(s: Session, dev: M.Deviation, verdict: str, comment: str = "", reviewer: str = "инженер",
                     contractor: str = "", due_days: int = 3) -> M.Deviation:
    if verdict not in ("accepted", "rejected", "force_majeure"):
        raise ValueError("verdict: accepted | rejected | force_majeure")
    s.add(M.DeviationReview(deviation_id=dev.id, verdict=verdict, comment=comment, reviewer=reviewer))
    dev.review_status = verdict
    vio = s.scalar(select(M.Violation).where(M.Violation.deviation_id == dev.id))
    if verdict == "accepted":
        if vio is None:
            n = (s.scalar(select(func.count(M.Violation.id)).where(M.Violation.project_id == dev.project_id)) or 0) + 1
            task = s.get(M.ScheduleTask, dev.task_id) if dev.task_id else None
            vio = M.Violation(project_id=dev.project_id, deviation_id=dev.id, number=f"НР-{dev.day:%Y}-{n:04d}",
                              title=dev.title, description=dev.message, zone_id=dev.zone_id, task_id=dev.task_id,
                              contractor=contractor or (task.contractor if task else ""), severity=dev.severity,
                              due_date=dev.day + dt.timedelta(days=due_days))
            s.add(vio)
        vio.status = "open"
    elif vio is not None:
        vio.status, vio.closed_at = "closed", M.utcnow()
    if verdict == "rejected":
        # ложное срабатывание: рамки-доказательства помечаются, кадры уходят в датасет дообучения
        for ev in dev.evidence:
            for d in s.scalars(select(M.Detection).where(M.Detection.id.in_(ev.detection_ids or [-1]))):
                if dev.rule_code in ("UNEXPECTED_EQUIPMENT", "NO_ACTIVE_STAGE", "EARLY_START", "STAGE_OVERRUN"):
                    d.is_rejected = True
    s.flush()
    return dev


# ============================================================ представления для API

def box_view(d: M.Detection, m: Methodology, zkeys: dict[int, str]) -> dict:
    return {"id": d.id, "cls": d.equipment_class, "label": m.class_name(d.equipment_class), "conf": d.confidence,
            "xyxy": [round(d.x1, 1), round(d.y1, 1), round(d.x2, 1), round(d.y2, 1)],
            "zone": zkeys.get(d.zone_id) if d.zone_id else None, "source": d.source, "rejected": d.is_rejected,
            "color": m.classes.get(d.equipment_class, {}).get("color", "#888888"),
            "strong": d.confidence >= m.presence_threshold(d.equipment_class)}


def snapshot_brief(sn: M.Snapshot, cam: M.Camera, m: Methodology) -> dict:
    counts: dict[str, int] = defaultdict(int)
    for d in sn.detections:
        if not d.is_rejected and d.confidence >= m.presence_threshold(d.equipment_class):
            counts[d.equipment_class] += 1
    return {"id": sn.id, "camera": cam.key, "camera_name": cam.name, "taken_at": sn.taken_at.isoformat(),
            "time_source": sn.time_source, "quality_ok": sn.quality_ok, "quality_reason": sn.quality_reason,
            "counts": dict(counts), "width": sn.width, "height": sn.height, "detector": sn.detector,
            "processing_ms": sn.processing_ms, "image": f"/api/snapshots/{sn.id}/image",
            "thumb": f"/api/snapshots/{sn.id}/image?w=480"}


def deviation_view(s: Session, d: M.Deviation, m: Methodology, project: M.Project | None = None,
                   with_evidence: bool = True) -> dict:
    project = project or s.get(M.Project, d.project_id)
    zones = {z.id: z for z in project.zones}
    task = s.get(M.ScheduleTask, d.task_id) if d.task_id else None
    z = zones.get(d.zone_id)
    out = {"id": d.id, "day": d.day.isoformat(), "rule": d.rule_code, "title": d.title, "severity": d.severity,
           "status": d.status, "review_status": d.review_status, "message": d.message,
           "recommendation": d.recommendation, "metrics": d.metrics, "equipment_class": d.equipment_class,
           "equipment_name": m.class_name(d.equipment_class) if d.equipment_class else None,
           "zone": {"key": z.key, "name": z.name, "color": z.color} if z else
                   {"key": SITE_ZONE, "name": "Вся площадка", "color": "#687480"},
           "task": {"id": task.id, "wbs": task.wbs, "name": task.name, "work_type_id": task.work_type_id,
                    "start": task.start_date.isoformat(), "end": task.end_date.isoformat()} if task else None,
           "first_seen": d.first_seen.isoformat() if d.first_seen else None,
           "last_seen": d.last_seen.isoformat() if d.last_seen else None,
           "snapshots_count": d.snapshots_count, "rules_version": d.rules_version}
    vio = s.scalar(select(M.Violation).where(M.Violation.deviation_id == d.id))
    if vio:
        out["violation"] = {"number": vio.number, "status": vio.status,
                            "due_date": vio.due_date.isoformat() if vio.due_date else None}
    if with_evidence:
        ev = []
        for e in d.evidence:
            sn = s.get(M.Snapshot, e.snapshot_id)
            if sn is None:
                continue
            cam = s.get(M.Camera, sn.camera_id)
            ev.append({"snapshot_id": sn.id, "camera": cam.key, "taken_at": sn.taken_at.isoformat(),
                       "detection_ids": e.detection_ids, "note": e.note, "quality_ok": sn.quality_ok,
                       "thumb": f"/api/snapshots/{sn.id}/image?w=480&annotate=1&highlight="
                                + ",".join(map(str, e.detection_ids or [])),
                       "image": f"/api/snapshots/{sn.id}/image"})
        out["evidence"] = ev
    return out


def project_days(s: Session, project_id: int) -> list[dict]:
    rows = s.execute(select(M.Snapshot.taken_at).where(M.Snapshot.project_id == project_id)).scalars().all()
    by_day: dict[dt.date, int] = defaultdict(int)
    for t in rows:
        by_day[t.date()] += 1
    devs = s.execute(select(M.Deviation.day, M.Deviation.severity).where(M.Deviation.project_id == project_id)).all()
    sev: dict[dt.date, dict] = defaultdict(lambda: defaultdict(int))
    for day, sv in devs:
        sev[day][sv] += 1
    return [{"day": d.isoformat(), "snapshots": n, "deviations": dict(sev[d])} for d, n in sorted(by_day.items())]


def project_summary(s: Session, project: M.Project, day: dt.date) -> dict:
    """Сводка на день: KPI, доска зон (этапы, ожидаемая и видимая техника, статус), лента отклонений."""
    m = get_methodology()
    zkeys = {z.id: z.key for z in project.zones}
    cams = {c.id: c for c in project.cameras}
    start = dt.datetime.combine(day, dt.time.min)
    snaps = list(s.scalars(select(M.Snapshot).where(M.Snapshot.project_id == project.id, M.Snapshot.taken_at >= start,
                                                    M.Snapshot.taken_at < start + dt.timedelta(days=1))
                           .order_by(M.Snapshot.taken_at)))
    devs = list(s.scalars(select(M.Deviation).where(M.Deviation.project_id == project.id, M.Deviation.day == day)))
    tasks = task_infos(s, project, m)
    # проверки зон за день — тот же расчёт, что и у отклонений (engine.evaluate_window)
    ctx = _ctx(m)
    day_checks = {zw.zone_key: evaluate_window(zw, ctx)[1] for zw in _windows_for_day(s, project, day, snaps, m)}
    board = []
    for z in project.zones:
        active = [t for t in tasks if t.zone_key == z.key and t.start <= day <= t.end]
        zsnaps = [sn for sn in snaps if z.key in camera_zone_keys(cams[sn.camera_id], zkeys)]
        valid = [sn for sn in zsnaps if sn.quality_ok]
        last = (valid or zsnaps)[-1] if zsnaps else None
        seen: dict[str, int] = defaultdict(int)
        if last:
            for d in last.detections:
                in_zone = d.zone_id == z.id or (d.zone_id is None and d.equipment_class in m.site_wide)
                if in_zone and not d.is_rejected and d.confidence >= m.presence_threshold(d.equipment_class):
                    seen[d.equipment_class] += 1
        zc = day_checks.get(z.key, {})
        group_state = {t_["task_id"]: t_ for t_ in zc.get("tasks", [])}
        zd = [d for d in devs if d.zone_id == z.id and d.review_status != "rejected"]
        worst = min((d.severity for d in zd), key=lambda x: {"critical": 0, "warning": 1, "info": 2}[x], default=None)
        board.append({
            "zone": {"key": z.key, "name": z.name, "color": z.color},
            "cameras": sorted({cams[sn.camera_id].key for sn in zsnaps}) or
                       [c.key for c in project.cameras if z.key in camera_zone_keys(c, zkeys)],
            "tasks": [{"id": t.id, "wbs": t.wbs, "name": t.name, "work_type_id": t.work_type_id,
                       "profile": t.profile.name if t.profile else None, "observability": t.observability,
                       "summary": t.is_summary,
                       "expected": [{"role": g.role, "classes": list(g.any_of), "window": g.window}
                                    for g in (t.profile.required if t.profile else ())],
                       "day_check": group_state.get(t.id),
                       "planned": t.planned, "end": t.end.isoformat()} for t in active],
            "last_snapshot": snapshot_brief(last, cams[last.camera_id], m) if last else None,
            "seen": dict(seen), "seen_day": zc.get("observed", {}), "snapshots": len(zsnaps),
            "valid": sum(1 for sn in zsnaps if sn.quality_ok),
            "deviations": len(zd), "worst": worst,
            "status": ("no_snapshots" if not zsnaps else "no_data" if not any(sn.quality_ok for sn in zsnaps)
                       else worst or ("ok" if active else "no_tasks")),
        })
    active_devs = [d for d in devs if d.review_status != "rejected"]
    kpi = {"snapshots": len(snaps), "valid": sum(1 for sn in snaps if sn.quality_ok),
           "cameras": len({sn.camera_id for sn in snaps}),
           "critical": sum(1 for d in active_devs if d.severity == "critical"),
           "warning": sum(1 for d in active_devs if d.severity == "warning"),
           "info": sum(1 for d in active_devs if d.severity == "info"),
           "violations": s.scalar(select(func.count(M.Violation.id)).where(M.Violation.project_id == project.id,
                                                                           M.Violation.status == "open")) or 0,
           "avg_processing_ms": round(sum(sn.processing_ms for sn in snaps) / len(snaps)) if snaps else 0,
           "active_tasks": sum(1 for t in tasks if t.start <= day <= t.end and not t.is_summary)}
    order = {"critical": 0, "warning": 1, "info": 2}
    feed = sorted(active_devs, key=lambda d: (order[d.severity], d.status != "confirmed", d.last_seen or start))
    return {"day": day.isoformat(), "kpi": kpi, "board": board,
            "deviations": [deviation_view(s, d, m, project) for d in feed]}
