"""Сбор выводов всех моделей и правил в один JSON — вход для LLM и запись для аудита (ai_reviews.context).

Снимок: качество кадра, рамки детектора (координаты в долях кадра, зона, выше/ниже порога), зоны камеры,
этапы графика по зонам с профилями техники и итогами проверок, отклонения правил, стадия по модели готовности
и по графику, описание локальной VLM, каталог классов и классы, которые распознаёт детектор.
День: доска зон, отклонения, итоги ИИ-анализа снимков, стадия по кадрам общего плана, история и ближайшие этапы.
"""
from __future__ import annotations

import datetime as dt
import re
from pathlib import Path

from sqlalchemy import select
from sqlalchemy.orm import Session

from .. import models as M
from .. import services as S
from ..detection import get_detector
from ..imaging import load_image, render_annotated
from ..methodology import Methodology, get_methodology

VISION_INTRO = """Проверь снимок камеры стройплощадки.
Изображение 1 — исходный кадр. Изображение 2 — тот же кадр с разметкой: цветные полигоны — зоны площадки,
рамки — находки детектора RF-DETR с номерами #id (пунктир — уверенность ниже порога, в правила не идёт)."""

TEXT_INTRO = """Проверь снимок камеры стройплощадки. Самого изображения у тебя нет: о кадре суди по независимым слоям —
рамкам детектора RF-DETR (detections), описанию и рамкам локальной VLM (vlm), их сверке (cross_check:
agree — совпали, class_conflict — одна рамка, разные классы, vlm_only — VLM видит технику, которой нет у детектора,
detector_only — рамку детектора VLM не подтверждает), стадии по модели готовности и данным графика."""

SNAPSHOT_TASKS = """
1. detections — вердикт по КАЖДОЙ рамке детектора (по id): confirmed — техника есть и класс верный;
   false_positive — техники нет (контейнер, бытовка, куча грунта, тень, забор); wrong_class — техника есть,
   но класс другой (укажи correct_class из classes); uncertain — не разобрать.
2. missed — техника из каталога classes, которую детектор пропустил: рамка в долях кадра [x1, y1, x2, y2]
   (если опираешься на VLM — бери её рамку из vlm_only) и уверенность. Только то, что подтверждается
   уверенно; рамки детектора не дублируй."""

SNAPSHOT_REST = """
3. stage — стадия объекта на кадре (S1–S5, unknown) и готовность в %; сверь модель готовности
   (stage.readiness_model: стадия, готовность, план по графику и отставание; модель обучена на кадрах общего
   плана — на кадре зоны её оценка ориентировочная),
   VLM (vlm.stage) и график (stage.planned).
4. deviations — вердикт по каждому отклонению правил (по id) с учётом твоих исправлений техники:
   confirmed / doubtful / rejected и почему.
5. new_findings — что правила не увидели, но явно следует из данных: несоответствие графику, технологии,
   безопасность (люди в зоне работы техники и т. п.).
6. forecast — чем ситуация грозит этапам зон (сроки этапов — в schedule[].tasks[].period).
7. recommendations — 1–5 конкретных действий для инженера строительного контроля.

open_vocab — Grounding DINO, поиск по текстовым подсказкам: objects — прочие объекты (рабочие, леса, опалубка,
строящееся здание, котлован, ограждение) с зоной; это признаки работ там, где техники мало (монолит, фасад,
отделка) — учитывай их в вердиктах по отклонениям «работы не ведутся» / «нет техники» и в стадии. only_open_vocab —
техника, которую нашёл только Grounding DINO: в missed бери её, если подтверждают другие слои (VLM, изображение);
class_conflict — рамка детектора, которую Grounding DINO считает другим классом.
VLM — более слабая локальная модель, она может ошибаться. Детектор распознаёт только классы
detector.classes: если на кадре нет техники других классов, это не отклонение."""


def snapshot_instructions(vision: bool) -> str:
    return (VISION_INTRO if vision else TEXT_INTRO) + SNAPSHOT_TASKS + SNAPSHOT_REST


DAY_INSTRUCTIONS = """Сделай итоговый анализ стройплощадки за день по данным всех слоёв:
board — зоны: идущие этапы графика, требуемая техника и итог проверки, увиденная за день техника;
deviations — отклонения правил (review_status — вердикт инженера); ai_snapshots — итоги ИИ-анализа отдельных
снимков (исправления детектора, новые находки, objects — прочие объекты Grounding DINO: опалубка, леса, рабочие);
stage — стадия и готовность по модели готовности (с планом
по графику) и стадии идущих этапов графика; history — снимки и отклонения по дням; upcoming и finishing —
ближайшие этапы.{images}

Оцени статус площадки, риски срыва сроков (вероятность 0..1, влияние, зона, этап, причина, что сделать),
прогноз задержки по этапам в днях, где модели или правила, вероятно, ошиблись, и рекомендации инженеру.
Опирайся только на данные; если снимков мало или они плохого качества — так и скажи."""


def day_instructions(vision: bool) -> str:
    return DAY_INSTRUCTIONS.format(
        images="\nИзображения — последние снимки камер за день с разметкой детектора." if vision else "")


def _r(x: float, n: int = 3) -> float:
    return round(float(x), n)


def assessment_brief(a: dict | None) -> dict | None:
    """Оценка модели готовности без карты внимания (она для интерфейса, не для LLM)."""
    return {k: v for k, v in a.items() if k not in ("attention", "seconds")} if a else None


def stage_of(value) -> str | None:
    """«S2 фундамент», "s3", {"stage": "S4"} → S2 / S3 / S4."""
    if isinstance(value, dict):
        value = value.get("stage") or value.get("value")
    mt = re.search(r"\bS\s*([1-5])\b", str(value or ""), flags=re.I)
    return f"S{mt.group(1)}" if mt else None


def planned_stages(m: Methodology, tasks, day: dt.date, zone_key: str | None = None) -> list[str]:
    """Стадии идущих этапов графика (по видам работ справочника), от ранней к поздней."""
    out = set()
    for t in tasks:
        if t.is_summary or not (t.start <= day <= t.end) or not t.work_type_id:
            continue
        if zone_key is not None and t.zone_key not in (zone_key, None):
            continue
        st = m.work_types[t.work_type_id].stage if t.work_type_id in m.work_types else None
        if st:
            out.add(st)
    return sorted(out)


def _dets(snap: M.Snapshot, m: Methodology, zkeys: dict[int, str]) -> list[dict]:
    W, H = snap.width or 1, snap.height or 1
    return [{"id": d.id, "cls": d.equipment_class, "name": m.class_name(d.equipment_class),
             "conf": round(d.confidence, 2),
             "box": [_r(d.x1 / W), _r(d.y1 / H), _r(d.x2 / W), _r(d.y2 / H)],
             "zone": zkeys.get(d.zone_id) if d.zone_id else ("вся площадка" if d.equipment_class in m.site_wide else None),
             "above_threshold": d.confidence >= m.presence_threshold(d.equipment_class), "source": d.source}
            for d in snap.detections if not d.is_rejected]


def _dev_brief(s: Session, d: M.Deviation, znames: dict[int, str]) -> dict:
    task = s.get(M.ScheduleTask, d.task_id) if d.task_id else None
    keep = ("planned", "observed_max", "share", "lead", "partner", "expected", "role", "max_conf", "confirmed_by",
            "days_before", "days_after", "task_start", "task_end", "planned_stage")
    return {"id": d.id, "rule": d.rule_code, "title": d.title, "severity": d.severity, "status": d.status,
            "review_status": d.review_status, "zone": znames.get(d.zone_id, "вся площадка"),
            "task": f"{task.wbs} {task.name}".strip() if task else None, "equipment_class": d.equipment_class,
            "message": d.message, "snapshots_count": d.snapshots_count,
            "metrics": {k: v for k, v in (d.metrics or {}).items() if k in keep}}


def snapshot_context(s: Session, snap: M.Snapshot, vlm_output: dict | None = None) -> dict:
    m = get_methodology()
    p = s.get(M.Project, snap.project_id)
    cam = s.get(M.Camera, snap.camera_id)
    det = get_detector()
    zkeys = {z.id: z.key for z in p.zones}
    znames = {z.id: z.name for z in p.zones}
    day = snap.taken_at.date()
    tasks = S.task_infos(s, p, m)
    checks = [c.payload for c in s.scalars(select(M.SnapshotCheck).where(M.SnapshotCheck.snapshot_id == snap.id))]
    dev_ids = set(s.scalars(select(M.DeviationEvidence.deviation_id)
                            .where(M.DeviationEvidence.snapshot_id == snap.id)).all())
    devs = [_dev_brief(s, d, znames) for d in
            s.scalars(select(M.Deviation).where(M.Deviation.id.in_(dev_ids or {-1})).order_by(M.Deviation.id))]
    cam_zones = S.snapshot_zone_keys(snap, cam, zkeys)
    return {
        "project": {"name": p.name, "object_type": p.object_type, "address": p.address},
        "camera": {"key": cam.key, "name": cam.name, "overview": cam.is_overview, "note": cam.note},
        "taken_at": snap.taken_at.isoformat(timespec="minutes"),
        "image": {"width": snap.width, "height": snap.height},
        "quality": {"ok": snap.quality_ok, "reason": snap.quality_reason, "brightness": snap.brightness,
                    "contrast": snap.contrast, "sharpness": snap.sharpness},
        "detector": {"model": snap.detector, "classes": sorted(det.classes),
                     "presence_threshold": {c: m.presence_threshold(c) for c in sorted(det.classes)}},
        "classes": {k: v.get("name", k) for k, v in m.classes.items()},
        "detections": _dets(snap, m, zkeys),
        "rejected_by_engineer": [d.id for d in snap.detections if d.is_rejected],
        "zones": ([{"key": snap.frame_zone, "name": next(z.name for z in p.zones if z.key == snap.frame_zone),
                    "polygon": "весь кадр"}] if snap.frame_zone in zkeys.values() else
                  [{"key": zkeys[cz.zone_id], "name": znames[cz.zone_id],
                    "polygon": [[_r(x, 3), _r(y, 3)] for x, y in cz.polygon]}
                   for cz in cam.zone_polygons if cz.zone_id in zkeys]),
        "schedule": checks,
        "deviations": devs,
        "stage": {"planned": {zk: planned_stages(m, tasks, day, zk) for zk in cam_zones} or
                             {"вся площадка": planned_stages(m, tasks, day)},
                  "readiness_model": assessment_brief(snap.assessment)},
        "vlm": vlm_output if vlm_output is not None else {"available": False},
    }


def day_context(s: Session, p: M.Project, day: dt.date) -> dict:
    m = get_methodology()
    summ = S.project_summary(s, p, day)
    tasks = S.task_infos(s, p, m)
    zname = {z.key: z.name for z in p.zones}
    board = []
    for z in summ["board"]:
        board.append({
            "zone": z["zone"]["name"], "status": z["status"], "cameras": z["cameras"],
            "snapshots": z["snapshots"], "valid": z["valid"], "seen_day": z["seen_day"],
            "tasks": [{"wbs": t["wbs"], "name": t["name"], "profile": t["profile"], "observability": t["observability"],
                       "end": t["end"], "planned": t["planned"],
                       "check": {k: (t["day_check"] or {}).get(k) for k in ("status", "groups", "companions")}
                       if t.get("day_check") else None} for t in z["tasks"]]})
    devs = [{"id": d["id"], "rule": d["rule"], "title": d["title"], "severity": d["severity"], "status": d["status"],
             "review_status": d["review_status"], "zone": d["zone"]["name"],
             "task": d["task"]["name"] if d["task"] else None, "message": d["message"],
             "first_seen": d["first_seen"], "last_seen": d["last_seen"], "snapshots_count": d["snapshots_count"]}
            for d in summ["deviations"]]
    start = dt.datetime.combine(day, dt.time.min)
    snaps = list(s.scalars(select(M.Snapshot).where(M.Snapshot.project_id == p.id, M.Snapshot.taken_at >= start,
                                                    M.Snapshot.taken_at < start + dt.timedelta(days=1))
                           .order_by(M.Snapshot.taken_at)))
    cams = {c.id: c for c in p.cameras}
    reviews = latest_reviews(s, [sn.id for sn in snaps])
    ai_snaps = []
    for sn in snaps:
        r = reviews.get(sn.id)
        if not r or r.status != "done" or not r.final:
            continue
        f = r.final
        ai_snaps.append({"camera": cams[sn.camera_id].key, "time": f"{sn.taken_at:%H:%M}", "summary": f.get("summary"),
                         "stage": (f.get("stage") or {}).get("final"),
                         "corrections": [c["text"] for c in f.get("corrections", [])],
                         "applied": bool(r.applied and not r.applied.get("reverted")),
                         "objects": ((f.get("open_vocab") or {}).get("counts") or {}),
                         "new_findings": [n.get("title") for n in f.get("new_findings", [])]})
    # стадию площадки дают кадры общего плана: на них обучена модель готовности
    assessed = [{"camera": cams[sn.camera_id].key, "time": f"{sn.taken_at:%H:%M}", **assessment_brief(sn.assessment)}
                for sn in snaps if sn.assessment and cams[sn.camera_id].is_overview]

    def task_brief(t) -> dict:
        return {"wbs": t.wbs, "name": t.name, "zone": zname.get(t.zone_key, "вся площадка"),
                "start": t.start.isoformat(), "end": t.end.isoformat(),
                "profile": t.profile.name if t.profile else None, "planned": t.planned}

    real = [t for t in tasks if not t.is_summary]
    history = [h for h in S.project_days(s, p.id) if dt.date.fromisoformat(h["day"]) <= day][-14:]
    return {
        "project": {"name": p.name, "object_type": p.object_type, "address": p.address},
        "day": day.isoformat(), "kpi": summ["kpi"], "board": board, "deviations": devs[:60],
        "ai_snapshots": ai_snaps[-40:],            # YandexGPT: контекст 32k токенов — берём последние
        "stage": {"planned": planned_stages(m, tasks, day), "readiness_model": assessed},
        "history": history,
        "upcoming": [task_brief(t) for t in sorted(real, key=lambda t: t.start)
                     if day < t.start <= day + dt.timedelta(days=14)][:15],
        "finishing": [task_brief(t) for t in sorted(real, key=lambda t: t.end)
                      if day <= t.end <= day + dt.timedelta(days=7) and t.start <= day][:15],
    }


def latest_reviews(s: Session, snapshot_ids: list[int]) -> dict[int, M.AIReview]:
    out: dict[int, M.AIReview] = {}
    if not snapshot_ids:
        return out
    for r in s.scalars(select(M.AIReview).where(M.AIReview.snapshot_id.in_(snapshot_ids))
                       .order_by(M.AIReview.id)):
        out[r.snapshot_id] = r
    return out


# ------------------------------------------------------------------ изображения для LLM

def _jpeg(img, width: int) -> bytes:
    import io
    im = img.convert("RGB")
    im.thumbnail((width, width))
    buf = io.BytesIO()
    im.save(buf, "JPEG", quality=85)
    return buf.getvalue()


def llm_images(s: Session, snap: M.Snapshot, width: int = 1280) -> list[bytes]:
    """Исходный кадр и кадр с зонами и рамками «#id класс» — чтобы LLM ссылалась на рамки по номерам."""
    m = get_methodology()
    p = s.get(M.Project, snap.project_id)
    cam = s.get(M.Camera, snap.camera_id)
    img = load_image(Path(snap.file_path).read_bytes())
    zmeta = {z.id: z for z in p.zones}
    boxes = [{"id": d.id, "label": f"#{d.id} {m.class_name(d.equipment_class)}", "conf": d.confidence,
              "color": m.classes.get(d.equipment_class, {}).get("color", "#E1A21C"),
              "x1": d.x1, "y1": d.y1, "x2": d.x2, "y2": d.y2,
              "dashed": d.confidence < m.presence_threshold(d.equipment_class)}
             for d in snap.detections if not d.is_rejected]
    zones = [] if snap.frame_zone else [{"name": zmeta[cz.zone_id].name, "color": zmeta[cz.zone_id].color,
                                         "polygon": cz.polygon} for cz in cam.zone_polygons if cz.zone_id in zmeta]
    return frame_images(img, boxes, zones, width)


def day_images(s: Session, p: M.Project, day: dt.date, limit: int = 4, width: int = 1024) -> list[bytes]:
    """Последний годный снимок каждой камеры за день с разметкой (не больше limit камер)."""
    start = dt.datetime.combine(day, dt.time.min)
    last: dict[int, M.Snapshot] = {}
    for sn in s.scalars(select(M.Snapshot).where(M.Snapshot.project_id == p.id, M.Snapshot.taken_at >= start,
                                                 M.Snapshot.taken_at < start + dt.timedelta(days=1),
                                                 M.Snapshot.quality_ok.is_(True)).order_by(M.Snapshot.taken_at)):
        last[sn.camera_id] = sn
    out = []
    for sn in list(last.values())[:limit]:
        if Path(sn.file_path).exists():
            out.append(llm_images(s, sn, width)[1])
    return out


# ------------------------------------------------------------------ быстрая проверка (вкладка «Проверить снимок»)

def quick_context(res: dict, work_type_ids: list[str], filename: str) -> dict:
    """Контекст для снимка без проекта: результат POST /api/analyze (рамки, проверки методики по выбранным видам
    работ, предупреждения). Поля — как у snapshot_context, чтобы сведение слоёв и интерфейс были общими."""
    m = get_methodology()
    det = get_detector()
    W, H = res["width"] or 1, res["height"] or 1
    stages = sorted({m.work_types[w].stage for w in work_type_ids if w in m.work_types and m.work_types[w].stage})
    return {
        "project": None,
        "camera": {"key": "снимок", "name": filename or "загруженный снимок", "overview": None,
                   "note": "быстрая проверка: снимок без проекта, этапы выбраны вручную"},
        "taken_at": res["taken_at"][:16],
        "image": {"width": res["width"], "height": res["height"]},
        "quality": res["quality"],
        "detector": {"model": res["detector"], "classes": sorted(det.classes),
                     "presence_threshold": {c: m.presence_threshold(c) for c in sorted(det.classes)}},
        "classes": {k: v.get("name", k) for k, v in m.classes.items()},
        "detections": [{"id": b["id"], "cls": b["cls"], "name": b["label"], "conf": round(b["conf"], 2),
                        "box": [_r(b["xyxy"][0] / W), _r(b["xyxy"][1] / H), _r(b["xyxy"][2] / W), _r(b["xyxy"][3] / H)],
                        "zone": "кадр", "above_threshold": b["strong"], "source": "model"} for b in res["boxes"]],
        "rejected_by_engineer": [],
        "zones": [],
        "schedule": [res["check"]],
        "deviations": [{"id": i, "rule": f["rule"], "title": f["title"], "severity": f["severity"],
                        "status": f["status"], "review_status": None, "zone": "кадр", "task": None,
                        "equipment_class": f.get("cls"), "message": f["message"], "snapshots_count": 1, "metrics": {}}
                       for i, f in enumerate(res["findings"], 1)],
        "stage": {"planned": {"кадр": stages}, "readiness_model": None},
        "vlm": {"available": False},
    }


def frame_images(img, boxes: list[dict], zones: list[dict] | None = None, width: int = 1280) -> list[bytes]:
    """Исходный кадр и кадр с рамками «#id класс» (boxes — как для render_annotated)."""
    return [_jpeg(img, width), render_annotated(img, boxes, zones, width=width)]
