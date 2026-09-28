"""Консольное приложение ОКО (ТЗ допускает веб-интерфейс, консоль или API).

Примеры (из папки backend/):
  python -m app.cli check ../data/demo/housing/snapshots/CAM-01_2026-09-24_10-30.jpg --work-type 12.3.1
  python -m app.cli rules 12.3.7-1
  python -m app.cli seed-demo
  python -m app.cli ingest --project 1 --camera CAM-01 ./снимки/
  python -m app.cli report --project 1 --day 2026-09-24 --csv отчёт.csv
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from pathlib import Path

SEV = {"critical": "КРИТИЧНО", "warning": "ВНИМАНИЕ", "info": "к сведению"}
STATUS = {"preliminary": "предварительно", "confirmed": "подтверждено"}


def cmd_rules(a):
    from .methodology import get_methodology
    m = get_methodology()
    wt = m.work_types.get(a.work_type)
    if not wt:
        sys.exit(f"вид работ {a.work_type} не найден; поиск: python -m app.cli find <слово>")
    p = m.profiles[wt.profile]
    print(f"{wt.id} {wt.name}\n  стадия {wt.stage} · профиль {p.id} «{p.name}» · наблюдаемость {wt.observability}")
    if wt.expert_note:
        print(f"  замечание экспертов: {wt.expert_note}")
    for g in p.required:
        print(f"  нужна: {' или '.join(m.class_name(c) for c in g.any_of)} ×{g.min} "
              f"({'за смену' if g.window == 'shift' else 'на снимке'}) — {g.role}")
    for c in p.companions:
        print(f"  пара: {'/'.join(m.class_name(x) for x in c.lead)} → {'/'.join(m.class_name(x) for x in c.partner)}: {c.message}")
    print(f"  допустимо: {', '.join(m.class_name(c) for c in sorted(p.allowed)) or '—'}")


def cmd_find(a):
    from .methodology import get_methodology
    m = get_methodology()
    q = a.query.lower()
    for wt in m.work_types.values():
        if q in wt.name.lower():
            print(f"{wt.id:14s} {wt.name}  [{wt.profile}, {wt.observability}]")


def cmd_check(a):
    """Один снимок против вида работ — без БД."""
    from .detection import get_detector
    from .engine import DetInfo, EngineContext, Obs, TaskInfo, ZoneWindow, evaluate_window
    from .imaging import assess_quality, load_image, resolve_time, sha256
    from .methodology import get_methodology
    from .schedule_io import parse_equipment
    m = get_methodology()
    data = Path(a.image).read_bytes()
    img = load_image(data)
    when, src = resolve_time(data, a.image, None)
    q = assess_quality(img, m.config.get("quality", {}))
    det = get_detector()
    res = det.detect(img, tiles=a.tiles, sha256=sha256(data)) if q.ok else None
    dets = res.dets if res else []
    tasks = []
    for i, w in enumerate(a.work_type, 1):
        wt = m.work_types[w]
        tasks.append(TaskInfo(i, wt.id, wt.name, wt.id, m.profiles[wt.profile], wt.observability, "FRAME",
                              when.date(), when.date(), planned=parse_equipment(a.planned, m) if i == 1 else {}))
    zw = ZoneWindow("FRAME", "Кадр", when.date(), [Obs([0], when, q.ok, q.reason,
                                                        [DetInfo(k, d.cls, d.conf) for k, d in enumerate(dets)])], tasks)
    findings, check = evaluate_window(zw, EngineContext(m, frozenset(det.classes)))
    print(f"снимок: {a.image} · {when:%d.%m.%Y %H:%M} ({src}) · качество: {'годен' if q.ok else q.reason}")
    print(f"детектор: {res.model if res else det.name}" + (f" · {res.note}" if res and res.note else ""))
    for d in dets:
        print(f"  {m.class_name(d.cls):22s} {d.conf:.2f}  [{', '.join(f'{v:.0f}' for v in d.box)}]")
    if not findings:
        print("отклонений нет")
    for f in findings:
        print(f"\n[{SEV[f.severity]}] {f.title} ({STATUS.get(f.status, f.status)})\n  {f.message}\n  → {f.recommendation}")
    if a.json:
        print(json.dumps({"check": check, "findings": [f.__dict__ for f in findings]}, ensure_ascii=False,
                         default=str, indent=1))


def _session():
    from .db import SessionLocal, init_db
    from .methodology import get_methodology, sync_to_db
    init_db()
    s = SessionLocal()
    sync_to_db(s, get_methodology())
    s.commit()
    return s


def cmd_seed(a):
    from .demo import seed_demo
    s = _session()
    ids = seed_demo(s)
    s.commit()
    print(f"загружены демо-проекты: {ids}")


def cmd_ingest(a):
    from . import models as M
    from . import services as S
    s = _session()
    p = s.get(M.Project, a.project) or sys.exit(f"проект {a.project} не найден")
    cam = S.camera_by_key(s, p, a.camera)
    files = sorted(f for f in Path(a.folder).rglob("*") if f.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp", ".webp"))
    for f in files:
        snap, info = S.ingest_snapshot(s, p, cam, f.read_bytes(), f.name)
        s.commit()
        print(f"{f.name}: {snap.taken_at:%d.%m.%Y %H:%M} ({snap.time_source}) · {info['processing_ms']} мс"
              + (" · дубликат" if info["duplicate"] else ""))
    print(f"обработано снимков: {len(files)}")


def cmd_report(a):
    import csv
    from sqlalchemy import select
    from . import models as M
    from . import services as S
    from .methodology import get_methodology
    s = _session()
    m = get_methodology()
    p = s.get(M.Project, a.project) or sys.exit(f"проект {a.project} не найден")
    q = select(M.Deviation).where(M.Deviation.project_id == p.id)
    if a.day:
        q = q.where(M.Deviation.day == dt.date.fromisoformat(a.day))
    devs = [S.deviation_view(s, d, m, p) for d in s.scalars(q.order_by(M.Deviation.day, M.Deviation.id))]
    for d in devs:
        print(f"{d['day']} [{SEV[d['severity']]}] {d['zone']['name']}: {d['title']} — {d['message']}")
    if a.csv:
        with open(a.csv, "w", newline="", encoding="utf-8-sig") as fh:
            w = csv.writer(fh, delimiter=";")
            w.writerow(["дата", "зона", "отклонение", "важность", "статус", "описание", "снимки"])
            for d in devs:
                w.writerow([d["day"], d["zone"]["name"], d["title"], d["severity"], d["status"], d["message"],
                            " ".join(str(e["snapshot_id"]) for e in d["evidence"])])
        print(f"CSV: {a.csv}")


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m app.cli", description="ОКО — поиск нарушений на стройплощадке")
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("check", help="проверить снимок против вида(ов) работ справочника")
    s.add_argument("image")
    s.add_argument("--work-type", "-w", action="append", required=True, help="id вида работ, можно несколько раз")
    s.add_argument("--planned", default="", help="техника по графику: «экскаватор ×1; самосвал ×3»")
    s.add_argument("--tiles", type=int, default=0)
    s.add_argument("--json", action="store_true")
    s.set_defaults(fn=cmd_check)
    s = sub.add_parser("rules", help="правила методики для вида работ")
    s.add_argument("work_type")
    s.set_defaults(fn=cmd_rules)
    s = sub.add_parser("find", help="найти вид работ в справочнике")
    s.add_argument("query")
    s.set_defaults(fn=cmd_find)
    s = sub.add_parser("seed-demo", help="загрузить демо-проекты в БД")
    s.set_defaults(fn=cmd_seed)
    s = sub.add_parser("ingest", help="загрузить папку снимков камеры в проект")
    s.add_argument("--project", type=int, required=True)
    s.add_argument("--camera", required=True)
    s.add_argument("folder")
    s.set_defaults(fn=cmd_ingest)
    s = sub.add_parser("report", help="отчёт об отклонениях")
    s.add_argument("--project", type=int, required=True)
    s.add_argument("--day")
    s.add_argument("--csv")
    s.set_defaults(fn=cmd_report)
    a = ap.parse_args(argv)
    a.fn(a)


if __name__ == "__main__":
    main()
