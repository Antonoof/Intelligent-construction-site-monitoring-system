"""Загрузка демо-проектов из data/demo (создаются tools/make_demo_data.py).

Демо-проекты всегда загружаются с эталонной разметкой сцен (демо-детектор), даже если подключён
RF-DETR: синтетические снимки нужны для показа методики, а не для оценки детектора.
"""
from __future__ import annotations

import datetime as dt
import json
import logging
from pathlib import Path

from sqlalchemy.orm import Session

from . import services as S
from .config import settings
from .detection.demo import DemoDetector

log = logging.getLogger("oko.demo")


def seed_demo(s: Session, demo_dir: Path | None = None) -> list[int]:
    demo_dir = Path(demo_dir or settings.demo_dir)
    manifest = demo_dir / "projects.json"
    if not manifest.exists():
        log.warning("нет демо-данных: %s (запустите python tools/make_demo_data.py)", manifest)
        return []
    detector = DemoDetector(demo_dir)
    ids = []
    for pj in json.loads(manifest.read_text(encoding="utf-8"))["projects"]:
        p = S.create_project(s, pj["name"], pj["object_type"], pj.get("address", ""), pj.get("description", ""),
                             is_demo=True)
        S.apply_site_config(s, p, json.loads((demo_dir / pj["site"]).read_text(encoding="utf-8")))
        sched = demo_dir / pj["schedule"]
        S.import_schedule(s, p, sched.name, sched.read_bytes())
        # служебные файлы macOS (._имя — AppleDouble из архивов, собранных на Mac) пропускаем
        shots = sorted(f for f in (demo_dir / pj["snapshots"]).glob("*.jpg") if not f.name.startswith("."))
        for img in shots:
            meta = json.loads(img.with_suffix(".json").read_text(encoding="utf-8"))
            cam = S.camera_by_key(s, p, meta["camera"])
            S.ingest_snapshot(s, p, cam, img.read_bytes(), img.name, dt.datetime.fromisoformat(meta["taken_at"]),
                              recompute=False, detector=detector)
        S.recompute_project(s, p)
        s.flush()
        ids.append(p.id)
        log.info("демо-проект «%s»: %d снимков", p.name, len(shots))
    return ids
