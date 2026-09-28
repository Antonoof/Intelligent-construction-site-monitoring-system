"""Справочники методики: классы техники, профили, виды работ, правила отклонений."""
from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import FileResponse

from ..config import settings
from ..detection import get_detector
from ..methodology import get_methodology

router = APIRouter(prefix="/api/reference", tags=["Методика"])


@router.get("/equipment", summary="Классы техники и какие из них распознаёт текущая модель")
def equipment():
    m = get_methodology()
    det = get_detector()
    return [{"key": k, "name": v["name"], "name_en": v.get("name_en"), "in_tz": bool(v.get("tz")),
             "site_wide": bool(v.get("site_wide")), "models": v.get("models", []), "color": v.get("color"),
             "detected_now": k in det.classes} for k, v in m.classes.items()]


def _profile_view(p, m):
    return {"id": p.id, "name": p.name, "description": p.description, "observability": p.observability,
            "is_composite": p.is_composite,
            "required": [{"role": g.role, "any_of": list(g.any_of), "min": g.min, "window": g.window} for g in p.required],
            "companions": [{"lead": list(c.lead), "partner": list(c.partner), "severity": c.severity,
                            "message": c.message, "window": c.window} for c in p.companions],
            "allowed": sorted(p.allowed), "not_detected": sorted(p.not_detected)}


@router.get("/profiles", summary="Профили техники (ядро методики «этап → техника»)")
def profiles(include_composite: bool = False):
    m = get_methodology()
    return [_profile_view(p, m) for p in m.profiles.values() if include_composite or not p.is_composite]


@router.get("/profiles/{profile_id}")
def profile(profile_id: str):
    m = get_methodology()
    return _profile_view(m.profiles[profile_id], m)


@router.get("/work-types", summary="377 видов работ справочника с профилем, стадией и наблюдаемостью")
def work_types(q: str = "", object_type: str = "", stage: str = "", limit: int = Query(500, le=1000)):
    m = get_methodology()
    ql = q.lower().strip()
    out = []
    for wt in m.work_types.values():
        if ql and ql not in wt.name.lower() and not wt.id.startswith(ql):
            continue
        if object_type and object_type not in wt.object_types:
            continue
        if stage and wt.stage != stage:
            continue
        p = m.profiles[wt.profile]
        out.append({"id": wt.id, "code": wt.code, "name": wt.name, "level": wt.level, "parent": wt.parent,
                    "stage": wt.stage, "profile": wt.profile, "profile_name": p.name,
                    "observability": wt.observability, "object_types": list(wt.object_types),
                    "expert_note": wt.expert_note, "has_children": wt.has_children,
                    "required": [{"role": g.role, "any_of": list(g.any_of), "window": g.window} for g in p.required],
                    "companions": [{"lead": list(c.lead), "partner": list(c.partner)} for c in p.companions],
                    "allowed": sorted(p.allowed)})
        if len(out) >= limit:
            break
    return out


@router.get("/rules", summary="Правила выявления отклонений")
def rules():
    m = get_methodology()
    return {"version": m.rules_version, "config": m.config,
            "rules": [{"code": r.code, "title": r.title, "kind": r.kind, "severity": r.severity,
                       "confirmed_severity": r.confirmed_severity, "message": r.message,
                       "recommendation": r.recommendation} for r in m.rules.values()],
            "object_types": m.object_types}


@router.get("/methodology.xlsx", summary="Методика в XLSX: справочник организаторов + столбцы методики")
def methodology_xlsx():
    p = settings.methodology_dir / "Методика_этап_техника.xlsx"
    if not p.exists():
        raise HTTPException(404, "файл методики не собран: python tools/build_methodology.py")
    return FileResponse(p, filename="Методика_этап_техника.xlsx",
                        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
