"""Слияние слоёв: детектор + модель готовности + правила + VLM + LLM → итог и исправления.

Итог по снимку (ai_reviews.final):
  equipment    — техника по классам: сколько видит детектор, VLM и LLM, итог и согласие слоёв;
  detections   — вердикт LLM по каждой рамке детектора (подтверждена, ложная, другой класс, не разобрать);
  missed       — техника, которую детектор пропустил (рамка в долях кадра);
  stage        — стадия по графику, модели готовности, VLM и LLM, итог и согласие;
  deviations   — вердикт LLM по каждому отклонению правил;
  new_findings, forecast, recommendations, confidence;
  corrections  — что меняется в данных при «Применить»: ложные рамки исключаются, класс исправляется,
                 пропущенная техника добавляется (source=llm). Затем правила пересчитываются по снимку и дню,
                 а «Отменить» возвращает всё как было.
"""
from __future__ import annotations

import json
from collections import Counter

from sqlalchemy.orm import Session

from .. import models as M
from .. import services as S
from ..methodology import Methodology, get_methodology
from .context import stage_of

VERDICTS = ("confirmed", "false_positive", "wrong_class", "uncertain")


def _num(v, default: float = 0.0) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


_CONF_WORDS = (("очень высок", 0.9), ("высок", 0.8), ("средн", 0.6), ("низк", 0.3),
               ("very high", 0.9), ("high", 0.8), ("medium", 0.6), ("moderate", 0.6), ("low", 0.3))


def conf01(v) -> float | None:
    """Уверенность LLM в долях 0..1. Модели пишут её по-разному: 0.85, \"0,85\", 85, \"85%\", \"высокая\",
    {\"value\": 0.85}. Нет или не разобрать — None (а не 0: иначе все исправления считались бы неуверенными)."""
    if isinstance(v, dict):
        v = next((v[k] for k in ("value", "score", "overall", "confidence") if k in v), None)
    if v is None or isinstance(v, bool):
        return None
    pct = False
    if isinstance(v, str):
        s = v.strip().lower().replace(",", ".")
        pct = s.endswith("%")
        try:
            v = float(s.rstrip("%").strip())
        except ValueError:
            return next((x for w, x in _CONF_WORDS if w in s), None)
    try:
        x = float(v)
    except (TypeError, ValueError):
        return None
    if x != x:                                  # NaN
        return None
    if pct or 1.0 < x <= 100.0:
        x /= 100.0
    return min(1.0, max(0.0, x))


def vlm_counts(m: Methodology, vlm_out: dict | None) -> Counter:
    out: Counter = Counter()
    for it in (vlm_out or {}).get("equipment") or []:
        if not isinstance(it, dict):
            continue
        cls = m.normalize_class(str(it.get("type", "")).split("|")[0].strip())
        if cls:
            out[cls] += max(1, int(_num(it.get("count"), 1)))
    return out


def iou(a, b) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def cross_check(m: Methodology, dets: list[dict], vlm_out: dict | None, min_iou: float = 0.3) -> dict | None:
    """Сверка рамок детектора и VLM (обе в долях кадра): совпали, спорный класс, только VLM, только детектор.

    None — VLM выключена или не дала рамок: сверять нечего.
    """
    items = []
    for it in (vlm_out or {}).get("equipment") or []:
        if isinstance(it, dict) and it.get("box"):
            items.append({"type": str(it.get("type", "")), "cls": m.normalize_class(str(it.get("type", ""))),
                          "box": it["box"], "state": it.get("state")})
    if not items:
        return None
    used: set[int] = set()
    out = {"agree": [], "class_conflict": [], "vlm_only": [], "detector_only": []}
    for it in items:
        best, best_iou = None, 0.0
        for d in dets:
            v = iou(it["box"], d["box"])
            if v > best_iou:
                best, best_iou = d, v
        if best is not None and best_iou >= min_iou:
            used.add(best["id"])
            row = {"detection_id": best["id"], "detector_cls": best["cls"], "vlm_cls": it["cls"],
                   "vlm_type": it["type"], "iou": round(best_iou, 2)}
            out["agree" if it["cls"] == best["cls"] else "class_conflict"].append(row)
        elif it["cls"]:
            out["vlm_only"].append({"cls": it["cls"], "vlm_type": it["type"], "box": it["box"], "state": it["state"]})
    out["detector_only"] = [d["id"] for d in dets if d["above_threshold"] and d["id"] not in used]
    return out


def _box01(box, width: int, height: int, per_mille: bool = False) -> list[float] | None:
    """Рамка LLM в долях кадра. per_mille — модель, которая видит кадр, даёт рамки в тысячных долях (0–1000);
    иначе числа больше 1 — пиксели исходного кадра. Вырожденные рамки отбрасываются."""
    if not isinstance(box, (list, tuple)) or len(box) != 4:
        return None
    b = [_num(v, -1) for v in box]
    if max(b) > 1.5 and per_mille and max(b) <= 1000.5:
        b = [v / 1000 for v in b]
    elif max(b) > 1.5:                                # модель вернула пиксели
        b = [b[0] / width, b[1] / height, b[2] / width, b[3] / height]
    x1, y1, x2, y2 = (min(1.0, max(0.0, v)) for v in b)
    if x2 - x1 < 0.01 or y2 - y1 < 0.01:
        return None
    return [round(x1, 4), round(y1, 4), round(x2, 4), round(y2, 4)]


def _as_list(v) -> list:
    if v is None or v == "":
        return []
    return v if isinstance(v, list) else [v]


def _as_text(v) -> str:
    if v is None:
        return ""
    if isinstance(v, list):
        return " ".join(_as_text(x) for x in v)
    return v if isinstance(v, str) else json.dumps(v, ensure_ascii=False)


def _as_dict(v, key: str) -> dict:
    return v if isinstance(v, dict) else ({key: v} if v not in (None, "") else {})


def clean_snapshot_output(out: dict | None) -> dict:
    """Ответ LLM → ожидаемые типы полей. Схема строго соблюдается не всегда (json_object у OpenAI, запасной
    режим YandexGPT): строка вместо списка или объекта не должна ронять сведение слоёв."""
    if not out:
        return {}
    o = dict(out)
    for k in ("detections", "missed", "deviations", "new_findings"):
        o[k] = [x for x in _as_list(o.get(k)) if isinstance(x, dict)]
    o["recommendations"] = [_as_text(x) for x in _as_list(o.get("recommendations")) if x]
    o["stage"] = _as_dict(o.get("stage"), "value")
    o["scene"] = _as_dict(o.get("scene"), "conditions")
    o["summary"], o["forecast"] = _as_text(o.get("summary")), _as_text(o.get("forecast"))
    return o


def clean_day_output(out: dict) -> dict:
    o = dict(out)
    for k in ("risks", "forecast", "zones"):
        o[k] = [x if isinstance(x, dict) else {"title": _as_text(x), "task": _as_text(x)} for x in _as_list(o.get(k))]
    for x in o["forecast"]:
        x["expected_delay_days"] = _num(x.get("expected_delay_days"), 0.0)
    for x in o["risks"]:
        x["probability"] = min(1.0, max(0.0, _num(x.get("probability"), 0.0)))
    for k in ("model_errors", "recommendations"):
        o[k] = [_as_text(x) for x in _as_list(o.get(k)) if x]
    o["summary"], o["status"] = _as_text(o.get("summary")), _as_text(o.get("status"))
    o["confidence"] = conf01(o.get("confidence"))
    return o


def fuse_snapshot(ctx: dict, vlm_out: dict | None, llm_out: dict | None, min_conf: float,
                  meta: dict | None = None) -> dict:
    m = get_methodology()
    llm = clean_snapshot_output(llm_out)
    has_llm = bool(llm_out) and "raw" not in llm
    vlm_ok = bool(vlm_out) and "error" not in vlm_out and "raw" not in vlm_out
    W, H = ctx["image"]["width"], ctx["image"]["height"]
    per_mille = bool((meta or {}).get("llm_vision"))   # LLM видела кадр — рамки в тысячных долях
    llm_conf = conf01(llm.get("confidence"))
    if llm_conf is None:                         # общей нет — берём уверенность в стадии, если есть
        llm_conf = conf01((llm.get("stage") or {}).get("confidence"))

    # --- рамки детектора: вердикт LLM
    by_id = {}
    for x in llm.get("detections") or []:
        if isinstance(x, dict) and str(x.get("id", "")).lstrip("#").isdigit():
            by_id[int(str(x["id"]).lstrip("#"))] = x
    det_rows = []
    for d in ctx["detections"]:
        v = by_id.get(d["id"], {})
        verdict = v.get("verdict") if v.get("verdict") in VERDICTS else ("uncertain" if has_llm else "not_checked")
        cc = v.get("correct_class")
        if verdict == "wrong_class" and (cc not in m.classes or cc == d["cls"]):
            verdict, cc = "uncertain", None
        det_rows.append({"id": d["id"], "cls": d["cls"], "name": d["name"], "conf": d["conf"], "zone": d["zone"],
                         "above_threshold": d["above_threshold"], "verdict": verdict,
                         "correct_class": cc if verdict == "wrong_class" else None,
                         "correct_name": m.class_name(cc) if verdict == "wrong_class" else None,
                         "comment": str(v.get("comment", ""))})

    # --- пропущенная техника
    missed = []
    for x in llm.get("missed") or []:
        if not isinstance(x, dict) or x.get("cls") not in m.classes:
            continue
        box = _box01(x.get("box"), W, H, per_mille)
        if box is None:
            continue
        conf = conf01(x.get("confidence"))
        conf = 0.5 if conf is None else conf
        missed.append({"cls": x["cls"], "name": m.class_name(x["cls"]), "box": box,
                       "xyxy": [round(box[0] * W, 1), round(box[1] * H, 1), round(box[2] * W, 1), round(box[3] * H, 1)],
                       "conf": round(conf, 2), "comment": str(x.get("comment", ""))})

    # --- техника по классам: детектор / VLM / LLM
    det_c = Counter(d["cls"] for d in ctx["detections"] if d["above_threshold"])
    vlm_c = vlm_counts(m, vlm_out) if vlm_ok else Counter()
    llm_c: Counter = Counter()
    for r in det_rows:
        if r["verdict"] in ("confirmed", "uncertain") and (r["above_threshold"] or r["verdict"] == "confirmed"):
            llm_c[r["cls"]] += 1
        elif r["verdict"] == "wrong_class":
            llm_c[r["correct_class"]] += 1
    for x in missed:
        llm_c[x["cls"]] += 1
    equipment = []
    for c in sorted(set(det_c) | set(vlm_c) | set(llm_c), key=lambda k: m.class_name(k)):
        row = {"cls": c, "name": m.class_name(c), "detector": det_c.get(c, 0),
               "vlm": vlm_c.get(c, 0) if vlm_ok else None, "llm": llm_c.get(c, 0) if has_llm else None}
        row["final"] = row["llm"] if has_llm else row["detector"]
        votes = [v for v in (row["detector"], row["vlm"], row["llm"]) if v is not None]
        row["agree"] = len(set(votes)) == 1
        equipment.append(row)

    # --- стадия: график, модель готовности, VLM, LLM
    rm = ctx["stage"].get("readiness_model") or {}
    planned = sorted({st for v in ctx["stage"]["planned"].values() for st in v})
    llm_stage = llm.get("stage") or {}
    st = {"planned": planned, "model": stage_of(rm), "model_readiness": rm.get("readiness"),
          "model_expected": rm.get("expected"), "model_status": rm.get("status_ru"),
          "model_delta_days": rm.get("delta_days"), "model_name": rm.get("model"),
          "vlm": stage_of(vlm_out.get("stage")) if vlm_ok else None,
          "llm": stage_of(llm_stage.get("value")) if has_llm else None,
          "llm_readiness": llm_stage.get("readiness"), "comment": llm_stage.get("comment", "")}
    votes = [v for v in (st["model"], st["vlm"], st["llm"]) if v]
    st["final"] = st["llm"] or st["model"] or (Counter(votes).most_common(1)[0][0] if votes else None)
    st["agree"] = len(set(votes)) <= 1
    st["matches_plan"] = (st["final"] in planned) if st["final"] and planned else None

    # --- отклонения правил: вердикт LLM
    dv = {}
    for x in llm.get("deviations") or []:
        if isinstance(x, dict) and str(x.get("id", "")).isdigit():
            dv[int(x["id"])] = x
    devs = [{"id": d["id"], "rule": d["rule"], "title": d["title"], "severity": d["severity"], "zone": d["zone"],
             "verdict": (dv.get(d["id"]) or {}).get("verdict") or ("not_checked" if not has_llm else "doubtful"),
             "comment": str((dv.get(d["id"]) or {}).get("comment", ""))} for d in ctx["deviations"]]

    # --- исправления, которые можно применить
    corrections = []
    gate = has_llm and llm_conf is not None and llm_conf >= min_conf
    for r in det_rows:
        if r["verdict"] == "false_positive":
            corrections.append({"action": "reject", "detection_id": r["id"], "cls": r["cls"], "apply": gate,
                                "text": f"#{r['id']} {r['name']}: ложная рамка" + (f" — {r['comment']}" if r["comment"] else "")})
        elif r["verdict"] == "wrong_class":
            corrections.append({"action": "reclass", "detection_id": r["id"], "from": r["cls"], "to": r["correct_class"],
                                "apply": gate, "text": f"#{r['id']} {r['name']} → {r['correct_name']}"
                                                       + (f" — {r['comment']}" if r["comment"] else "")})
    for x in missed:
        corrections.append({"action": "add", "cls": x["cls"], "box": x["box"], "conf": x["conf"],
                            "apply": has_llm and x["conf"] >= min_conf,
                            "text": f"+ {x['name']} ({x['conf']:.2f}), пропущен детектором"
                                    + (f" — {x['comment']}" if x["comment"] else "")})

    scene = dict(llm.get("scene") or {}) if has_llm else {}
    if vlm_ok:
        for k in ("activity", "people", "conditions"):
            scene.setdefault(k, vlm_out.get(k))
        scene["vlm_scene"] = vlm_out.get("scene")
        scene["vlm_notes"] = vlm_out.get("notes")
    summary = llm.get("summary") if has_llm else (vlm_out.get("scene") if vlm_ok else "")
    return {
        "summary": summary or "", "scene": scene, "equipment": equipment, "detections": det_rows, "missed": missed,
        "stage": st, "deviations": devs, "cross_check": ctx.get("cross_check"),
        "new_findings": [x for x in llm.get("new_findings") or [] if isinstance(x, dict)],
        "forecast": llm.get("forecast", "") if has_llm else "",
        "recommendations": [str(x) for x in llm.get("recommendations") or []],
        "confidence": llm_conf if has_llm else None,
        "corrections": corrections,
        "layers": {"detector": ctx["detector"]["model"], "readiness_model": bool(rm), "vlm": vlm_ok,
                   "llm": has_llm, **(meta or {})},
        "min_conf": min_conf,
    }


# ------------------------------------------------------------------ применение и отмена исправлений

def _recompute(s: Session, snap: M.Snapshot) -> None:
    p = s.get(M.Project, snap.project_id)
    s.flush()
    s.refresh(snap)
    S.recompute_snapshot(s, p, snap)
    S.recompute_day(s, p, snap.taken_at.date())


def apply_corrections(s: Session, review: M.AIReview) -> dict:
    """Применить исправления LLM (только помеченные apply=true) и пересчитать правила по снимку и дню."""
    if review.kind != "snapshot" or review.status != "done" or not review.final:
        raise ValueError("анализ не завершён")
    if review.applied and not review.applied.get("reverted"):
        raise ValueError("исправления уже применены")
    m = get_methodology()
    snap = s.get(M.Snapshot, review.snapshot_id)
    p = s.get(M.Project, snap.project_id)
    cam = s.get(M.Camera, snap.camera_id)
    zone_for = S.zone_assigner(cam, {z.id: z.key for z in p.zones}, snap.frame_zone, snap.width, snap.height, m)

    dets = {d.id: d for d in snap.detections}
    applied = {"rejected": [], "reclassified": [], "added": [], "at": M.utcnow().isoformat(timespec="seconds")}
    for c in review.final.get("corrections", []):
        if not c.get("apply"):
            continue
        d = dets.get(c.get("detection_id"))
        if c["action"] == "reject" and d is not None and not d.is_rejected:
            d.is_rejected = True
            applied["rejected"].append(d.id)
        elif c["action"] == "reclass" and d is not None and d.equipment_class == c["from"] and c["to"] in m.classes:
            applied["reclassified"].append({"id": d.id, "from": d.equipment_class, "to": c["to"], "zone_id": d.zone_id})
            d.equipment_class = c["to"]
            d.zone_id = zone_for(c["to"], (d.x1, d.y1, d.x2, d.y2))
        elif c["action"] == "add" and c.get("cls") in m.classes:
            b = c["box"]
            px = (b[0] * snap.width, b[1] * snap.height, b[2] * snap.width, b[3] * snap.height)
            nd = M.Detection(snapshot_id=snap.id, equipment_class=c["cls"], confidence=round(float(c["conf"]), 3),
                             x1=px[0], y1=px[1], x2=px[2], y2=px[3], zone_id=zone_for(c["cls"], px), source="llm")
            s.add(nd)
            s.flush()
            applied["added"].append(nd.id)
    applied["changes"] = len(applied["rejected"]) + len(applied["reclassified"]) + len(applied["added"])
    _recompute(s, snap)
    review.applied = applied
    s.flush()
    return applied


def revert_corrections(s: Session, review: M.AIReview) -> dict:
    """Вернуть рамки к состоянию до применения и пересчитать правила."""
    a = review.applied
    if not a or a.get("reverted"):
        raise ValueError("нечего отменять")
    snap = s.get(M.Snapshot, review.snapshot_id)
    for did in a.get("rejected", []):
        d = s.get(M.Detection, did)
        if d is not None:
            d.is_rejected = False
    for r in a.get("reclassified", []):
        d = s.get(M.Detection, r["id"])
        if d is not None:
            d.equipment_class, d.zone_id = r["from"], r["zone_id"]
    for did in a.get("added", []):
        d = s.get(M.Detection, did)
        if d is not None:
            s.delete(d)
    s.flush()
    _recompute(s, snap)
    review.applied = {**a, "reverted": M.utcnow().isoformat(timespec="seconds")}
    s.flush()
    return review.applied
