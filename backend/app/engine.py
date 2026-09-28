"""Движок сопоставления «снимки → этапы графика → отклонения».

Чистые функции без БД и ввода-вывода: на входе наблюдения зоны за окно (день) и этапы графика,
на выходе — отклонения с доказательствами и подробное объяснение проверок. Один и тот же код
оценивает и отдельный снимок (окно из одного снимка → «предварительно»), и серию снимков за день
(→ «подтверждено»), поэтому вывод по снимку и по дню всегда согласованы.

Правила (коды — methodology/rules.yaml):
  NO_DATA              снимки зоны непригодны (темно, размыто, засветка) — «нет данных», не «нет техники»;
  NO_EQUIPMENT         этап идёт по графику, а техники в зоне нет;
  MISSING_REQUIRED     нет техники для одной из обязательных ролей профиля (например, вывоза грунта);
  INCOMPLETE_SET       ведущая техника без парной — пример ТЗ: экскаватор без самосвалов;
  INSUFFICIENT_COUNT   техники меньше, чем записано в графике;
  EARLY_START          техника этапа, который ещё не начался (опережение);
  STAGE_OVERRUN        техника этапа, который по графику закончился (затягивание);
  UNEXPECTED_EQUIPMENT техника не нужна ни одному идущему этапу зоны;
  NO_ACTIVE_STAGE      техника в зоне, где по графику работ нет.
"""
from __future__ import annotations

import datetime as dt
from collections import Counter
from dataclasses import dataclass, field

from .methodology import Group, Methodology, Profile

SITE_ZONE = "__site__"          # псевдозона «вся площадка» для этапов графика без зоны


# ------------------------------------------------------------------ входные данные

@dataclass(frozen=True)
class DetInfo:
    id: int | None
    cls: str
    conf: float


@dataclass
class Obs:
    """Наблюдение зоны на одном снимке (или на нескольких снимках одного 5-минутного слота)."""
    snapshot_ids: list[int]
    taken_at: dt.datetime
    valid: bool
    reason: str = ""
    dets: list[DetInfo] = field(default_factory=list)


@dataclass
class TaskInfo:
    id: int | None
    wbs: str
    name: str
    work_type_id: str | None
    profile: Profile | None
    observability: str
    zone_key: str | None            # None — этап на всю площадку
    start: dt.date
    end: dt.date
    is_summary: bool = False
    planned: dict[str, int] = field(default_factory=dict)

    @property
    def title(self) -> str:
        return f"{self.wbs} {self.name}".strip() if self.wbs else self.name


@dataclass
class ZoneWindow:
    zone_key: str
    zone_name: str
    day: dt.date
    obs: list[Obs]
    active: list[TaskInfo]
    upcoming: list[TaskInfo] = field(default_factory=list)
    recent: list[TaskInfo] = field(default_factory=list)
    site_equipment: set[str] = field(default_factory=set)   # заявленная техника, обслуживающая зону
    site_active: list[TaskInfo] = field(default_factory=list)  # идущие этапы «на всю площадку»
    check_unexpected: bool = True                            # в псевдозоне «вся площадка» не проверяем


@dataclass
class Finding:
    rule: str
    zone_key: str
    severity: str
    status: str                      # preliminary | confirmed
    title: str
    message: str
    recommendation: str
    task_id: int | None = None
    cls: str | None = None
    role: str | None = None
    metrics: dict = field(default_factory=dict)
    evidence: list[tuple[int, list[int], str]] = field(default_factory=list)   # (snapshot_id, det_ids, note)
    first_seen: dt.datetime | None = None
    last_seen: dt.datetime | None = None
    snapshots_count: int = 0

    def key(self, day: dt.date) -> str:
        """Ключ дедупликации: одно отклонение на (день, зона, этап, правило, класс/роль)."""
        return "|".join([day.isoformat(), self.zone_key, str(self.task_id or "-"), self.rule,
                         self.cls or self.role or "-"])


@dataclass
class EngineContext:
    m: Methodology
    detectable: frozenset            # классы, которые распознаёт текущая модель

    @property
    def windows(self) -> dict:
        return self.m.config.get("windows", {})


# ------------------------------------------------------------------ вспомогательные

def _plural(n: int, one: str, few: str, many: str) -> str:
    n = abs(n) % 100
    if 11 <= n <= 14:
        return many
    n %= 10
    return one if n == 1 else few if 2 <= n <= 4 else many


def snapshots_phrase(obs: list[Obs]) -> str:
    if not obs:
        return "нет снимков"
    ts = sorted(o.taken_at for o in obs)
    n = sum(len(o.snapshot_ids) for o in obs)
    word = _plural(n, "снимок", "снимка", "снимков")
    if len(ts) == 1 or ts[0] == ts[-1]:
        return f"{n} {word} в {ts[0]:%H:%M}"
    return f"{n} {word} с {ts[0]:%H:%M} до {ts[-1]:%H:%M}"


class _Counts:
    """Счётчики техники на наблюдении с двумя порогами уверенности."""

    def __init__(self, obs: Obs, m: Methodology):
        weak_thr = m.weak_threshold()
        self.strong = Counter(d.cls for d in obs.dets if d.conf >= m.presence_threshold(d.cls))
        self.weak = Counter(d.cls for d in obs.dets if d.conf >= weak_thr)
        self.ids: dict[str, list[int]] = {}
        for d in obs.dets:
            if d.id is not None and d.conf >= weak_thr:
                self.ids.setdefault(d.cls, []).append(d.id)
        self.max_conf: dict[str, float] = {}
        for d in obs.dets:
            self.max_conf[d.cls] = max(self.max_conf.get(d.cls, 0.0), d.conf)


def merge_slots(obs: list[Obs], slot_minutes: int = 5) -> list[Obs]:
    """Снимки разных камер одной зоны за один слот — одно наблюдение.

    Счёт по классу — максимум по камерам, а не сумма: две камеры, видящие один экскаватор,
    не дают двух экскаваторов.
    """
    slots: dict[tuple, list[Obs]] = {}
    for o in obs:
        t = o.taken_at
        key = (t.date(), (t.hour * 60 + t.minute) // slot_minutes)
        slots.setdefault(key, []).append(o)
    merged: list[Obs] = []
    for key in sorted(slots):
        group = slots[key]
        if len(group) == 1:
            merged.append(group[0])
            continue
        valid = [o for o in group if o.valid]
        best: dict[str, list[DetInfo]] = {}
        for o in valid:
            by_cls: dict[str, list[DetInfo]] = {}
            for d in o.dets:
                by_cls.setdefault(d.cls, []).append(d)
            for c, lst in by_cls.items():
                if len(lst) > len(best.get(c, [])):
                    best[c] = lst
        merged.append(Obs(snapshot_ids=[i for o in group for i in o.snapshot_ids],
                          taken_at=min(o.taken_at for o in group), valid=bool(valid),
                          reason="; ".join(sorted({o.reason for o in group if o.reason})),
                          dets=[d for lst in best.values() for d in lst]))
    return merged


def _fmt(template: str, **kw) -> str:
    class _Safe(dict):
        def __missing__(self, k):
            return ""
    return template.format_map(_Safe(**kw)).replace("  ", " ").strip()


def _names(m: Methodology, classes, sep=" или ") -> str:
    return sep.join(m.class_name(c).lower() for c in classes)


def _observed_text(m: Methodology, counts: Counter) -> str:
    items = [f"{m.class_name(c).lower()} ×{n}" for c, n in sorted(counts.items(), key=lambda x: -x[1]) if n]
    return ", ".join(items) if items else "техники нет"


# ------------------------------------------------------------------ оценка окна зоны

def evaluate_window(zw: ZoneWindow, ctx: EngineContext) -> tuple[list[Finding], dict]:
    """Оценить зону за окно: вернуть отклонения и объяснение проверок для интерфейса."""
    m = ctx.m
    win = ctx.windows
    confirm_min = int(win.get("confirm_min_snapshots", 2))
    confirm_conf = float(win.get("confirm_single_confidence", 0.8))
    share_thr = float(win.get("incomplete_share", 0.5))
    min_span = float(win.get("no_equipment_min_span_minutes", 60))
    site_wide = m.site_wide
    rules = m.rules

    obs = merge_slots(zw.obs)
    valid = [o for o in obs if o.valid]
    invalid = [o for o in obs if not o.valid]
    findings: list[Finding] = []
    check: dict = {"zone": zw.zone_key, "zone_name": zw.zone_name, "day": zw.day.isoformat(),
                   "snapshots": sum(len(o.snapshot_ids) for o in obs), "valid": len(valid),
                   "tasks": [], "unexpected": [], "observed": {}}

    def mk(rule_code: str, *, confirmed: bool, severity: str | None = None, obs_list: list[Obs],
           det_ids_for=None, note: str = "", **fields) -> Finding:
        r = rules[rule_code]
        sev = severity or (r.confirmed_severity if confirmed else r.severity)
        ev = []
        for o in obs_list[:8]:
            ids = det_ids_for(o) if det_ids_for else []
            for sid in o.snapshot_ids:
                ev.append((sid, ids, note))
        f = Finding(rule=rule_code, zone_key=zw.zone_key, severity=sev,
                    status="confirmed" if confirmed else "preliminary", title=r.title,
                    message=_fmt(r.message, **fields.pop("fmt", {})),
                    recommendation=_fmt(r.recommendation, **fields.pop("fmt_rec", {})),
                    evidence=ev,
                    first_seen=min((o.taken_at for o in obs_list), default=None),
                    last_seen=max((o.taken_at for o in obs_list), default=None),
                    snapshots_count=sum(len(o.snapshot_ids) for o in obs_list), **fields)
        return f

    # --- нет годных снимков: «нет данных», а не «нет техники»
    if not valid:
        if invalid:
            reason = "; ".join(sorted({o.reason for o in invalid if o.reason})) or "качество снимка"
            findings.append(mk("NO_DATA", confirmed=len(invalid) >= confirm_min, obs_list=invalid,
                               fmt={"reason": reason, "zone": zw.zone_name}))
        check["status"] = "no_data"
        return findings, check

    counts = {id(o): _Counts(o, m) for o in valid}
    C = lambda o: counts[id(o)]  # noqa: E731

    def zone_weak(o: Obs) -> Counter:
        """Техника зоны без общеплощадочной (башенный кран не делает зону «рабочей»)."""
        return Counter({c: n for c, n in C(o).weak.items() if c not in site_wide})

    empty_obs = [o for o in valid if sum(zone_weak(o).values()) == 0]
    nonempty_obs = [o for o in valid if o not in empty_obs]
    observed_max: Counter = Counter()
    for o in valid:
        for c, n in C(o).strong.items():
            observed_max[c] = max(observed_max[c], n)
    check["observed"] = dict(observed_max)
    single = len(valid) == 1
    span_min = (max(o.taken_at for o in valid) - min(o.taken_at for o in valid)).total_seconds() / 60

    def ids_of(classes):
        return lambda o: [i for c in classes for i in C(o).ids.get(c, [])]

    # --- требования идущих этапов
    for t in zw.active:
        tc: dict = {"task_id": t.id, "wbs": t.wbs, "name": t.name, "work_type_id": t.work_type_id,
                    "profile": t.profile.id if t.profile else None,
                    "profile_name": t.profile.name if t.profile else "не сопоставлен со справочником",
                    "observability": t.observability, "period": [t.start.isoformat(), t.end.isoformat()],
                    "groups": [], "companions": [], "planned": [], "status": "ok"}
        check["tasks"].append(tc)
        if t.is_summary:
            tc["status"] = "summary"
            continue
        p = t.profile
        if p is None:
            tc["status"] = "not_matched"
            continue
        groups: list[Group] = list(p.required)
        covered = {c for g in groups for c in g.any_of}
        for c, n in t.planned.items():            # техника из графика, которой нет в профиле
            if c not in covered and c in m.classes and c not in site_wide:
                # присутствие проверяет группа, количество — правило INSUFFICIENT_COUNT
                groups.append(Group(role="по графику", any_of=(c,), min=1, window="shift"))
        obs_level = t.observability
        if obs_level in ("low", "none"):
            tc["status"] = "not_checked"
            tc["reason"] = "наблюдаемость низкая: проверяется только лишняя техника"
            for g in groups:
                tc["groups"].append({"role": g.role, "expected": _names(m, g.any_of), "min": g.min,
                                     "window": g.window, "state": "reference"})
            continue
        if not groups and not p.companions:
            tc["status"] = "not_checked"
            continue

        def group_ok(o: Obs, g: Group, det_classes) -> bool:
            if any(c in zw.site_equipment for c in g.any_of):
                return True
            return sum(C(o).weak.get(c, 0) for c in det_classes) >= g.min

        # 1) зона пуста на всех снимках → «работы не ведутся»
        checkable = [g for g in groups if any(c in ctx.detectable for c in g.any_of)
                     and not any(c in zw.site_equipment for c in g.any_of)]
        if checkable and len(empty_obs) == len(valid) and (obs_level == "high" or len(valid) >= confirm_min):
            confirmed = len(valid) >= confirm_min and span_min >= min_span
            findings.append(mk("NO_EQUIPMENT", confirmed=confirmed, obs_list=empty_obs, task_id=t.id,
                               metrics={"snapshots": len(valid), "span_minutes": round(span_min),
                                        "expected": [_names(m, g.any_of) for g in checkable]},
                               fmt={"task": t.title, "zone": zw.zone_name, "snapshots": snapshots_phrase(empty_obs)}))
            tc["status"] = "deviation"
            for g in groups:
                tc["groups"].append({"role": g.role, "expected": _names(m, g.any_of), "min": g.min,
                                     "window": g.window, "observed": 0, "state": "missing"})
            continue

        # 2) обязательные группы
        missing: list[tuple[Group, list[Obs], bool]] = []
        for g in groups:
            det_classes = [c for c in g.any_of if c in ctx.detectable]
            gi = {"role": g.role, "expected": _names(m, g.any_of), "min": g.min, "window": g.window}
            if not det_classes:
                gi["state"] = "not_detectable"
                tc["groups"].append(gi)
                continue
            if any(c in zw.site_equipment for c in g.any_of):
                gi["state"] = "site_equipment"
                tc["groups"].append(gi)
                continue
            gi["observed"] = max((sum(C(o).weak.get(c, 0) for c in det_classes) for o in valid), default=0)
            window = g.window if obs_level == "high" else "shift"
            if window == "snapshot":
                miss = [o for o in nonempty_obs if not group_ok(o, g, det_classes)]
                share = len(miss) / len(nonempty_obs) if nonempty_obs else 0.0
                if miss and share >= share_thr:
                    missing.append((g, miss, len(miss) >= confirm_min))
                    gi["state"] = "missing"
                    gi["share"] = round(share, 2)
                else:
                    gi["state"] = "ok"
            else:
                ok_any = any(group_ok(o, g, det_classes) for o in valid)
                if ok_any:
                    gi["state"] = "ok"
                elif len(valid) >= confirm_min:
                    missing.append((g, nonempty_obs or valid, True))
                    gi["state"] = "missing"
                else:
                    gi["state"] = "pending"      # проверяется за смену: нужен ещё хотя бы один снимок
            if len(det_classes) < len(g.any_of):
                gi["limited"] = [c for c in g.any_of if c not in ctx.detectable]
            tc["groups"].append(gi)

        # 3) парные правила (пример ТЗ: экскаватор без самосвалов)
        fired_partner_sets: list[set] = []
        for comp in p.companions:
            det_partner = [c for c in comp.partner if c in ctx.detectable]
            undetected = [c for c in comp.partner if c not in ctx.detectable]
            ci = {"lead": _names(m, comp.lead), "partner": _names(m, comp.partner), "message": comp.message,
                  "window": comp.window}
            lead_obs = [o for o in valid if any(C(o).strong.get(c, 0) for c in comp.lead)]
            if not lead_obs:
                ci["state"] = "no_lead"
                tc["companions"].append(ci)
                continue

            def partner_ok(o: Obs) -> bool:
                if any(c in zw.site_equipment for c in comp.partner):
                    return True
                return any(C(o).weak.get(c, 0) for c in det_partner)

            partner_never = not any(partner_ok(o) for o in valid)
            if comp.window == "shift":
                bad = lead_obs if partner_never else []
                violated = bool(bad) and len(valid) >= confirm_min
                pending = bool(bad) and not violated
                confirmed = violated
            else:
                bad = [o for o in lead_obs if not partner_ok(o)]
                share = len(bad) / len(lead_obs)
                violated = bool(bad) and share >= share_thr
                pending = False
                confirmed = violated and len(bad) >= confirm_min
                ci["share"] = round(share, 2)
            if pending:
                ci["state"] = "pending"
            elif not violated:
                ci["state"] = "ok"
            else:
                ci["state"] = "violated"
                sev = comp.severity
                note = ""
                if undetected:
                    # в паре есть техника, которую модель пока не видит (бетононасос) — только к сведению
                    sev = "info"
                    note = f" {_names(m, undetected).capitalize()} детектор пока не распознаёт — проверьте по снимку."
                elif sev == "warning" and confirmed and partner_never:
                    sev = "critical"
                lead_here = [c for c in comp.lead if any(C(o).strong.get(c, 0) for o in bad)]
                findings.append(mk(
                    "INCOMPLETE_SET", confirmed=confirmed, severity=sev, obs_list=bad, task_id=t.id,
                    role=f"{'+'.join(comp.lead)}>{'+'.join(comp.partner)}", det_ids_for=ids_of(comp.lead),
                    note=f"{_names(m, lead_here)} без пары",
                    metrics={"lead": list(comp.lead), "partner": list(comp.partner),
                             "snapshots_with_lead": sum(len(o.snapshot_ids) for o in lead_obs),
                             "snapshots_without_partner": sum(len(o.snapshot_ids) for o in bad),
                             "share": round(len(bad) / len(lead_obs), 2), "partner_never_seen": partner_never,
                             "undetected_partner": undetected},
                    fmt={"companion_message": comp.message + note, "zone": zw.zone_name, "task": t.title,
                         "lead": _names(m, lead_here or comp.lead), "partner": _names(m, comp.partner)}))
                fired_partner_sets.append(set(comp.partner))
            tc["companions"].append(ci)

        # 4) отсутствующие группы — если их уже объяснило парное правило, отдельно не выводим
        for g, miss_obs, confirmed in missing:
            if any(set(g.any_of) <= ps for ps in fired_partner_sets):
                continue
            det_classes = [c for c in g.any_of if c in ctx.detectable]
            seen = Counter()
            for o in miss_obs:
                seen.update(zone_weak(o))
            findings.append(mk("MISSING_REQUIRED", confirmed=confirmed, obs_list=miss_obs, task_id=t.id,
                               role=g.role, det_ids_for=lambda o: [i for ids in C(o).ids.values() for i in ids],
                               metrics={"role": g.role, "expected": list(g.any_of), "min": g.min, "window": g.window,
                                        "limited": [c for c in g.any_of if c not in ctx.detectable]},
                               fmt={"task": t.title, "role": g.role, "expected": _names(m, det_classes),
                                    "observed": _observed_text(m, seen), "zone": zw.zone_name},
                               fmt_rec={"role": g.role}))

        # 5) количество техники из графика
        for c, n in t.planned.items():
            n = int(n)
            if n <= 1 or c not in ctx.detectable or c in site_wide:
                tc["planned"].append({"cls": c, "planned": n, "state": "not_checked"})
                continue
            best = max(valid, key=lambda o: C(o).weak.get(c, 0))
            seen_n = C(best).weak.get(c, 0)
            state = "ok" if seen_n >= n else ("missing" if seen_n == 0 else "short")
            tc["planned"].append({"cls": c, "planned": n, "observed": seen_n, "state": state})
            if 0 < seen_n < n:
                # доказательства — все снимки дня (ни на одном не было нужного числа), лучший — первым
                findings.append(mk("INSUFFICIENT_COUNT", confirmed=len(valid) >= confirm_min,
                                   obs_list=[best] + [o for o in valid if o is not best],
                                   task_id=t.id, cls=c, det_ids_for=ids_of([c]),
                                   metrics={"planned": n, "observed_max": seen_n},
                                   fmt={"task": t.title, "planned": n, "cls": m.class_name(c).lower(),
                                        "observed": f"не больше {seen_n}"}))

        if any(f.task_id == t.id for f in findings):
            tc["status"] = "deviation"
        elif any(gi.get("state") == "pending" for gi in tc["groups"]) or \
                any(ci.get("state") == "pending" for ci in tc["companions"]):
            tc["status"] = "pending"

    # --- лишняя техника
    allowed: set[str] = set(site_wide)
    if zw.active or zw.site_active:
        allowed |= m.delivery
    for t in zw.active + zw.site_active:
        if t.is_summary:        # сводный этап: допустимую технику задают его подэтапы
            continue
        if t.profile:
            allowed |= t.profile.classes_used()
        allowed |= set(t.planned)
    check["allowed"] = sorted(allowed)
    seen_classes = sorted({c for o in valid for c, n in C(o).strong.items() if n}) if zw.check_unexpected else []
    for c in seen_classes:
        if c in allowed:
            continue
        seen_obs = [o for o in valid if C(o).strong.get(c, 0)]
        max_conf = max(C(o).max_conf.get(c, 0.0) for o in seen_obs)
        # подтверждение: серия снимков или одна очень уверенная рамка (confirm_single_confidence)
        confirmed_by = "series" if len(seen_obs) >= confirm_min else "confidence" if max_conf >= confirm_conf else None
        confirmed = confirmed_by is not None
        # «своя» техника этапа: обязательная и ведущая по профилю плюс записанная в графике (без доставки)
        def belongs(t: TaskInfo) -> bool:
            return bool(t.profile) and not t.is_summary and c not in m.delivery and \
                (c in t.profile.core_classes() or c in t.planned)
        fut = sorted((t for t in zw.upcoming if belongs(t)), key=lambda t: t.start)
        past = sorted((t for t in zw.recent if belongs(t)), key=lambda t: t.end, reverse=True)
        ui = {"cls": c, "count": max(C(o).strong.get(c, 0) for o in seen_obs), "max_conf": round(max_conf, 2)}
        common = dict(obs_list=seen_obs, cls=c, det_ids_for=ids_of([c]), note=m.class_name(c))
        cls_name = m.class_name(c)
        if fut:
            t = fut[0]
            ui.update(verdict="EARLY_START", task=t.title, start=t.start.isoformat())
            findings.append(mk("EARLY_START", confirmed=confirmed, task_id=t.id,
                               metrics={"task_start": t.start.isoformat(), "days_before": (t.start - zw.day).days,
                                        "max_conf": round(max_conf, 2), "confirmed_by": confirmed_by},
                               fmt={"cls": cls_name, "zone": zw.zone_name, "task": t.title,
                                    "start": t.start.strftime("%d.%m.%Y")}, **common))
        elif past:
            t = past[0]
            ui.update(verdict="STAGE_OVERRUN", task=t.title, end=t.end.isoformat())
            findings.append(mk("STAGE_OVERRUN", confirmed=confirmed, task_id=t.id,
                               metrics={"task_end": t.end.isoformat(), "days_after": (zw.day - t.end).days,
                                        "max_conf": round(max_conf, 2), "confirmed_by": confirmed_by},
                               fmt={"cls": cls_name, "zone": zw.zone_name, "task": t.title,
                                    "end": t.end.strftime("%d.%m.%Y")}, **common))
        elif not zw.active:
            if c in m.delivery:
                continue
            ui.update(verdict="NO_ACTIVE_STAGE")
            findings.append(mk("NO_ACTIVE_STAGE", confirmed=confirmed,
                               metrics={"max_conf": round(max_conf, 2), "confirmed_by": confirmed_by},
                               fmt={"cls": cls_name, "zone": zw.zone_name, "date": zw.day.strftime("%d.%m.%Y")},
                               **common))
        else:
            ui.update(verdict="UNEXPECTED_EQUIPMENT")
            findings.append(mk("UNEXPECTED_EQUIPMENT", confirmed=confirmed,
                               metrics={"max_conf": round(max_conf, 2), "confirmed_by": confirmed_by,
                                        "active_tasks": [t.title for t in zw.active]},
                               fmt={"cls": cls_name, "zone": zw.zone_name, "date": zw.day.strftime("%d.%m.%Y"),
                                    "tasks": "; ".join(f"«{t.name}»" for t in zw.active[:3])}, **common))
        check["unexpected"].append(ui)

    if invalid:
        check["invalid"] = len(invalid)
    check["status"] = "deviation" if any(f.rule != "NO_DATA" for f in findings) else "ok"
    if not zw.active:
        check["status"] = "no_tasks" if check["status"] == "ok" else check["status"]
    return findings, check


# ------------------------------------------------------------------ выбор этапов для окна

def tasks_for_zone(tasks: list[TaskInfo], zone_key: str, day: dt.date, horizon_days: int, lookback_days: int
                   ) -> tuple[list[TaskInfo], list[TaskInfo], list[TaskInfo]]:
    """Идущие, предстоящие и недавно завершённые этапы зоны на дату.

    Этапы без зоны (на всю площадку) относятся к псевдозоне SITE_ZONE; для обычных зон они
    учитываются только как «допустимая техника», чтобы не требовать технику в каждой зоне.
    """
    def in_zone(t: TaskInfo) -> bool:
        return (t.zone_key or SITE_ZONE) == zone_key

    active = [t for t in tasks if in_zone(t) and t.start <= day <= t.end]
    upcoming = [t for t in tasks if in_zone(t) and day < t.start <= day + dt.timedelta(days=horizon_days)]
    recent = [t for t in tasks if in_zone(t) and day - dt.timedelta(days=lookback_days) <= t.end < day]
    return active, upcoming, recent
