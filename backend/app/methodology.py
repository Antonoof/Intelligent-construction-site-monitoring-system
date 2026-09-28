"""Методика «этап работ → необходимая техника»: загрузка из methodology/*.yaml.

Методика — данные, а не код: классы техники, профили, привязка 377 видов работ справочника
организаторов и правила отклонений лежат в YAML и собираются tools/build_methodology.py.
Здесь они загружаются в неизменяемые структуры, с которыми работает движок (engine.py),
и синхронизируются в справочные таблицы БД для API и отчётов.
"""
from __future__ import annotations

import difflib
import re
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import yaml

from .config import settings


# ------------------------------------------------------------------ структуры

@dataclass(frozen=True)
class Group:
    """Группа обязательной техники: достаточно любого класса из any_of в количестве min."""
    role: str
    any_of: tuple[str, ...]
    min: int = 1
    window: str = "snapshot"          # snapshot | shift


@dataclass(frozen=True)
class Companion:
    """Парное правило: при ведущей технике должна быть парная (экскаватор → самосвал)."""
    lead: tuple[str, ...]
    partner: tuple[str, ...]
    severity: str = "warning"
    message: str = ""
    window: str = "snapshot"


@dataclass(frozen=True)
class Profile:
    id: str
    name: str
    description: str
    observability: str
    required: tuple[Group, ...] = ()
    companions: tuple[Companion, ...] = ()
    allowed: frozenset = frozenset()
    not_detected: frozenset = frozenset()
    is_composite: bool = False

    def classes_used(self) -> set[str]:
        """Вся техника, которая «принадлежит» этапу: обязательная, парная, допустимая."""
        s = set(self.allowed) | set(self.not_detected)
        for g in self.required:
            s |= set(g.any_of)
        for c in self.companions:
            s |= set(c.lead) | set(c.partner)
        return s

    def core_classes(self) -> set[str]:
        """Техника, характерная для этапа (обязательная и ведущая в парных правилах)."""
        s: set[str] = set()
        for g in self.required:
            s |= set(g.any_of)
        for c in self.companions:
            s |= set(c.lead)
        return s


@dataclass(frozen=True)
class WorkType:
    id: str
    code: str | None
    name: str
    level: int
    parent: str | None
    stage: str | None
    profile: str
    observability: str
    object_types: tuple[str, ...]
    expert_note: str | None
    has_children: bool


@dataclass(frozen=True)
class Rule:
    code: str
    title: str
    kind: str
    severity: str
    confirmed_severity: str
    message: str
    recommendation: str


@dataclass
class Methodology:
    version: str
    rules_version: str
    classes: dict[str, dict]
    object_types: list[dict]
    profiles: dict[str, Profile]
    work_types: dict[str, WorkType]
    rules: dict[str, Rule]
    config: dict                                   # detection/windows/quality из rules.yaml
    _alias: dict[str, str] = field(default_factory=dict)

    # ---------------- классы техники
    def class_name(self, key: str | None) -> str:
        if not key:
            return ""
        return self.classes.get(key, {}).get("name", key)

    def normalize_class(self, raw: str) -> str | None:
        """Название класса из датасета или чекпойнта → ключ класса (или None, если класс не наш)."""
        return self._alias.get(_norm(raw))

    @property
    def site_wide(self) -> set[str]:
        return set(self.config.get("site_wide_classes", [])) | {k for k, v in self.classes.items() if v.get("site_wide")}

    @property
    def delivery(self) -> set[str]:
        return set(self.config.get("delivery_classes", []))

    def presence_threshold(self, cls: str) -> float:
        d = self.config.get("detection", {})
        return float(d.get("per_class_presence", {}).get(cls, d.get("presence_threshold", 0.5)))

    def weak_threshold(self) -> float:
        return float(self.config.get("detection", {}).get("weak_presence_threshold", 0.3))

    # ---------------- виды работ
    def profile_for_work_type(self, wt_id: str | None) -> Profile | None:
        if not wt_id or wt_id not in self.work_types:
            return None
        return self.profiles[self.work_types[wt_id].profile]

    def match_work_type(self, code: str | None, name: str | None, object_type: str | None = None
                        ) -> tuple[str | None, str, float]:
        """Сопоставить строку календарного графика с видом работ справочника.

        1) по коду («12.3.7-1», «12.3.1», «12.3.1.»); 2) по названию — нормализованные слова +
        difflib, с предпочтением видов работ, применимых к типу объекта. Возвращает
        (id вида работ, способ: code|name|none, оценка 0..1).
        """
        if code:
            c = str(code).strip().rstrip(".")
            if c.endswith(".0"):
                c = c[:-2]
            if c in self.work_types:
                return c, "code", 1.0
            # подпункты без кода в справочнике имеют id «12.3.4-6»; в графике их пишут «12.3.4.6»
            parts = c.split(".")
            if len(parts) > 1:
                cand = ".".join(parts[:-1]) + "-" + parts[-1]
                if cand in self.work_types and (not name or _similar(name, self.work_types[cand].name) >= 0.5):
                    return cand, "code", 0.95
        if not name:
            return None, "none", 0.0
        target = _stem(name)
        tset = set(target.split())
        if not tset:
            return None, "none", 0.0
        best, best_score = None, 0.0
        for wt in self.work_types.values():          # порядок справочника: при равенстве — первый
            cand = _stem(wt.name)
            cset = set(cand.split())
            if not cset:
                continue
            common = len(tset & cset)
            score = (0.35 * difflib.SequenceMatcher(None, target, cand).ratio()
                     + 0.25 * common / len(tset | cset)     # сходство наборов слов
                     + 0.20 * common / len(cset)            # название справочника покрыто запросом
                     + 0.20 * common / len(tset))           # запрос покрыт названием справочника
            if object_type and wt.object_types and object_type not in wt.object_types:
                score -= 0.04
            if wt.has_children:
                score -= 0.02          # при равенстве предпочитаем конкретный вид работ группе
            if score > best_score:
                best, best_score = wt.id, score
        if best and best_score >= 0.6:
            return best, "name", round(best_score, 3)
        return None, "none", round(best_score, 3)


def _norm(s: str) -> str:
    return re.sub(r"[\s_\-]+", " ", str(s).strip().lower().replace("ё", "е"))


_STOP = {"устройство", "в", "т", "ч", "и", "с", "по", "на", "из", "для", "под", "работы", "работ"}


def _norm_words(s: str) -> str:
    s = _norm(s)
    s = re.sub(r"[^a-zа-я0-9 ]+", " ", s)
    return " ".join(w for w in s.split() if w not in _STOP)


_SYNONYMS = {
    "стройплощадки": "строительной площадки", "стройплощадка": "строительная площадка",
    "стройплощадке": "строительной площадке", "ж/б": "железобетонных", "жб": "железобетонных",
    "асфальтирование": "асфальтобетонное покрытие", "бытовки": "бытового городка",
}


def _stem(s: str) -> str:
    """Грубая нормализация русских слов для сопоставления названий: синонимы + первые 5 букв."""
    s = _norm(s)
    for a, b in _SYNONYMS.items():
        s = s.replace(a, b)
    return " ".join(w[:5] for w in _norm_words(s).split())


def _similar(a: str, b: str) -> float:
    ta, tb = set(_stem(a).split()), set(_stem(b).split())
    return len(ta & tb) / len(ta | tb) if ta and tb else 0.0


def _load_yaml(p: Path):
    return yaml.safe_load(p.read_text(encoding="utf-8"))


def _mk_profile(pid: str, d: dict, composite: bool) -> Profile:
    return Profile(
        id=pid, name=d["name"], description=d.get("description", ""), observability=d["observability"],
        required=tuple(Group(role=g["role"], any_of=tuple(g["any_of"]), min=int(g.get("min", 1)),
                             window=g.get("window", "snapshot")) for g in d.get("required", [])),
        companions=tuple(Companion(lead=tuple(c["lead"]), partner=tuple(c["partner"]),
                                   severity=c.get("severity", "warning"), message=c.get("message", ""),
                                   window=c.get("window", "snapshot")) for c in d.get("companions", [])),
        allowed=frozenset(d.get("allowed", [])), not_detected=frozenset(d.get("not_detected", [])),
        is_composite=composite,
    )


def load_methodology(directory: Path | None = None) -> Methodology:
    d = Path(directory or settings.methodology_dir)
    eq = _load_yaml(d / "equipment.yaml")
    prof = _load_yaml(d / "profiles.yaml")
    comp = _load_yaml(d / "profiles_composite.yaml")
    wts = _load_yaml(d / "work_types.yaml")
    rules = _load_yaml(d / "rules.yaml")
    classes = eq["classes"]
    profiles = {pid: _mk_profile(pid, p, False) for pid, p in prof["profiles"].items()}
    profiles.update({pid: _mk_profile(pid, p, True) for pid, p in (comp.get("profiles") or {}).items()})
    work_types = {}
    for w in wts["work_types"]:
        work_types[w["id"]] = WorkType(
            id=w["id"], code=w.get("code"), name=w["name"], level=w["level"], parent=w.get("parent"),
            stage=w.get("stage"), profile=w["profile"], observability=w["observability"],
            object_types=tuple(w.get("object_types") or ()), expert_note=w.get("expert_note"),
            has_children=bool(w.get("has_children")))
    rr = {}
    for code, r in rules["rules"].items():
        rr[code] = Rule(code=code, title=r["title"], kind=r["kind"], severity=r["severity"],
                        confirmed_severity=r.get("confirmed_severity", r["severity"]),
                        message=r["message"], recommendation=r["recommendation"])
    alias = {}
    for key, v in classes.items():
        alias[_norm(key)] = key
        alias[_norm(v["name"])] = key
        if v.get("name_en"):
            alias[_norm(v["name_en"])] = key
        for a in v.get("aliases", []):
            alias[_norm(a)] = key
    cfg = {k: rules[k] for k in ("detection", "site_wide_classes", "delivery_classes", "quality", "windows",
                                 "severity_levels") if k in rules}
    m = Methodology(version=str(wts.get("version", prof.get("version", ""))), rules_version=str(rules["version"]),
                    classes=classes, object_types=wts.get("object_types", []), profiles=profiles,
                    work_types=work_types, rules=rr, config=cfg)
    m._alias = alias
    _validate(m)
    return m


def _validate(m: Methodology) -> None:
    for wt in m.work_types.values():
        if wt.profile not in m.profiles:
            raise ValueError(f"вид работ {wt.id}: нет профиля {wt.profile}")
    for p in m.profiles.values():
        unknown = p.classes_used() - set(m.classes)
        if unknown:
            raise ValueError(f"профиль {p.id}: неизвестные классы {unknown}")


@lru_cache(maxsize=1)
def get_methodology() -> Methodology:
    return load_methodology()


# ------------------------------------------------------------------ синхронизация в БД

def sync_to_db(session, m: Methodology | None = None) -> None:
    """Записать справочники методики в БД (идемпотентно, по версии)."""
    from . import models as M
    m = m or get_methodology()
    for key, v in m.classes.items():
        session.merge(M.EquipmentClass(key=key, name=v["name"], name_en=v.get("name_en", ""),
                                       in_tz=bool(v.get("tz")), site_wide=bool(v.get("site_wide")),
                                       models=list(v.get("models", [])), color=v.get("color", "#888888")))
    for ot in m.object_types:
        session.merge(M.ObjectType(key=ot["key"], name=ot["name"]))
    for p in m.profiles.values():
        session.merge(M.EquipmentProfile(
            id=p.id, name=p.name, description=p.description, observability=p.observability,
            required=[{"role": g.role, "any_of": list(g.any_of), "min": g.min, "window": g.window} for g in p.required],
            companions=[{"lead": list(c.lead), "partner": list(c.partner), "severity": c.severity,
                         "message": c.message, "window": c.window} for c in p.companions],
            allowed=sorted(p.allowed), not_detected=sorted(p.not_detected), is_composite=p.is_composite,
            methodology_version=m.version))
    session.flush()
    # родитель должен существовать раньше потомка — справочник уже упорядочен сверху вниз
    for wt in m.work_types.values():
        session.merge(M.WorkType(id=wt.id, code=wt.code, name=wt.name, level=wt.level, parent_id=wt.parent,
                                 stage=wt.stage, profile_id=wt.profile, observability=wt.observability,
                                 object_types=list(wt.object_types), expert_note=wt.expert_note,
                                 has_children=wt.has_children))
    for r in m.rules.values():
        session.merge(M.DeviationRule(code=r.code, title=r.title, kind=r.kind, severity=r.severity,
                                      confirmed_severity=r.confirmed_severity, message=r.message,
                                      recommendation=r.recommendation, rules_version=m.rules_version))
    session.flush()
