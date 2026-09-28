"""Сборка методики «вид работ → техника» из справочника организаторов.

Вход:
  methodology/source/Справочник видов работ (организаторы).xlsx — справочник видов работ;
  methodology/equipment.yaml   — классы техники;
  methodology/profiles.yaml    — профили техники;
  methodology/assignments.yaml — привязка видов работ к профилям и стадиям.
Выход:
  methodology/work_types.yaml          — 377 видов работ с профилем, стадией, наблюдаемостью;
  methodology/profiles_composite.yaml  — составные профили для групп справочника;
  methodology/Методика_этап_техника.xlsx — справочник организаторов + столбцы методики;
  methodology/work_types.md            — та же таблица для просмотра на GitHub.

Запуск из корня репозитория:  python tools/build_methodology.py
Скрипт проверяет, что каждому виду работ назначены профиль и стадия, а в профилях нет
неизвестных классов техники, и падает с понятной ошибкой, если это не так.
"""
from __future__ import annotations

import copy
import sys
from collections import OrderedDict
from pathlib import Path

import openpyxl
import yaml
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.utils import get_column_letter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
from parse_directory import OBJECT_TYPES, parse  # noqa: E402

M = ROOT / "methodology"
SRC_XLSX = M / "source" / "Справочник видов работ (организаторы).xlsx"
OBS_ORDER = {"none": 0, "low": 1, "medium": 2, "high": 3}
OBS_RU = {"high": "высокая", "medium": "средняя", "low": "низкая", "none": "не наблюдается"}
STAGE_RU = {
    "S1": "S1 подготовка и котлован", "S2": "S2 фундамент и подземная часть", "S3": "S3 каркас",
    "S4": "S4 фасад и инженерные системы", "S5": "S5 отделка и благоустройство",
}


def load_yaml(p: Path):
    return yaml.safe_load(p.read_text(encoding="utf-8"))


def fail(msg: str):
    raise SystemExit(f"ОШИБКА методики: {msg}")


def validate_profiles(profiles: dict, classes: dict):
    for pid, p in profiles.items():
        for key in ("name", "observability"):
            if key not in p:
                fail(f"профиль {pid}: нет поля {key}")
        if p["observability"] not in OBS_ORDER:
            fail(f"профиль {pid}: неизвестная наблюдаемость {p['observability']}")
        used = set(p.get("allowed", [])) | set(p.get("not_detected", []))
        for g in p.get("required", []):
            used |= set(g["any_of"])
            if g.get("window", "snapshot") not in ("snapshot", "shift"):
                fail(f"профиль {pid}: окно группы {g}")
        for c in p.get("companions", []):
            used |= set(c["lead"]) | set(c["partner"])
        unknown = used - set(classes)
        if unknown:
            fail(f"профиль {pid}: неизвестные классы {sorted(unknown)}")


def composite(pid: str, name: str, children_profiles: list[dict], level: int) -> dict:
    """Составной профиль группы: объединение техники подпунктов.

    Если в графике есть только строка-группа (без подпунктов), этап считается идущим, когда на
    площадке за смену появилась хоть какая-то техника его подпунктов.
    """
    req_classes: list[str] = []
    allowed: list[str] = []
    companions: list[dict] = []
    not_detected: list[str] = []
    obs = "none"
    for p in children_profiles:
        for g in p.get("required", []):
            for c in g["any_of"]:
                if c not in req_classes:
                    req_classes.append(c)
        for c in p.get("allowed", []):
            if c not in allowed:
                allowed.append(c)
        for comp in p.get("companions", []):
            key = (tuple(comp["lead"]), tuple(comp["partner"]))
            if key not in {(tuple(x["lead"]), tuple(x["partner"])) for x in companions}:
                companions.append(copy.deepcopy(comp))
        for c in p.get("not_detected", []):
            if c not in not_detected:
                not_detected.append(c)
        if OBS_ORDER[p["observability"]] > OBS_ORDER[obs]:
            obs = p["observability"]
    # у широких групп (уровень 1–2) и у групп без «видимых» подпунктов проверяем только лишнюю технику
    if level <= 1:
        obs = "low"
    elif obs in ("high", "medium"):
        obs = "medium"
    allowed = [c for c in allowed if c not in req_classes]
    prof = {
        "name": f"Составной: {name}",
        "description": "Группа справочника: объединение техники подпунктов. Этап идёт, если за смену видна техника хотя бы одного подпункта.",
        "observability": obs,
        "required": ([{"role": "техника одного из подпунктов", "any_of": req_classes, "min": 1, "window": "shift"}]
                     if req_classes and obs in ("high", "medium") else []),
        "companions": [dict(c, severity="info") for c in companions],
        "allowed": allowed + ([c for c in req_classes] if not (req_classes and obs in ("high", "medium")) else []),
        "not_detected": not_detected,
        "composite_of": [],
    }
    return prof


def fmt_group(g: dict, classes: dict) -> str:
    names = " или ".join(classes[c]["name"].lower() for c in g["any_of"])
    names = names[0].upper() + names[1:]
    win = " (за смену)" if g.get("window") == "shift" else ""
    n = g.get("min", 1)
    return f"{names}{' ×' + str(n) if n > 1 else ''}{win} — {g['role']}"


def fmt_companion(c: dict, classes: dict) -> str:
    lead = " или ".join(classes[x]["name"].lower() for x in c["lead"])
    partner = " или ".join(classes[x]["name"].lower() for x in c["partner"])
    return f"{lead} → {partner}"


def req_text(prof: dict, obs: str, classes: dict, sep: str) -> str:
    """Обязательная техника для таблиц; при низкой наблюдаемости — только справочно."""
    txt = sep.join(fmt_group(g, classes) for g in prof.get("required", []))
    if not txt:
        return "—"
    return f"справочно (не проверяется): {txt}" if obs in ("low", "none") else txt


def main():
    eq = load_yaml(M / "equipment.yaml")
    classes = eq["classes"]
    profiles = load_yaml(M / "profiles.yaml")["profiles"]
    validate_profiles(profiles, classes)
    asg = load_yaml(M / "assignments.yaml")
    stages_map: dict = {str(k): v for k, v in asg["stages"].items()}
    prof_map: dict = {}
    for k, v in asg["profiles"].items():
        prof_map[str(k)] = v if isinstance(v, dict) else {"profile": v}
    for k, v in prof_map.items():
        if v["profile"] not in profiles:
            fail(f"вид работ {k}: неизвестный профиль {v['profile']}")

    rows = parse(SRC_XLSX)
    by_id = OrderedDict((r["id"], r) for r in rows)
    unknown_ids = (set(prof_map) | set(stages_map)) - set(by_id)
    if unknown_ids:
        fail(f"в assignments.yaml есть id, которых нет в справочнике: {sorted(unknown_ids)}")
    children: dict[str, list[str]] = {}
    for r in rows:
        if r["parent"]:
            children.setdefault(r["parent"], []).append(r["id"])

    def ancestors(i):
        p = by_id[i]["parent"]
        while p:
            yield p
            p = by_id[p]["parent"]

    # 1) стадия
    for r in rows:
        st = stages_map.get(r["id"])
        if st is None:
            st = next((stages_map[a] for a in ancestors(r["id"]) if a in stages_map), None)
        r["stage"] = st

    # 2) профиль: явный → унаследованный → составной (снизу вверх)
    composites: dict[str, dict] = {}

    def resolve(i: str) -> str:
        r = by_id[i]
        if "profile" in r:
            return r["profile"]
        if i in prof_map:
            r["profile"] = prof_map[i]["profile"]
            r["profile_source"] = "явно"
        else:
            inh = next((a for a in ancestors(i) if a in prof_map), None)
            if inh and i not in children:
                r["profile"] = prof_map[inh]["profile"]
                r["profile_source"] = f"от {inh}"
            elif inh and i in children:
                # группа внутри явно размеченной группы — берём профиль предка
                r["profile"] = prof_map[inh]["profile"]
                r["profile_source"] = f"от {inh}"
            elif i in children:
                kids = [resolve(c) for c in children[i]]
                kid_profiles = [profiles[k] if k in profiles else composites[k] for k in kids]
                cid = f"COMPOSITE:{i}"
                cp = composite(cid, r["name"], kid_profiles, r["level"])
                cp["composite_of"] = sorted(set(kids))
                composites[cid] = cp
                r["profile"] = cid
                r["profile_source"] = "составной"
            else:
                fail(f"виду работ {i} «{r['name']}» не назначен профиль")
        return r["profile"]

    for r in rows:
        resolve(r["id"])
    # явные профили у групп тоже наследуются вниз, если у подпункта нет своего
    for r in rows:
        prof = profiles.get(r["profile"]) or composites[r["profile"]]
        ov = prof_map.get(r["id"], {}).get("observability")
        r["observability"] = ov or prof["observability"]
        if ov:
            r["observability_source"] = "замечание/переопределение"
    # стадия обязательна для всех, кроме корней 10 и 12 (несколько стадий)
    for r in rows:
        if r["stage"] is None and r["level"] > 1:
            fail(f"виду работ {r['id']} не назначена стадия")

    # ---------- вывод YAML
    wt_out = []
    for r in rows:
        item = OrderedDict()
        item["id"] = r["id"]
        item["code"] = r["code"]
        item["name"] = r["name"]
        item["level"] = r["level"]
        item["parent"] = r["parent"]
        item["stage"] = r["stage"]
        item["profile"] = r["profile"]
        item["observability"] = r["observability"]
        item["object_types"] = r["object_types"]
        if r.get("expert_note"):
            item["expert_note"] = r["expert_note"]
        item["has_children"] = r["id"] in children
        wt_out.append(dict(item))
    header = ("# СГЕНЕРИРОВАНО tools/build_methodology.py из справочника организаторов и assignments.yaml.\n"
              "# Не редактируйте вручную: меняйте assignments.yaml / profiles.yaml и пересоберите.\n")
    (M / "work_types.yaml").write_text(
        header + yaml.safe_dump({"version": asg["version"], "object_types": [
            {"key": k, "name": n} for k, n in OBJECT_TYPES], "work_types": wt_out},
            allow_unicode=True, sort_keys=False, width=120), encoding="utf-8")
    (M / "profiles_composite.yaml").write_text(
        header + yaml.safe_dump({"version": asg["version"], "profiles": composites},
                                allow_unicode=True, sort_keys=False, width=120), encoding="utf-8")

    all_profiles = dict(profiles)
    all_profiles.update(composites)

    # ---------- расширенный XLSX: исходный лист + столбцы методики, листы профилей и правил
    wb = openpyxl.load_workbook(SRC_XLSX)
    ws = wb.worksheets[0]
    ws.title = "Справочник + методика"
    head_fill = PatternFill("solid", fgColor="520978")
    head_font = Font(bold=True, color="FFFFFF")
    new_cols = ["ID вида работ", "Стадия", "Профиль техники", "Наблюдаемость камерой",
                "Обязательная техника", "Парные правила (ведущая → парная)", "Допустимая техника",
                "Нужна, но пока не распознаётся", "Как задан профиль"]
    start_col = 13  # M
    for j, h in enumerate(new_cols):
        c = ws.cell(2, start_col + j, h)
        c.fill = head_fill
        c.font = head_font
        c.alignment = Alignment(wrap_text=True, vertical="center")
        ws.column_dimensions[get_column_letter(start_col + j)].width = [12, 16, 30, 14, 48, 40, 36, 20, 14][j]
    ws.cell(1, start_col, "Методика «этап работ → необходимая техника» (ОКО)").font = Font(bold=True, color="520978", size=12)
    row_by_excel = {r["row"]: r for r in rows}
    for excel_row, r in row_by_excel.items():
        prof = all_profiles[r["profile"]]
        vals = [
            r["id"], STAGE_RU.get(r["stage"], "несколько стадий"),
            f'{prof["name"]} [{r["profile"]}]', OBS_RU[r["observability"]],
            req_text(prof, r["observability"], classes, "\n"),
            "\n".join(fmt_companion(c, classes) for c in prof.get("companions", [])) or "—",
            ", ".join(classes[c]["name"] for c in prof.get("allowed", [])) or "—",
            ", ".join(classes[c]["name"] for c in prof.get("not_detected", [])) or "—",
            r.get("profile_source", ""),
        ]
        for j, v in enumerate(vals):
            c = ws.cell(excel_row, start_col + j, v)
            c.alignment = Alignment(wrap_text=True, vertical="top")
    ws.freeze_panes = "C3"

    # лист профилей
    wp = wb.create_sheet("Профили техники")
    hdr = ["Профиль", "Название", "Описание", "Наблюдаемость", "Обязательная техника",
           "Парные правила", "Допустимая техника", "Нужна, но пока не распознаётся", "Видов работ"]
    usage: dict[str, int] = {}
    for r in rows:
        usage[r["profile"]] = usage.get(r["profile"], 0) + 1
    for j, h in enumerate(hdr, 1):
        c = wp.cell(1, j, h)
        c.fill, c.font = head_fill, head_font
    for i, (pid, p) in enumerate(all_profiles.items(), 2):
        vals = [pid, p["name"], p.get("description", ""), OBS_RU[p["observability"]],
                "\n".join(fmt_group(g, classes) for g in p.get("required", [])) or "—",
                "\n".join(f'{fmt_companion(c, classes)}: {c.get("message", "")}' for c in p.get("companions", [])) or "—",
                ", ".join(classes[c]["name"] for c in p.get("allowed", [])) or "—",
                ", ".join(classes[c]["name"] for c in p.get("not_detected", [])) or "—",
                usage.get(pid, 0)]
        for j, v in enumerate(vals, 1):
            wp.cell(i, j, v).alignment = Alignment(wrap_text=True, vertical="top")
    for j, w in enumerate([22, 30, 50, 14, 50, 60, 36, 20, 10], 1):
        wp.column_dimensions[get_column_letter(j)].width = w
    wp.freeze_panes = "B2"

    # лист классов
    wc = wb.create_sheet("Классы техники")
    for j, h in enumerate(["Класс", "Название", "В перечне ТЗ", "Модели детектора", "Общеплощадочная", "Синонимы в датасетах"], 1):
        c = wc.cell(1, j, h)
        c.fill, c.font = head_fill, head_font
    for i, (k, v) in enumerate(classes.items(), 2):
        for j, val in enumerate([k, v["name"], "да" if v.get("tz") else "—", ", ".join(v["models"]),
                                 "да" if v.get("site_wide") else "—", ", ".join(v.get("aliases", []))], 1):
            wc.cell(i, j, val)
    for j, w in enumerate([18, 22, 12, 16, 14, 70], 1):
        wc.column_dimensions[get_column_letter(j)].width = w

    # лист правил отклонений
    rules = load_yaml(M / "rules.yaml")
    wr = wb.create_sheet("Правила отклонений")
    for j, h in enumerate(["Код", "Отклонение", "Тип", "Важность (1 снимок)", "Важность (подтверждено)", "Текст предупреждения", "Рекомендация"], 1):
        c = wr.cell(1, j, h)
        c.fill, c.font = head_fill, head_font
    sev = {k: v["name"] for k, v in rules["severity_levels"].items()}
    for i, (k, v) in enumerate(rules["rules"].items(), 2):
        vals = [k, v["title"], v["kind"], sev[v["severity"]], sev.get(v.get("confirmed_severity", v["severity"])),
                v["message"], v["recommendation"]]
        for j, val in enumerate(vals, 1):
            wr.cell(i, j, val).alignment = Alignment(wrap_text=True, vertical="top")
    for j, w in enumerate([20, 30, 12, 14, 16, 70, 60], 1):
        wr.column_dimensions[get_column_letter(j)].width = w
    wb.save(M / "Методика_этап_техника.xlsx")

    # ---------- markdown-таблица
    lines = ["# Методика: виды работ справочника → профили техники", "",
             "Сгенерировано `tools/build_methodology.py`. Профили — в [profiles.yaml](profiles.yaml), "
             "правила отклонений — в [rules.yaml](rules.yaml), объяснение — в [README.md](README.md).", "",
             "| ID | Вид работ | Стадия | Профиль | Наблюдаемость | Обязательная техника |",
             "|---|---|---|---|---|---|"]
    for r in rows:
        prof = all_profiles[r["profile"]]
        req = req_text(prof, r["observability"], classes, "; ")
        indent = "&nbsp;&nbsp;" * (r["level"] - 1)
        name = r["name"].replace("|", "/")
        lines.append(f'| {r["id"]} | {indent}{name} | {r["stage"] or "—"} | {prof["name"]} | {OBS_RU[r["observability"]]} | {req} |')
    (M / "work_types.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    # ---------- сводка
    from collections import Counter
    obs_count = Counter(r["observability"] for r in rows)
    prof_count = Counter(r["profile"] for r in rows)
    print(f"видов работ: {len(rows)}; профилей: {len(profiles)} + составных {len(composites)}")
    print("наблюдаемость:", dict(obs_count))
    print("топ профилей:", prof_count.most_common(8))


if __name__ == "__main__":
    main()
