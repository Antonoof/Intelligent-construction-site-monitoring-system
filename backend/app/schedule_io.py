"""Разбор календарного графика (XLSX или CSV).

Столбцы определяются по заголовкам, а не по порядку: «Код», «Шифр», «Наименование работ»,
«Захватка», «Начало», «Окончание», «Техника»… Строка графика сопоставляется с видом работ
справочника организаторов по коду, а если кода нет — по названию (methodology.match_work_type).
Иерархия: строка, у которой в графике есть подэтапы (по коду или по уровню группировки Excel),
считается сводной и не проверяется сама — проверяются её подэтапы.
"""
from __future__ import annotations

import csv
import datetime as dt
import io
import re
from dataclasses import dataclass, field

import openpyxl

from .methodology import Methodology

HEADERS = {
    "wbs": ["код", "шифр", "wbs", "№", "№ п/п", "номер", "код работ", "код этапа", "п/п"],
    "work_type": ["вид работ (справочник)", "код справочника", "id вида работ", "вид работ id", "код вида работ"],
    "name": ["наименование", "наименование работ", "наименование этапа", "этап", "работа", "работы",
             "название", "вид работ", "содержание работ"],
    "zone": ["зона", "захватка", "участок", "секция", "место работ", "локация", "зона работ"],
    "start": ["начало", "дата начала", "старт", "план начало", "начало план", "плановое начало", "начало работ"],
    "end": ["окончание", "дата окончания", "конец", "завершение", "финиш", "план окончание", "окончание план",
            "плановое окончание", "окончание работ"],
    "equipment": ["техника", "потребность в технике", "механизмы", "машины", "техника на смену", "техника (план)"],
    "contractor": ["подрядчик", "исполнитель", "субподрядчик"],
}


def _norm(s) -> str:
    return re.sub(r"\s+", " ", str(s or "").strip().lower().replace("ё", "е")).strip(" :.")


@dataclass
class ParsedTask:
    row_no: int
    wbs: str
    name: str
    zone: str
    start: dt.date
    end: dt.date
    planned: dict[str, int] = field(default_factory=dict)
    contractor: str = ""
    work_type_hint: str = ""
    outline: int = 0
    is_summary: bool = False
    parent_row: int | None = None
    work_type_id: str | None = None
    match_method: str = "none"
    match_score: float = 0.0


@dataclass
class ParseResult:
    tasks: list[ParsedTask]
    columns: dict[str, str]
    warnings: list[str]
    sheet: str = ""


def parse_date(v) -> dt.date | None:
    if v is None or v == "":
        return None
    if isinstance(v, dt.datetime):
        return v.date()
    if isinstance(v, dt.date):
        return v
    if isinstance(v, (int, float)):            # серийный номер даты Excel
        if 20000 < v < 80000:
            return (dt.datetime(1899, 12, 30) + dt.timedelta(days=float(v))).date()
        return None
    s = str(v).strip()
    for fmt in ("%d.%m.%Y", "%d.%m.%y", "%Y-%m-%d", "%d/%m/%Y", "%Y.%m.%d", "%d-%m-%Y"):
        try:
            return dt.datetime.strptime(s[:10], fmt).date()
        except ValueError:
            continue
    return None


_NUM = re.compile(r"(?:[×xх*]\s*(\d+)|(\d+)\s*(?:шт|ед|×|x|х)|[-–—:]\s*(\d+)\s*$|\((\d+)\)|^\s*(\d+)\s)", re.I)


def parse_equipment(text: str, m: Methodology) -> dict[str, int]:
    """«Экскаватор ×2; самосвал - 3; каток» → {"excavator": 2, "dump_truck": 3, "roller": 1}."""
    out: dict[str, int] = {}
    if not text:
        return out
    for part in re.split(r"[;,\n/+]| и ", str(text)):
        part = part.strip()
        if not part:
            continue
        n = 1
        mm = _NUM.search(part)
        if mm:
            n = int(next(g for g in mm.groups() if g))
        label = re.sub(r"[\d×*()\-–—:.]+|шт|ед", " ", part, flags=re.I).strip()
        cls = m.normalize_class(label)
        if not cls:                                  # «автобетоносмеситель 8 м3» и т. п.
            low = _norm(label)
            for key, v in m.classes.items():
                names = [v["name"]] + list(v.get("aliases", []))
                if any(_norm(a) and _norm(a) in low for a in names if len(_norm(a)) > 3):
                    cls = key
                    break
        if cls:
            out[cls] = out.get(cls, 0) + n
    return out


def _map_headers(row: list) -> dict[str, int]:
    found: dict[str, int] = {}
    for idx, cell in enumerate(row):
        h = _norm(cell)
        if not h:
            continue
        for key, variants in HEADERS.items():
            if key in found:
                continue
            if h in variants or any(h.startswith(v) for v in variants if len(v) > 4):
                # «вид работ» — название, если нет отдельного столбца «наименование»
                found[key] = idx
                break
    return found


def _rows_from_xlsx(data: bytes) -> tuple[list[list], list[int], str]:
    wb = openpyxl.load_workbook(io.BytesIO(data), data_only=True)
    best = None
    for ws in wb.worksheets:
        rows, outline = [], []
        for r in range(1, ws.max_row + 1):
            rows.append([ws.cell(r, c).value for c in range(1, min(ws.max_column, 40) + 1)])
            outline.append(int(ws.row_dimensions[r].outlineLevel or 0))
        score = max((len(_map_headers(rw)) for rw in rows[:20]), default=0)
        if best is None or score > best[0]:
            best = (score, rows, outline, ws.title)
    return best[1], best[2], best[3]


def _rows_from_csv(data: bytes) -> tuple[list[list], list[int], str]:
    text = data.decode("utf-8-sig", errors="replace")
    dialect = csv.Sniffer().sniff(text[:4096], delimiters=";,\t")
    rows = [list(r) for r in csv.reader(io.StringIO(text), dialect)]
    return rows, [0] * len(rows), "csv"


def parse_schedule(data: bytes, filename: str, m: Methodology, object_type: str | None = None) -> ParseResult:
    if filename.lower().endswith((".csv", ".txt")):
        rows, outline, sheet = _rows_from_csv(data)
    else:
        rows, outline, sheet = _rows_from_xlsx(data)
    warnings: list[str] = []
    header_idx, cols = None, {}
    for i, row in enumerate(rows[:25]):
        mapped = _map_headers(row)
        if "name" in mapped and "start" in mapped and "end" in mapped:
            header_idx, cols = i, mapped
            break
    if header_idx is None:
        raise ValueError("Не найдена строка заголовков: нужны столбцы «Наименование», «Начало» и «Окончание».")
    header = rows[header_idx]
    tasks: list[ParsedTask] = []
    for i in range(header_idx + 1, len(rows)):
        row = rows[i]

        def cell(key):
            j = cols.get(key)
            return row[j] if j is not None and j < len(row) else None

        name = str(cell("name") or "").strip()
        if not name:
            continue
        start, end = parse_date(cell("start")), parse_date(cell("end"))
        if not start or not end:
            warnings.append(f"строка {i + 1}: «{name[:50]}» — нет дат начала/окончания, пропущена")
            continue
        if end < start:
            warnings.append(f"строка {i + 1}: окончание раньше начала — даты переставлены")
            start, end = end, start
        wbs_raw = cell("wbs")
        if isinstance(wbs_raw, (dt.date, dt.datetime)):       # Excel превратил «10.1» в дату
            wbs_raw = f"{wbs_raw.day}.{wbs_raw.month}"
        wbs = str(wbs_raw).strip().rstrip(".") if wbs_raw not in (None, "") else ""
        if wbs.endswith(".0"):
            wbs = wbs[:-2]
        tasks.append(ParsedTask(
            row_no=i + 1, wbs=wbs, name=" ".join(name.split()), zone=str(cell("zone") or "").strip(),
            start=start, end=end, planned=parse_equipment(cell("equipment"), m),
            contractor=str(cell("contractor") or "").strip(),
            work_type_hint=str(cell("work_type") or "").strip(), outline=outline[i] if i < len(outline) else 0))

    # иерархия: по кодам (12.3 — родитель 12.3.1) и по группировке строк Excel
    by_wbs = {t.wbs: t for t in tasks if t.wbs}
    for t in tasks:
        if t.wbs:
            parts = re.split(r"[.\-]", t.wbs)
            for k in range(len(parts) - 1, 0, -1):
                cand = ".".join(parts[:k])
                if cand in by_wbs and by_wbs[cand] is not t:
                    t.parent_row = by_wbs[cand].row_no
                    by_wbs[cand].is_summary = True
                    break
    for a, b in zip(tasks, tasks[1:]):
        if b.outline > a.outline:
            a.is_summary = True
            if b.parent_row is None:
                b.parent_row = a.row_no

    # сопоставление со справочником видов работ
    for t in tasks:
        hint = t.work_type_hint or t.wbs
        wt, how, score = m.match_work_type(hint, t.name, object_type)
        if wt is None and t.work_type_hint:
            wt, how, score = m.match_work_type(t.wbs, t.name, object_type)
        t.work_type_id, t.match_method, t.match_score = wt, how, score
        if wt is None:
            warnings.append(f"строка {t.row_no}: «{t.name[:60]}» не сопоставлена со справочником — "
                            "укажите вид работ вручную")
    columns = {k: str(header[j]) for k, j in cols.items()}
    return ParseResult(tasks, columns, warnings, sheet)
