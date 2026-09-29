"""Сборка датасета детектора v2 из открытых источников (training/datasets.yaml) в формат COCO для RF-DETR.

Детектор v1 из ML-части распознаёт 10 классов; ТЗ требует ещё каток и грузовик. Скрипт:
  1. читает разметку каждого источника (YOLO, COCO или VOC — структура папок не важна);
  2. приводит имена классов к классам ОКО (synonyms из methodology/equipment.yaml + map в datasets.yaml);
  3. убирает дубликаты между источниками (SHA-256 и перцептивный хеш 8×8);
  4. размечает условия съёмки: освещённость (день / сумерки / ночь), зима (снег), дальний план (мелкие объекты);
  5. делит на train / valid / test со стратификацией по классу и условиям — тестовая выборка по ТЗ
     формируется самостоятельно и должна покрывать освещённость, сезон и виды техники;
  6. пишет datasets/oko_v2/{train,valid,test}/_annotations.coco.json + снимки, manifest.csv и stats.md.

Запуск:  python training/05_build_detector_dataset.py --raw datasets/raw --out datasets/oko_v2
Источники кладутся в datasets/raw/<id из datasets.yaml>/ (распакованные архивы).
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import shutil
import sys
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import yaml
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def norm(s: str) -> str:
    return " ".join(str(s).lower().replace("ё", "е").replace("_", " ").replace("-", " ").split())


def load_aliases() -> dict[str, str]:
    eq = yaml.safe_load((ROOT / "methodology" / "equipment.yaml").read_text(encoding="utf-8"))["classes"]
    out = {}
    for key, v in eq.items():
        for a in [key, v["name"], v.get("name_en", "")] + list(v.get("aliases", [])):
            if a:
                out[norm(a)] = key
    return out


@dataclass
class Item:
    source: str
    path: Path
    width: int
    height: int
    boxes: list[tuple[str, float, float, float, float]] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)


class Mapper:
    def __init__(self, src: dict, aliases: dict, targets: list[str]):
        self.map = {norm(k): v for k, v in (src.get("map") or {}).items()}
        self.aliases, self.targets = aliases, set(targets)
        self.unmapped: Counter = Counter()

    def __call__(self, name: str) -> str | None:
        n = norm(name)
        cls = self.map.get(n) or self.aliases.get(n)
        if cls == "skip" or cls is None or cls not in self.targets:
            self.unmapped[name] += 1
            return None
        return cls


# ------------------------------------------------------------------ читатели разметки

def _yolo_names(root: Path, src: dict) -> list[str]:
    if src.get("classes"):
        return list(src["classes"])
    for p in list(root.rglob("data.yaml")) + list(root.rglob("*.yaml")):
        try:
            d = yaml.safe_load(p.read_text(encoding="utf-8"))
        except Exception:
            continue
        if isinstance(d, dict) and "names" in d:
            names = d["names"]
            return [names[k] for k in sorted(names)] if isinstance(names, dict) else list(names)
    for fn in ("classes.txt", "obj.names", "_darknet.labels"):
        for p in root.rglob(fn):
            return [ln.strip() for ln in p.read_text(encoding="utf-8").splitlines() if ln.strip()]
    raise SystemExit(f"{src['id']}: не найдены имена классов YOLO — укажите classes в datasets.yaml")


def read_yolo(root: Path, src: dict, mapper: Mapper) -> list[Item]:
    names = _yolo_names(root, src)
    labels = {}
    for t in root.rglob("*.txt"):
        if t.name in ("classes.txt", "README.txt"):
            continue
        labels[(t.parent.name, t.stem)] = t
        labels.setdefault(("*", t.stem), t)
    items = []
    for img in root.rglob("*"):
        if img.suffix.lower() not in IMG_EXT:
            continue
        # images/xxx.jpg ↔ labels/xxx.txt или разметка рядом со снимком
        lab = labels.get(("labels", img.stem)) or labels.get((img.parent.name, img.stem)) or labels.get(("*", img.stem))
        if lab is None:
            continue
        with Image.open(img) as im:
            w, h = im.size
        it = Item(src["id"], img, w, h)
        for line in lab.read_text(encoding="utf-8").splitlines():
            parts = line.split()
            if len(parts) < 5:
                continue
            k, cx, cy, bw, bh = int(float(parts[0])), *map(float, parts[1:5])
            if k >= len(names):
                continue
            cls = mapper(names[k])
            if cls:
                it.boxes.append((cls, (cx - bw / 2) * w, (cy - bh / 2) * h, (cx + bw / 2) * w, (cy + bh / 2) * h))
        items.append(it)
    return items


def read_coco(root: Path, src: dict, mapper: Mapper) -> list[Item]:
    items = []
    for js in root.rglob("*.json"):
        try:
            d = json.loads(js.read_text(encoding="utf-8"))
        except Exception:
            continue
        if not isinstance(d, dict) or "images" not in d or "annotations" not in d:
            continue
        cats = {c["id"]: c["name"] for c in d.get("categories", [])}
        by_img = defaultdict(list)
        for a in d["annotations"]:
            by_img[a["image_id"]].append(a)
        for im in d["images"]:
            p = js.parent / im["file_name"]
            if not p.exists():
                continue
            it = Item(src["id"], p, im["width"], im["height"])
            for a in by_img[im["id"]]:
                cls = mapper(cats.get(a["category_id"], ""))
                if cls:
                    x, y, w, h = a["bbox"]
                    it.boxes.append((cls, x, y, x + w, y + h))
            items.append(it)
    return items


def read_voc(root: Path, src: dict, mapper: Mapper) -> list[Item]:
    items = []
    for xml in root.rglob("*.xml"):
        try:
            ann = ET.parse(xml).getroot()
        except ET.ParseError:
            continue
        if ann.tag != "annotation":
            continue
        fname = ann.findtext("filename") or xml.stem
        cands = [xml.parent / fname] + [xml.with_suffix(e) for e in IMG_EXT] + \
                [p for p in root.rglob(Path(fname).name)]
        img = next((c for c in cands if c.exists() and c.suffix.lower() in IMG_EXT), None)
        if img is None:
            continue
        size = ann.find("size")
        if size is not None and size.findtext("width"):
            w, h = int(float(size.findtext("width"))), int(float(size.findtext("height")))
        else:
            with Image.open(img) as im:
                w, h = im.size
        it = Item(src["id"], img, w, h)
        for obj in ann.iter("object"):
            cls = mapper(obj.findtext("name", ""))
            bb = obj.find("bndbox")
            if cls and bb is not None:
                it.boxes.append((cls, *(float(bb.findtext(k)) for k in ("xmin", "ymin", "xmax", "ymax"))))
        items.append(it)
    return items


READERS = {"yolo": read_yolo, "coco": read_coco, "voc": read_voc}


# ------------------------------------------------------------------ условия съёмки и дубликаты

def dhash(img: Image.Image) -> str:
    """Разностный хеш 8×8: одинаков у одного и того же снимка после пережатия или смены размера."""
    g = np.asarray(img.convert("L").resize((9, 8), Image.BILINEAR), dtype=np.float32)
    bits = (g[:, 1:] > g[:, :-1]).flatten()
    return "".join("1" if b else "0" for b in bits)


def condition_tags(img: Image.Image, it: Item) -> list[str]:
    """Освещённость, снег, дальний план — для стратификации и отчёта по условиям съёмки."""
    small = img.convert("RGB").resize((160, max(1, round(160 * img.height / img.width))))
    a = np.asarray(small, dtype=np.float32) / 255.0
    lum = (0.2126 * a[..., 0] + 0.7152 * a[..., 1] + 0.0722 * a[..., 2]).mean()
    tags = ["night" if lum < 0.22 else "dusk" if lum < 0.36 else "day"]
    lower = a[a.shape[0] // 2:]
    mx, mn = lower.max(axis=2), lower.min(axis=2)
    snow = ((mx > 0.78) & ((mx - mn) < 0.08)).mean()
    tags.append("winter" if snow > 0.25 else "no_snow")
    if it.boxes:
        areas = [(x2 - x1) * (y2 - y1) / (it.width * it.height) for _, x1, y1, x2, y2 in it.boxes]
        tags.append("far" if max(areas) < 0.01 else "near")
    return tags


# ------------------------------------------------------------------ сборка

def build(raw: Path, out: Path, cfg: dict, copy: bool = True) -> dict:
    aliases = load_aliases()
    targets = cfg["target_classes"]
    all_items: list[Item] = []
    report = {}
    for src in cfg["sources"]:
        root = raw / src["id"]
        if not root.exists():
            print(f"  пропуск {src['id']}: нет папки {root}")
            continue
        mapper = Mapper(src, aliases, targets)
        items = READERS[src["format"]](root, src, mapper)
        items = [i for i in items if i.boxes]
        report[src["id"]] = {"images": len(items), "boxes": sum(len(i.boxes) for i in items),
                             "unmapped": dict(mapper.unmapped.most_common(20))}
        print(f"  {src['id']}: снимков {len(items)}, рамок {report[src['id']]['boxes']}"
              + (f", не взяты классы: {dict(mapper.unmapped.most_common(6))}" if mapper.unmapped else ""))
        all_items.extend(items)

    # дубликаты: точные (SHA-256) — везде; перцептивные (тот же снимок, пережатый в другом датасете) —
    # только между разными источниками и при одинаковых пропорциях кадра
    seen_sha: set[str] = set()
    seen_hash: dict[tuple, str] = {}
    uniq = []
    for it in all_items:
        sha = hashlib.sha256(it.path.read_bytes()).hexdigest()
        with Image.open(it.path) as im:
            im.load()
            key = (dhash(im), round(it.width / it.height, 2))
            it.tags = condition_tags(im, it)
        if sha in seen_sha or (key in seen_hash and seen_hash[key] != it.source):
            continue
        seen_sha.add(sha)
        seen_hash.setdefault(key, it.source)
        uniq.append(it)
    print(f"  после удаления дубликатов: {len(uniq)} из {len(all_items)}")

    # стратификация: редчайший класс снимка × освещённость
    freq = Counter(c for it in uniq for c, *_ in it.boxes)
    groups = defaultdict(list)
    for it in uniq:
        primary = min({c for c, *_ in it.boxes}, key=lambda c: freq[c])
        groups[(primary, it.tags[0])].append(it)
    rnd = random.Random(cfg["split"]["seed"])
    split_of = {}
    for key in sorted(groups):
        g = groups[key]
        rnd.shuffle(g)
        n = len(g)
        n_test = max(1, round(n * cfg["split"]["test"])) if n >= 3 else 0
        n_val = max(1, round(n * cfg["split"]["valid"])) if n >= 3 else 0
        for i, it in enumerate(g):
            split_of[id(it)] = "test" if i < n_test else "valid" if i < n_test + n_val else "train"

    cats = [{"id": i + 1, "name": c, "supercategory": "equipment"} for i, c in enumerate(targets)]
    cat_id = {c["name"]: c["id"] for c in cats}
    coco = {s: {"images": [], "annotations": [], "categories": cats} for s in ("train", "valid", "test")}
    if out.exists():
        shutil.rmtree(out)
    rows = []
    ann_id = 1
    for k, it in enumerate(uniq):
        s = split_of[id(it)]
        d = out / s
        d.mkdir(parents=True, exist_ok=True)
        name = f"{it.source}_{k:06d}{it.path.suffix.lower()}"
        (shutil.copy2 if copy else _link)(it.path, d / name)
        coco[s]["images"].append({"id": k, "file_name": name, "width": it.width, "height": it.height})
        for cls, x1, y1, x2, y2 in it.boxes:
            x1, y1 = max(0.0, x1), max(0.0, y1)
            x2, y2 = min(float(it.width), x2), min(float(it.height), y2)
            if x2 - x1 < 2 or y2 - y1 < 2:
                continue
            coco[s]["annotations"].append({"id": ann_id, "image_id": k, "category_id": cat_id[cls],
                                           "bbox": [round(x1, 1), round(y1, 1), round(x2 - x1, 1), round(y2 - y1, 1)],
                                           "area": round((x2 - x1) * (y2 - y1), 1), "iscrowd": 0})
            ann_id += 1
        rows.append([f"{s}/{name}", s, it.source, " ".join(it.tags), " ".join(sorted({c for c, *_ in it.boxes})),
                     str(it.path)])
    for s, d in coco.items():
        (out / s).mkdir(parents=True, exist_ok=True)
        (out / s / "_annotations.coco.json").write_text(json.dumps(d, ensure_ascii=False), encoding="utf-8")
    with open(out / "manifest.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["file", "split", "source", "tags", "classes", "original"])
        w.writerows(rows)
    stats = write_stats(out, coco, rows, targets)
    return {"sources": report, "images": len(uniq), "stats": stats}


def _link(a: Path, b: Path):
    try:
        b.hardlink_to(a)
    except OSError:
        shutil.copy2(a, b)


def write_stats(out: Path, coco: dict, rows: list, targets: list[str]) -> dict:
    per = {s: Counter() for s in coco}
    for s, d in coco.items():
        cid = {c["id"]: c["name"] for c in d["categories"]}
        for a in d["annotations"]:
            per[s][cid[a["category_id"]]] += 1
    tags = {s: Counter() for s in coco}
    for r in rows:
        for t in r[3].split():
            tags[r[1]][t] += 1
    lines = ["# Датасет детектора v2", "", "| класс | train | valid | test |", "|---|---:|---:|---:|"]
    for c in targets:
        lines.append(f"| {c} | {per['train'][c]} | {per['valid'][c]} | {per['test'][c]} |")
    lines += ["", "| условие съёмки | train | valid | test |", "|---|---:|---:|---:|"]
    for t in ("day", "dusk", "night", "winter", "no_snow", "near", "far"):
        lines.append(f"| {t} | {tags['train'][t]} | {tags['valid'][t]} | {tags['test'][t]} |")
    (out / "stats.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    missing = [c for c in targets if per["test"][c] == 0]
    if missing:
        print(f"  ВНИМАНИЕ: в тестовой выборке нет классов {missing} — добавьте снимки")
    return {s: dict(v) for s, v in per.items()}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--raw", default=str(ROOT / "datasets" / "raw"))
    ap.add_argument("--out", default=str(ROOT / "datasets" / "oko_v2"))
    ap.add_argument("--config", default=str(ROOT / "training" / "datasets.yaml"))
    ap.add_argument("--link", action="store_true", help="жёсткие ссылки вместо копирования снимков")
    a = ap.parse_args(argv)
    cfg = yaml.safe_load(Path(a.config).read_text(encoding="utf-8"))
    res = build(Path(a.raw), Path(a.out), cfg, copy=not a.link)
    print(f"готово: {res['images']} снимков → {a.out} (статистика — stats.md)")
    return res


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
