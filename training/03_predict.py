#!/usr/bin/env python3
"""Шаг 3. Предсказание: стадия, готовность, график, техника, прочие объекты, доказательства → панели.

Результаты пишутся отдельно от обучения: predictions/<дата_время>/
    panels/<кадр>.png      панель: сверху легенда классов, слева фото с рамками, справа текст
    heatmaps/<кадр>.jpg    отдельно: карта внимания — куда смотрела модель
    frames/<кадр>.jpg      чистый кадр без рамок — для инструмента разметки на дашборде
    json/<кадр>.json       все числа по кадру, включая похожие кадры из обучения
    predictions.csv        сводная таблица

    python training/03_predict.py                                   # демо: 10 кадров из обучающих видео
    python training/03_predict.py --demo 20
    python training/03_predict.py photo.jpg "folder/*.jpg"          # свои фото
    python training/03_predict.py "videos for train/video3.mp4" --every 5   # кадр каждые 5 с видео
    python training/03_predict.py photo.jpg --expected 0.6          # плановая готовность для своих фото
    python training/03_predict.py --no-vlm --no-openvocab           # только DINOv3 + RF-DETR + голова
    python training/03_predict.py --no-boxes                        # панели без рамок: только фото и текст

Если на кадре есть экранная дата камеры, она распознаётся (EasyOCR) и план считается по календарю:
«план на 26.01.2022 — 20%», отклонение — в настоящих днях. Даты начала и конца съёмки каждого видео
распознаются по его первому и последнему кадру и кешируются в training/data/video_dates.json.
"""
from __future__ import annotations

import argparse
import datetime as dtm
import gc
import glob
import json
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image
from tqdm.auto import tqdm

from common import (CLASS_RU, Backbone, build_head, det_features, detect, load_config, load_detector, load_ocr, p,
                    pick_device, plan_expected, plan_status, read_frame_date, total_days)
from render import render_heatmap, render_panel, signed

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv"}


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description="Предсказание стадии и готовности стройки с панелями")
    ap.add_argument("sources", nargs="*", help="фото, папки, glob-шаблоны или видео")
    ap.add_argument("--config", default=None)
    ap.add_argument("--run", default=None, help="папка обученной головы; по умолчанию последняя")
    ap.add_argument("--demo", type=int, default=10, help="без sources: столько кадров из обучающих видео")
    ap.add_argument("--every", type=float, default=5.0, help="для видео: брать кадр каждые N секунд")
    ap.add_argument("--expected", type=float, default=None, help="плановая готовность 0–1 для своих фото")
    ap.add_argument("--out", default=None)
    ap.add_argument("--no-vlm", action="store_true")
    ap.add_argument("--no-openvocab", action="store_true")
    ap.add_argument("--no-ocr", action="store_true", help="не распознавать дату на кадре")
    ap.add_argument("--no-boxes", action="store_true", help="панели без рамок: только фото и текст")
    return ap.parse_args(argv)


# ---------------- входные кадры ----------------

def read_video_frame(path: Path, idx: int):
    cap = cv2.VideoCapture(str(path))
    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
    ok, im = cap.read()
    cap.release()
    return im if ok else None


def collect(args, cfg, frames_meta: pd.DataFrame):
    """-> список словарей: name, rgb (np.uint8), video, t, label_progress, label_stage."""
    by_name = frames_meta.set_index("image")
    items = []
    if not args.sources:
        pick = []
        per = max(1, args.demo // frames_meta["video"].nunique())
        for _, g in frames_meta.groupby("video"):
            qs = np.linspace(0.15, 0.85, per)
            pick += [g.iloc[(g["progress"] - q).abs().argmin()]["image"] for q in qs]
        sources = [str(p(cfg["paths"]["frames_dir"]) / "images" / n) for n in pick]
    else:
        sources = []
        for s in args.sources:
            path = Path(s)
            if path.is_dir():
                sources += sorted(str(x) for x in path.rglob("*") if x.suffix.lower() in IMG_EXTS)
            elif any(ch in s for ch in "*?["):
                sources += sorted(glob.glob(s, recursive=True))
            else:
                sources.append(s)

    for s in sources:
        path = Path(s)
        if path.suffix.lower() in VIDEO_EXTS:
            cap = cv2.VideoCapture(str(path))
            n, fps = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), cap.get(cv2.CAP_PROP_FPS) or 25
            last = max(n - 3, 1)
            step = max(1, round(args.every * fps))
            known = path.name in cfg["videos"]
            for f in range(0, last + 1, step):
                cap.set(cv2.CAP_PROP_POS_FRAMES, f)
                ok, im = cap.read()
                if not ok:
                    continue
                items.append({"name": f"{path.stem}_{f:06d}", "rgb": cv2.cvtColor(im, cv2.COLOR_BGR2RGB),
                              "video": path.name if known else None, "t": f / last if known else None,
                              "label_progress": f / last if known else None, "label_stage": None,
                              "train_image": None, "source": f"{path.stem} · кадр {f} из {last + 1}"})
            cap.release()
        elif path.suffix.lower() in IMG_EXTS and path.exists():
            row = by_name.loc[path.name] if path.name in by_name.index else None
            im = None
            if row is not None:
                # кадр из обучения — берём из исходного видео в полном разрешении, а не уменьшенную копию
                im = read_video_frame(p(cfg["paths"]["videos_dir"]) / row["video"], int(row["frame"]))
            if im is None:
                im = cv2.imread(str(path))
            if row is None:
                items.append({"name": path.stem, "rgb": cv2.cvtColor(im, cv2.COLOR_BGR2RGB), "video": None,
                              "t": None, "label_progress": None, "label_stage": None, "train_image": None,
                              "source": path.name})
                continue
            stem = Path(row["video"]).stem
            items.append({"name": f"{stem}_{int(row['frame']):06d}", "rgb": cv2.cvtColor(im, cv2.COLOR_BGR2RGB),
                          "video": row["video"], "t": float(row["t"]),
                          "label_progress": float(row["progress"]), "label_stage": int(row["stage"]),
                          "train_image": path.name, "source": f"{stem} · кадр {int(row['frame'])}"})
        else:
            print(f"! пропущено: {s}")
    return items


# ---------------- этапы вычислений ----------------

def free():
    """Освободить память GPU после удаления моделей (del в вызывающей функции)."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def stage_perception(items, cfg, run_dir: Path, device: str):
    """RF-DETR + DINOv3 + голова + поиск похожих обучающих кадров."""
    ck = torch.load(run_dir / "head.pt", map_location="cpu", weights_only=False)
    head = build_head(ck["dim"], ck["n_det"], len(ck["stages"]), ck["hidden"])
    head.load_state_dict(ck["state_dict"])
    head.to(device).eval()
    feat_dir = Path(ck["features_dir"])
    train_meta = pd.read_csv(feat_dir / "meta.csv")
    train_z = torch.from_numpy(np.load(run_dir / "train_embed.npy").astype(np.float32)).to(device)
    train_cls = torch.nn.functional.normalize(
        torch.from_numpy(np.load(feat_dir / "cls.npy").astype(np.float32)).to(device), dim=-1)

    detector = load_detector(cfg, device)
    class_names = list(detector.class_names)
    backbone = Backbone(cfg, device)
    thr = cfg["detector"]["threshold"]
    grid = ck["grid"]

    def infer(rgb):
        """Один кадр целиком через RF-DETR → DINOv3 → голову."""
        h, w = rgb.shape[:2]
        det = detect(detector, [rgb], thr)[0]
        cls, patches = backbone([rgb])
        dfeat = det_features(det, len(class_names), w, h)
        with torch.no_grad():
            out = head(torch.from_numpy(patches).to(device), torch.from_numpy(cls).to(device),
                       torch.from_numpy(dfeat[None]).to(device))
        return det, cls, out

    for it in tqdm(items, desc="DINOv3 + RF-DETR + голова", unit="кадр"):
        det, cls, (prog, logits, attn, z) = infer(it["rgb"])
        with torch.no_grad():
            z = torch.nn.functional.normalize(z, dim=-1)
            sims = (train_z @ z[0]).cpu().numpy()
            scene_sim = (train_cls @ torch.nn.functional.normalize(torch.from_numpy(cls).to(device), dim=-1)[0]).cpu().numpy()
        probs = logits.softmax(-1)[0].cpu().numpy()
        it.update({
            "pred_progress": float(prog[0]), "stage_probs": probs.tolist(), "stage_idx": int(probs.argmax()),
            "attention_grid": attn[0].cpu().numpy().reshape(grid).tolist(),
            "equipment_boxes": [[*map(float, b), class_names[int(c)], float(s)]
                                for b, c, s in zip(det.xyxy, det.class_id, det.confidence)],
            "max_similarity": float(scene_sim.max()),
        })
        # похожие кадры: лучшие из других видео (или любые, если кадр не из обучения), по одному на видео
        order = np.argsort(-sims)
        nbs, seen = [], set()
        for j in order:
            row = train_meta.iloc[j]
            if row["image"] == it["train_image"] or row["video"] == it["video"] or row["video"] in seen:
                continue
            seen.add(row["video"])
            nbs.append({"image": row["image"], "video": row["video"], "video_short": Path(row["video"]).stem,
                        "progress": float(row["progress"]), "stage": int(row["stage"]),
                        "sim": float(scene_sim[j])})
            if len(nbs) == 3:
                break
        it["neighbors"] = nbs

    # траектория по всему видео — тем же способом, что и сам кадр: N кадров из исходного видео
    # в полном разрешении через тот же конвейер. Прогноз по самому кадру — одна из точек кривой,
    # поэтому точка «этот кадр» всегда лежит на линии.
    n_traj = int(cfg.get("predict", {}).get("trajectory_frames", 48))
    for video in sorted({it["video"] for it in items if it["video"] and it["t"] is not None}):
        path = p(cfg["paths"]["videos_dir"]) / video
        cap = cv2.VideoCapture(str(path))
        last = max(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) - 3, 1)
        cap.release()
        pts = {}
        for f in tqdm(np.linspace(0, last, n_traj).round().astype(int), desc=f"траектория {Path(video).stem}", unit="кадр"):
            im = read_video_frame(path, int(f))
            if im is not None:
                pts[round(f / last, 4)] = float(infer(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))[2][0][0])
        own = [it for it in items if it["video"] == video and it["t"] is not None]
        for it in own:
            pts[round(it["t"], 4)] = it["pred_progress"]
        ts = sorted(pts)
        for it in own:
            it["trajectory"] = {"t": ts, "pred": [round(pts[t], 4) for t in ts]}
    del detector, backbone, head
    free()
    return ck


def stage_ocr(items, cfg, device):
    """Дата на кадре + даты начала и конца съёмки видео → план по календарю."""
    reader = load_ocr(device)
    cache_path = p(cfg["paths"]["frames_dir"]).parent / "video_dates.json"
    cache = json.loads(cache_path.read_text(encoding="utf-8")) if cache_path.exists() else {}

    def span(video):
        own = cfg["videos"].get(video) or {}
        if own.get("plan_start") and own.get("plan_end"):
            return {"start": str(own["plan_start"]), "end": str(own["plan_end"]), "source": "config"}
        if video in cache:
            return cache[video]
        path = p(cfg["paths"]["videos_dir"]) / video
        cap = cv2.VideoCapture(str(path))
        last = max(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) - 3, 1)
        cap.release()
        # ближайший к началу и к концу кадр с читаемой датой (на крайних кадрах дату часто обрезает)
        pts = []
        for fracs in ((0, 0.01, 0.03, 0.06, 0.1, 0.15, 0.2, 0.25, 0.3), (1, 0.997, 0.98, 0.95, 0.9, 0.85, 0.8, 0.75, 0.7)):
            for t in fracs:
                im = read_video_frame(path, round(t * last))
                if im is not None and (d := read_frame_date(reader, cv2.cvtColor(im, cv2.COLOR_BGR2RGB))["frame_date"]):
                    pts.append((t, dtm.date.fromisoformat(d)))
                    break
        if len(pts) == 2 and pts[1][1] > pts[0][1]:
            # таймлапс снимается равномерно: продолжаем прямую через две точки до начала и конца видео
            (t1, d1), (t2, d2) = pts
            rate = (d2 - d1) / (t2 - t1)
            cache[video] = {"start": (d1 - rate * t1).isoformat(), "end": (d1 + rate * (1 - t1)).isoformat(),
                            "source": "ocr", "extrapolated": t1 > 0 or t2 < 1}
        else:
            cache[video] = None
        cache_path.write_text(json.dumps(cache, ensure_ascii=False, indent=1), encoding="utf-8")
        return cache[video]

    for it in tqdm(items, desc="дата на кадре (OCR)", unit="кадр"):
        it.update(read_frame_date(reader, it["rgb"]))
        it["video_span"] = span(it["video"]) if it["video"] else None
    del reader
    free()


def stage_open_vocab(items, cfg, device):
    """Grounding DINO: рамки объектов по текстовым подсказкам — то, чего нет среди классов техники."""
    ov = cfg["open_vocab"]
    from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor
    proc = AutoProcessor.from_pretrained(ov["model"])
    model = AutoModelForZeroShotObjectDetection.from_pretrained(ov["model"]).to(device).eval()
    prompts = ov["prompts"]
    for it in tqdm(items, desc="Grounding DINO", unit="кадр"):
        pil = Image.fromarray(it["rgb"])
        area = pil.size[0] * pil.size[1]
        found = []
        # Каждая фраза — отдельный запрос: подпись рамки однозначно равна фразе, по которой её нашли.
        # Если спросить все фразы разом, модель возвращает кусок фразы, и подпись приходится угадывать.
        for en, val in prompts.items():
            # значение — русская подпись или {ru: подпись, threshold: свой порог для этой фразы}
            ru, thr = (val, ov["threshold"]) if isinstance(val, str) else (val["ru"], val.get("threshold", ov["threshold"]))
            inputs = proc(images=pil, text=f"{en}.", return_tensors="pt").to(device)
            with torch.no_grad():
                out = model(**inputs)
            kw = dict(text_threshold=thr, target_sizes=[pil.size[::-1]])
            try:
                res = proc.post_process_grounded_object_detection(out, inputs.input_ids, threshold=thr, **kw)[0]
            except TypeError:   # старые версии transformers
                res = proc.post_process_grounded_object_detection(out, inputs.input_ids, box_threshold=thr, **kw)[0]
            for box, score in zip(res["boxes"].tolist(), res["scores"].tolist()):
                if (box[2] - box[0]) * (box[3] - box[1]) > 0.6 * area:
                    continue                 # рамка на весь кадр — модель «нашла» фразу во всей сцене
                found.append([*box, ru, float(score)])
        # одна рамка — одна подпись: из пересекающихся рамок разных фраз оставляем самую уверенную
        found.sort(key=lambda b: -b[5])
        eq = [b[:4] for b in it["equipment_boxes"]]
        boxes = []
        for b in found:
            if any(iou(b[:4], e) > 0.5 for e in eq) or any(iou(b[:4], k[:4]) > 0.6 for k in boxes):
                continue                     # уже есть: техника RF-DETR или более уверенная рамка
            boxes.append(b)
        it["other_boxes"] = boxes
    del model, proc
    free()


def stage_vlm(items, cfg, stage_names):
    """Модель-свидетель: описывает кадр словами и независимо голосует за стадию."""
    vc = cfg["vlm"]
    from transformers import AutoModelForImageTextToText, AutoProcessor
    proc = AutoProcessor.from_pretrained(vc["model"])
    model = AutoModelForImageTextToText.from_pretrained(vc["model"], dtype=torch.bfloat16, device_map="auto").eval()
    options = "\n".join(f"{i + 1}. {n}" for i, n in enumerate(stage_names))
    q_describe = ("Ты инженер строительного надзора. Опиши этот кадр со стройплощадки в 2–3 предложениях: "
                  "на какой стадии строительство, что видно на площадке, какая техника и какие работы. "
                  "Не придумывай того, чего не видно.")
    q_stage = f"На какой стадии строительство на этом кадре?\n{options}\nОтветь одной цифрой."

    def ask(pil, question, max_new):
        msgs = [{"role": "user", "content": [{"type": "image", "image": pil}, {"type": "text", "text": question}]}]
        inputs = proc.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True, return_dict=True,
                                          return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=max_new, do_sample=False)
        return proc.batch_decode(out[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)[0].strip()

    for it in tqdm(items, desc="VLM-свидетель", unit="кадр"):
        pil = Image.fromarray(it["rgb"])
        it["vlm_text"] = ask(pil, q_describe, vc["max_new_tokens"])
        ans = ask(pil, q_stage, 5)
        digit = next((int(ch) for ch in ans if ch.isdigit()), None)
        it["vlm_stage"] = digit - 1 if digit and 1 <= digit <= len(stage_names) else None
        it["vlm_agree"] = None if it["vlm_stage"] is None else it["vlm_stage"] == it["stage_idx"]
        it["vlm_model"] = vc["model"].split("/")[-1]
    del model, proc
    free()


def iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    return inter / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter + 1e-9)


# ---------------- итог по кадру ----------------

def plan_for(it, cfg, args):
    """(плановая готовность, основа расчёта, плановая длительность в днях)."""
    if args.expected is not None:
        return args.expected, "manual", total_days(cfg, it["video"])
    sp, fd = it.get("video_span"), it.get("frame_date")
    if fd and sp:
        start, end = dtm.date.fromisoformat(sp["start"]), dtm.date.fromisoformat(sp["end"])
        finish = 1.0 if sp.get("source") == "config" else \
            float((cfg["videos"].get(it["video"]) or {}).get("plan_finish", 1.0))
        end = start + (end - start) * finish
        days = max((end - start).days, 1)
        return min(max((dtm.date.fromisoformat(fd) - start).days / days, 0.0), 1.0), "date", days
    exp = plan_expected(cfg, it["video"], it["t"])
    return exp, ("position" if exp is not None else None), total_days(cfg, it["video"])


def finalize(it, cfg, args, stage_names):
    exp, basis, plan_days = plan_for(it, cfg, args)
    delta = None if exp is None else it["pred_progress"] - exp
    status, status_ru = plan_status(cfg, delta)
    counts, conf = {}, {}
    for *_, cls, s in it["equipment_boxes"]:
        counts[cls] = counts.get(cls, 0) + 1
        conf[cls] = max(conf.get(cls, 0), s)
    other = {}
    for *_, lab, _ in it.get("other_boxes", []):
        other[lab] = other.get(lab, 0) + 1
    days = None if delta is None else delta * plan_days
    stage_name = stage_names[it["stage_idx"]]

    parts = [f"Идёт стадия «{stage_name}» (уверенность {max(it['stage_probs']):.0%})."]
    if delta is None:
        parts.append(f"Готовность объекта ≈ {it['pred_progress']:.0%}; плана для сравнения нет.")
    else:
        when = f" на {dtm.date.fromisoformat(it['frame_date']):%d.%m.%Y}" if basis == "date" else ""
        parts.append(f"Готовность ≈ {it['pred_progress']:.0%} при плане {exp:.0%}{when}: {status_ru} "
                     f"({signed(delta * 100)} п.п., ≈ {signed(days, 0)} дн.).")
    if counts:
        parts.append("В кадре: " + ", ".join(f"{CLASS_RU.get(k, k)} ×{v}" for k, v in counts.items()) + ".")
    else:
        parts.append("Строительная техника в кадре не обнаружена.")
    if other:
        parts.append("Прочие объекты: " + ", ".join(f"{k} ×{v}" for k, v in other.items()) + ".")
    if it["neighbors"]:
        nb_prog = np.mean([n["progress"] for n in it["neighbors"]])
        parts.append(f"Самые похожие кадры других объектов — в среднем {nb_prog:.0%} готовности.")
    if it["max_similarity"] < 0.5:
        parts.append("Сцена сильно отличается от обучающих — вывод менее надёжен.")

    title = it["source"]
    if it["t"] is not None:
        title += f" · {it['t']:.0%} срока съёмки"
    if it.get("frame_date"):
        title += f" · {dtm.date.fromisoformat(it['frame_date']):%d.%m.%Y}"
    it.update({
        "title": title, "expected": exp, "delta_pp": None if delta is None else delta * 100,
        "delta_days": days, "status": status, "status_ru": status_ru, "stage_name": stage_name,
        "stage_names": stage_names, "stage_prob": max(it["stage_probs"]), "equipment_counts": counts,
        "equipment_conf": conf, "other_counts": other, "other_boxes": it.get("other_boxes", []),
        "summary": " ".join(parts), "plan_basis": basis, "plan_days": plan_days,
        "plan_finish": float((cfg["videos"].get(it["video"]) or {}).get("plan_finish", 1.0)) if it["video"] else None,
    })


def main(argv=None):
    args = parse_args(argv)
    cfg = load_config(args.config)
    runs = p(cfg["paths"]["runs_dir"])
    run_dir = runs / (args.run or (runs / "latest.txt").read_text(encoding="utf-8").strip())
    frames_dir = p(cfg["paths"]["frames_dir"])
    frames_meta = pd.read_csv(frames_dir / "frames.csv")
    out = p(args.out) if args.out else p(cfg["paths"]["predictions_dir"]) / datetime.now().strftime("%Y%m%d_%H%M%S")
    for sub in ("panels", "heatmaps", "frames", "json"):
        (out / sub).mkdir(parents=True, exist_ok=True)

    items = collect(args, cfg, frames_meta)
    if not items:
        raise SystemExit("нет входных кадров")
    device = pick_device()
    print(f"кадров: {len(items)}; голова: {run_dir.name}; устройство: {device}")

    ck = stage_perception(items, cfg, run_dir, device)
    stage_names = [s["name"] for s in ck["stages"]]
    if cfg.get("ocr", {}).get("enabled", True) and not args.no_ocr:
        try:
            stage_ocr(items, cfg, device)
        except ImportError:
            print("! дата на кадре не распознаётся: pip install easyocr")
        except Exception as e:
            print(f"! OCR пропущен: {e}")
    if cfg["open_vocab"]["enabled"] and not args.no_openvocab:
        try:
            stage_open_vocab(items, cfg, device)
        except Exception as e:                       # модель не скачалась или не хватило памяти
            print(f"! Grounding DINO пропущен: {e}")
    if cfg["vlm"]["enabled"] and not args.no_vlm:
        try:
            stage_vlm(items, cfg, stage_names)
        except Exception as e:
            print(f"! VLM пропущена: {e}")

    rows = []
    for it in tqdm(items, desc="панели", unit="кадр"):
        finalize(it, cfg, args, stage_names)
        pil = Image.fromarray(it["rgb"])
        render_panel(it, pil, out / "panels" / f"{it['name']}.png", boxes=not args.no_boxes)
        render_heatmap(it, pil, out / "heatmaps" / f"{it['name']}.jpg")
        pil.save(out / "frames" / f"{it['name']}.jpg", quality=92)
        record = {k: v for k, v in it.items() if k != "rgb"}
        (out / "json" / f"{it['name']}.json").write_text(json.dumps(record, ensure_ascii=False, indent=1),
                                                         encoding="utf-8")
        rows.append({"image": it["name"], "source": it["source"], "video": it["video"],
                     "label_progress": it["label_progress"], "pred_progress": round(it["pred_progress"], 4),
                     "expected": None if it["expected"] is None else round(it["expected"], 4),
                     "delta_pp": None if it["delta_pp"] is None else round(it["delta_pp"], 2),
                     "delta_days": None if it["delta_days"] is None else round(it["delta_days"], 1),
                     "status": it["status_ru"], "plan_basis": it["plan_basis"], "frame_date": it.get("frame_date"),
                     "stage": it["stage_name"], "stage_prob": round(it["stage_prob"], 3),
                     "stage_label": None if it["label_stage"] is None else stage_names[it["label_stage"]],
                     "equipment": json.dumps(it["equipment_counts"], ensure_ascii=False),
                     "other_objects": json.dumps(it["other_counts"], ensure_ascii=False),
                     "vlm_stage": None if it.get("vlm_stage") is None else stage_names[it["vlm_stage"]],
                     "vlm_agree": it.get("vlm_agree"), "max_similarity": round(it["max_similarity"], 3)})
    df = pd.DataFrame(rows)
    df.to_csv(out / "predictions.csv", index=False, encoding="utf-8-sig")
    print(f"\nготово → {out}")
    cols = ["image", "pred_progress", "expected", "status", "stage", "stage_prob", "vlm_agree"]
    print(df[cols].to_string(index=False))


if __name__ == "__main__":
    sys.exit(main())
