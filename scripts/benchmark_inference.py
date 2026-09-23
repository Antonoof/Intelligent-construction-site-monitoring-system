#!/usr/bin/env python3
"""Замер скорости детектора на видео, найденных по glob-шаблонам.

Время считается по этапам:
    decode — чтение и декодирование кадра (OpenCV)
    infer  — предобработка + модель + постобработка (model.predict, результат уже на CPU,
             так что GPU синхронизирован)
    draw   — отрисовка боксов (только с --draw)
Перед замером модель прогревается, прогревочные кадры в статистику не входят.
Отчёт: outputs/benchmark/benchmark_<дата>.json и .csv

    python scripts/benchmark_inference.py
    python scripts/benchmark_inference.py --optimize --half --batch 4 --max-frames 500
"""
from __future__ import annotations

import argparse
import csv
import json
import platform
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import torch
from tqdm.auto import tqdm

from construction_monitor.config import load_config, resolve_path
from construction_monitor.detector import Detector
from construction_monitor.results import Annotator
from construction_monitor.video import find_videos, iter_frames, open_video


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Бенчмарк RF-DETR на видео")
    p.add_argument("--config", type=Path, default=None, help="по умолчанию configs/inference.yaml")
    p.add_argument("sources", nargs="*",
                   help="видеофайлы, папки или glob-шаблоны; если не заданы — берутся из конфига")
    p.add_argument("--videos", nargs="+", default=None, help="то же, что sources (для совместимости)")
    p.add_argument("--weights", type=Path, default=None)
    p.add_argument("--device", default=None, help="auto | cuda | cuda:0 | cpu")
    p.add_argument("--threshold", type=float, default=None)
    p.add_argument("--batch", type=int, default=None, help="кадров за один вызов модели")
    p.add_argument("--warmup", type=int, default=None, help="прогревочных вызовов модели")
    p.add_argument("--max-frames", type=int, default=None, help="кадров на видео; 0 — все")
    p.add_argument("--optimize", action="store_true", default=None, help="JIT-трассировка модели")
    p.add_argument("--half", action="store_true", default=None, help="fp16 (только CUDA)")
    p.add_argument("--draw", action="store_true", help="мерить и отрисовку боксов")
    p.add_argument("--out", type=Path, default=None)
    return p.parse_args(argv)


def pick(cli, cfg, key, default):
    return cli if cli is not None else cfg.get(key, default)


def sync(device: str):
    if device.startswith("cuda"):
        torch.cuda.synchronize(device)


def stats(ms_per_frame: list[float]) -> dict:
    if not ms_per_frame:
        return {}
    a = np.asarray(ms_per_frame)
    return {"mean_ms": round(a.mean(), 2), "p50_ms": round(np.percentile(a, 50), 2),
            "p95_ms": round(np.percentile(a, 95), 2), "min_ms": round(a.min(), 2),
            "max_ms": round(a.max(), 2)}


def fps(total_ms: float, frames: int) -> float:
    return round(frames / (total_ms / 1000), 2) if total_ms > 0 else 0.0


def main(argv=None):
    args = parse_args(argv)
    cfg = load_config(args.config)
    m, v, b = cfg["model"], cfg["videos"], cfg["benchmark"]

    patterns = args.sources or args.videos or v.get("patterns") or []
    batch = max(1, pick(args.batch, b, "batch_size", 1))
    warmup = max(0, pick(args.warmup, b, "warmup_frames", 20))
    max_frames = pick(args.max_frames, b, "max_frames", 300) or 0
    out_root = resolve_path(pick(args.out, b, "dir", "outputs/benchmark"))

    videos = find_videos(patterns)
    if not videos:
        raise SystemExit("видео не найдены:\n  " + "\n  ".join(patterns))

    t_load = time.perf_counter()
    det = Detector(resolve_path(pick(args.weights, m, "weights", None)),
                   device=pick(args.device, m, "device", "auto"),
                   threshold=pick(args.threshold, m, "threshold", 0.5),
                   optimize=pick(args.optimize, m, "optimize", False),
                   half=pick(args.half, m, "half", False),
                   batch_size=batch)
    load_s = time.perf_counter() - t_load
    print(f"{det.describe()}\nзагрузка модели: {load_s:.1f} с, видео: {len(videos)}, батч: {batch}")
    annotate = Annotator(det.class_names) if args.draw else None
    if det.device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats(det.device)

    # прогрев на первом кадре первого видео: cuDNN-автотюнинг, JIT, выделение памяти
    cap, _ = open_video(videos[0])
    ok, first = cap.read()
    cap.release()
    if not ok:
        raise SystemExit(f"не удалось прочитать кадр из {videos[0]}")
    for _ in tqdm(range(warmup), desc="прогрев", leave=False):
        det.predict([first] * batch)
    sync(det.device)

    per_video, all_infer, totals = [], [], {"decode": 0.0, "infer": 0.0, "draw": 0.0, "frames": 0}
    for path in videos:
        cap, info = open_video(path)
        frames_it = iter_frames(cap)
        limit = max_frames or info.frames or None
        t = {"decode": 0.0, "infer": 0.0, "draw": 0.0}
        infer_ms, boxes, n = [], 0, 0
        bar = tqdm(total=limit, desc=path.name, unit="кадр", leave=False)
        while limit is None or n < limit:
            t0 = time.perf_counter()
            frames = []
            for _, f in frames_it:
                frames.append(f)
                if len(frames) == batch or (limit and n + len(frames) >= limit):
                    break
            t1 = time.perf_counter()
            if not frames:
                break
            dets = det.predict(frames)
            sync(det.device)
            t2 = time.perf_counter()
            if annotate:
                for f, d in zip(frames, dets):
                    annotate(f, d)
            t3 = time.perf_counter()

            t["decode"] += (t1 - t0) * 1000
            t["infer"] += (t2 - t1) * 1000
            t["draw"] += (t3 - t2) * 1000
            infer_ms += [(t2 - t1) * 1000 / len(frames)] * len(frames)
            boxes += sum(len(d) for d in dets)
            n += len(frames)
            bar.update(len(frames))
        bar.close()
        cap.release()
        if not n:
            print(f"! {path.name}: не прочитано ни одного кадра, пропущено")
            continue

        e2e = t["decode"] + t["infer"] + t["draw"]
        row = {"video": path.name, "resolution": f"{info.width}x{info.height}", "frames": n,
               "boxes_per_frame": round(boxes / n, 2),
               "infer_fps": fps(t["infer"], n), "end_to_end_fps": fps(e2e, n),
               "decode_ms": round(t["decode"] / n, 2), "draw_ms": round(t["draw"] / n, 2),
               **{f"infer_{k}": val for k, val in stats(infer_ms).items()}}
        per_video.append(row)
        all_infer += infer_ms
        for k in t:
            totals[k] += t[k]
        totals["frames"] += n
        print(f"{path.name:<40} {n:>6} кадров  модель {row['infer_fps']:>7.2f} к/с "
              f"(p50 {row['infer_p50_ms']:.1f} мс, p95 {row['infer_p95_ms']:.1f} мс)  "
              f"сквозной {row['end_to_end_fps']:>7.2f} к/с")

    if not per_video:
        raise SystemExit("ни одно видео не удалось прочитать")

    N = totals["frames"]
    e2e = totals["decode"] + totals["infer"] + totals["draw"]
    overall = {"frames": N, "infer_fps": fps(totals["infer"], N), "end_to_end_fps": fps(e2e, N),
               "decode_ms": round(totals["decode"] / N, 2), "draw_ms": round(totals["draw"] / N, 2),
               **{f"infer_{k}": val for k, val in stats(all_infer).items()}}
    env = {"model": type(det.model).__name__, "resolution": det.resolution, "device": det.device,
           "gpu": torch.cuda.get_device_name(det.device) if det.device.startswith("cuda") else None,
           "cpu": platform.processor(), "torch": torch.__version__,
           "batch": batch, "optimize": bool(det.traced_batch), "half": det.half,
           "threshold": det.threshold, "warmup": warmup, "draw": bool(annotate),
           "model_load_s": round(load_s, 2),
           "peak_gpu_mem_mb": (round(torch.cuda.max_memory_allocated(det.device) / 2**20)
                               if det.device.startswith("cuda") else None)}

    print("\nитого:")
    print(f"  кадров: {N}")
    print(f"  модель: {overall['infer_fps']} к/с, задержка на кадр mean {overall['infer_mean_ms']} мс, "
          f"p50 {overall['infer_p50_ms']} мс, p95 {overall['infer_p95_ms']} мс")
    print(f"  декодирование: {overall['decode_ms']} мс/кадр"
          + (f", отрисовка: {overall['draw_ms']} мс/кадр" if annotate else ""))
    print(f"  сквозной конвейер: {overall['end_to_end_fps']} к/с")
    if env["peak_gpu_mem_mb"] is not None:
        print(f"  пик памяти GPU: {env['peak_gpu_mem_mb']} МБ")

    out_root.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    (out_root / f"benchmark_{stamp}.json").write_text(
        json.dumps({"env": env, "overall": overall, "videos": per_video}, ensure_ascii=False, indent=2),
        encoding="utf-8")
    with (out_root / f"benchmark_{stamp}.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(per_video[0]))
        w.writeheader()
        w.writerows(per_video)
    print(f"\nотчёт: {out_root / f'benchmark_{stamp}.json'}")


if __name__ == "__main__":
    sys.exit(main())
