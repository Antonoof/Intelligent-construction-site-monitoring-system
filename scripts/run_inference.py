#!/usr/bin/env python3
"""Инференс детектора техники на всех видео, найденных по glob-шаблонам.

На каждое видео в outputs/inference/<имя_видео>/ сохраняются:
    annotated.mp4    — видео с боксами
    detections.csv   — покадровые детекции
    summary.json     — сводка присутствия техники по классам
и общая таблица outputs/inference/summary.csv по всем обработанным видео.

    python scripts/run_inference.py                          # все видео по шаблонам из конфига
    python scripts/run_inference.py video.mp4                # одно видео
    python scripts/run_inference.py D:/cams/cam1             # вся папка (рекурсивно)
    python scripts/run_inference.py "D:/cams/**/*.mp4" --stride 5 --threshold 0.4
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import cv2
from tqdm.auto import tqdm

from construction_monitor.config import load_config, resolve_path
from construction_monitor.detector import Detector
from construction_monitor.results import Annotator, VideoReport, unique_dir, write_summary_table
from construction_monitor.video import batched, find_videos, iter_frames, make_writer, open_video


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Инференс RF-DETR на видео со стройплощадки")
    p.add_argument("--config", type=Path, default=None, help="по умолчанию configs/inference.yaml")
    p.add_argument("sources", nargs="*",
                   help="видеофайлы, папки или glob-шаблоны; если не заданы — берутся из конфига")
    p.add_argument("--videos", nargs="+", default=None, help="то же, что sources (для совместимости)")
    p.add_argument("--weights", type=Path, default=None)
    p.add_argument("--device", default=None, help="auto | cuda | cuda:0 | cpu")
    p.add_argument("--threshold", type=float, default=None)
    p.add_argument("--stride", type=int, default=None, help="обрабатывать каждый N-й кадр")
    p.add_argument("--batch", type=int, default=1, help="кадров за один вызов модели")
    p.add_argument("--optimize", action="store_true", default=None, help="JIT-трассировка модели")
    p.add_argument("--half", action="store_true", default=None, help="fp16 (только CUDA)")
    p.add_argument("--out", type=Path, default=None)
    p.add_argument("--no-video", action="store_true", help="не сохранять видео с боксами")
    p.add_argument("--no-csv", action="store_true", help="не сохранять покадровый CSV")
    p.add_argument("--show", action="store_true", help="показывать кадры в окне")
    return p.parse_args(argv)


def pick(cli, cfg, key, default):
    return cli if cli is not None else cfg.get(key, default)


def main(argv=None):
    args = parse_args(argv)
    cfg = load_config(args.config)
    m, v, o = cfg["model"], cfg["videos"], cfg["output"]

    patterns = args.sources or args.videos or v.get("patterns") or []
    stride = max(1, pick(args.stride, v, "frame_stride", 1))
    out_root = resolve_path(pick(args.out, o, "dir", "outputs/inference"))
    save_video = o.get("save_video", True) and not args.no_video
    save_csv = o.get("save_csv", True) and not args.no_csv
    show = args.show or o.get("show", False)

    videos = find_videos(patterns)
    if not videos:
        raise SystemExit("видео не найдены:\n  " + "\n  ".join(patterns))
    print(f"найдено видео: {len(videos)}")

    det = Detector(resolve_path(pick(args.weights, m, "weights", None)),
                   device=pick(args.device, m, "device", "auto"),
                   threshold=pick(args.threshold, m, "threshold", 0.5),
                   optimize=pick(args.optimize, m, "optimize", False),
                   half=pick(args.half, m, "half", False),
                   batch_size=args.batch)
    print(det.describe())
    annotate = Annotator(det.class_names)

    used = set()
    for path in videos:
        cap, info = open_video(path)
        vdir = unique_dir(out_root, path.stem, used)
        report = VideoReport(path.name, info.fps, stride, det.class_names,
                             vdir / "detections.csv" if save_csv else None)
        writer = (make_writer(vdir / "annotated.mp4", info.fps / stride, info.width, info.height)
                  if save_video else None)

        t0, stop = time.perf_counter(), False
        bar = tqdm(total=info.frames // stride or None, desc=path.name, unit="кадр", leave=False)
        try:
            for batch in batched(iter_frames(cap, stride), args.batch):
                idxs, frames = zip(*batch)
                for idx, frame, d in zip(idxs, frames, det.predict(list(frames))):
                    report.add(idx, d)
                    if writer or show:
                        vis = annotate(frame, d)
                        if writer:
                            writer.write(vis)
                        if show:
                            cv2.imshow("construction-monitor", vis)
                            if cv2.waitKey(1) & 0xFF == ord("q"):
                                stop = True
                bar.update(len(batch))
                if stop:
                    break
        finally:
            bar.close()
            cap.release()
            if writer:
                writer.release()

        s = report.close(vdir / "summary.json")
        dt = time.perf_counter() - t0
        found = ", ".join(f"{k}: {c['presence_ratio']:.0%}" for k, c in s["classes"].items()) or "ничего"
        print(f"{path.name}: {s['frames_processed']} кадров за {dt:.1f} с "
              f"({s['frames_processed'] / max(dt, 1e-9):.1f} к/с) -> {vdir}\n    {found}")

    write_summary_table(out_root)
    if show:
        cv2.destroyAllWindows()
    print(f"\nготово, результаты в {out_root}")


if __name__ == "__main__":
    sys.exit(main())
