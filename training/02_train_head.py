#!/usr/bin/env python3
"""Шаг 2. Обучение головы: прогресс 0–1 (регрессия) и стадия (классификация) по кешу признаков.

DINOv3 и RF-DETR заморожены, обучается только голова (~5 млн параметров для ViT-7B): минуты на 4090.
Результат — training/runs/<дата_время>/:
    head.pt               веса головы и всё, что нужно для предсказания
    history.csv           лосс, ошибка прогресса и точность стадии по эпохам
    curves.png            графики обучения
    train_predictions.csv предсказания на всех обучающих кадрах
    train_embed.npy       эмбеддинги кадров — для поиска похожих кадров при объяснении

    python training/02_train_head.py
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from common import build_head, load_config, p, pick_device


def load_cache(feat_dir: Path, device: str):
    patch_dtype = torch.float16 if device.startswith("cuda") else torch.float32
    patches = torch.from_numpy(np.load(feat_dir / "patches.npy", mmap_mode="r")[:]).to(device, patch_dtype)
    cls = torch.from_numpy(np.load(feat_dir / "cls.npy")).to(device, torch.float32)
    det = torch.from_numpy(np.load(feat_dir / "det.npy")).to(device, torch.float32)
    return patches, cls, det


def run_head(head, patches, cls, det, idx, bs=512):
    """Предсказания и эмбеддинги для индексов idx, без градиентов."""
    head.eval()
    outs = []
    with torch.no_grad():
        for i in range(0, len(idx), bs):
            b = idx[i:i + bs]
            prog, logits, _, z = head(patches[b].float(), cls[b], det[b])
            outs.append((prog, logits.softmax(-1), F.normalize(z, dim=-1)))
    return [torch.cat(t) for t in zip(*outs)]


def main(argv=None):
    ap = argparse.ArgumentParser(description="Обучение головы стадии и прогресса")
    ap.add_argument("--config", default=None)
    ap.add_argument("--epochs", type=int, default=None)
    args = ap.parse_args(argv)
    cfg = load_config(args.config)
    hc = cfg["head"]
    epochs = args.epochs or hc["epochs"]
    torch.manual_seed(42)

    feat_dir = p(cfg["paths"]["features_dir"])
    info = json.loads((feat_dir / "info.json").read_text(encoding="utf-8"))
    meta = pd.read_csv(feat_dir / "meta.csv")
    device = pick_device()
    patches, cls, det = load_cache(feat_dir, device)
    y_prog = torch.tensor(meta["progress"].values, dtype=torch.float32, device=device)
    y_stage = torch.tensor(meta["stage"].values, dtype=torch.long, device=device)

    hold = hc.get("holdout_video")
    is_hold = torch.tensor((meta["video"] == hold).values if hold else np.zeros(len(meta), bool), device=device)
    train_idx = torch.nonzero(~is_hold).squeeze(1)
    hold_idx = torch.nonzero(is_hold).squeeze(1)
    n_stages = len(cfg["stages"])
    print(f"кадров: обучение {len(train_idx)}, отложено {len(hold_idx)} ({hold or 'нет'}); "
          f"признаки: {info['backbone']} dim {info['dim']}, {patches.shape[1]} патчей; устройство {device}")

    head = build_head(info["dim"], det.shape[1], n_stages, hc["hidden"]).to(device)
    n_params = sum(t.numel() for t in head.parameters())
    opt = torch.optim.AdamW(head.parameters(), lr=hc["lr"], weight_decay=hc["weight_decay"])
    steps_per_epoch = math.ceil(len(train_idx) / hc["batch_size"])
    total, warm = epochs * steps_per_epoch, steps_per_epoch
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1, (s + 1) / warm) * 0.5 * (1 + math.cos(math.pi * min(s / total, 1))))

    history, t0 = [], time.perf_counter()
    for ep in range(1, epochs + 1):
        head.train()
        perm = train_idx[torch.randperm(len(train_idx), device=device)]
        sums = np.zeros(4)
        for i in range(0, len(perm), hc["batch_size"]):
            b = perm[i:i + hc["batch_size"]]
            prog, logits, _, _ = head(patches[b].float(), cls[b], det[b], hc["token_dropout"])
            l_prog = F.smooth_l1_loss(prog, y_prog[b], beta=0.05)
            l_stage = F.cross_entropy(logits, y_stage[b], label_smoothing=0.05)
            loss = l_prog + hc["stage_weight"] * l_stage
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
            opt.step()
            sched.step()
            k = len(b)
            sums += np.array([loss.item(), (prog - y_prog[b]).abs().sum().item(),
                              (logits.argmax(-1) == y_stage[b]).sum().item(), k]) * [k, 1, 1, 1]
        row = {"epoch": ep, "loss": sums[0] / sums[3], "mae_pp": sums[1] / sums[3] * 100,
               "stage_acc": sums[2] / sums[3], "lr": sched.get_last_lr()[0]}
        if len(hold_idx):
            prog, probs, _ = run_head(head, patches, cls, det, hold_idx)
            row["holdout_mae_pp"] = (prog - y_prog[hold_idx]).abs().mean().item() * 100
            row["holdout_stage_acc"] = (probs.argmax(-1) == y_stage[hold_idx]).float().mean().item()
        history.append(row)
        if ep == 1 or ep % 5 == 0 or ep == epochs:
            extra = (f" | отложено: MAE {row['holdout_mae_pp']:.1f} п.п., стадия {row['holdout_stage_acc']:.1%}"
                     if len(hold_idx) else "")
            print(f"эпоха {ep:>3}/{epochs}  лосс {row['loss']:.4f}  MAE {row['mae_pp']:.2f} п.п.  "
                  f"стадия {row['stage_acc']:.1%}{extra}")

    run = p(cfg["paths"]["runs_dir"]) / datetime.now().strftime("%Y%m%d_%H%M%S")
    run.mkdir(parents=True, exist_ok=True)
    all_idx = torch.arange(len(meta), device=device)
    prog, probs, z = run_head(head, patches, cls, det, all_idx)
    pred = meta.copy()
    pred["pred_progress"] = prog.cpu().numpy().round(4)
    pred["pred_stage"] = probs.argmax(-1).cpu().numpy()
    pred["stage_prob"] = probs.max(-1).values.cpu().numpy().round(4)
    pred["abs_err_pp"] = ((pred["pred_progress"] - pred["progress"]).abs() * 100).round(2)
    pred.to_csv(run / "train_predictions.csv", index=False)
    np.save(run / "train_embed.npy", z.cpu().numpy().astype(np.float16))
    pd.DataFrame(history).to_csv(run / "history.csv", index=False)
    torch.save({"state_dict": head.state_dict(), "dim": info["dim"], "n_det": det.shape[1], "hidden": hc["hidden"],
                "stages": cfg["stages"], "class_names": info["class_names"], "backbone": info["backbone"],
                "grid": info["grid"], "image_size": info["image_size"], "features_dir": str(feat_dir),
                "holdout_video": hold, "params": n_params}, run / "head.pt")
    (p(cfg["paths"]["runs_dir"]) / "latest.txt").write_text(run.name, encoding="utf-8")
    plot_curves(pd.DataFrame(history), run / "curves.png")

    by_video = pred.groupby("video").agg(mae_pp=("abs_err_pp", "mean"),
                                         stage_acc=("pred_stage", lambda s: (s == pred.loc[s.index, "stage"]).mean()))
    print(f"\nголова: {n_params / 1e6:.2f} млн параметров, обучение {time.perf_counter() - t0:.0f} с → {run}")
    print(by_video.round(3).to_string())


def plot_curves(h: pd.DataFrame, path: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, (col, title) in zip(axes, [("loss", "Лосс"), ("mae_pp", "Ошибка прогресса, п.п."),
                                       ("stage_acc", "Точность стадии")]):
        ax.plot(h["epoch"], h[col], color="#2a78d6", lw=2, label="обучение")
        hold = "holdout_" + col
        if hold in h:
            ax.plot(h["epoch"], h[hold], color="#eb6834", lw=2, label="отложенное видео")
            ax.legend(frameon=False)
        ax.set_title(title, loc="left")
        ax.set_xlabel("эпоха")
        ax.grid(alpha=0.3)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    sys.exit(main())
