#!/usr/bin/env python3
"""D-FINE: обучение на YOLO-датасете. Одна GPU.

    python train_dfine.py --data C:\\ml\\dataset
"""
from __future__ import annotations

import argparse
import math
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

CLASSES = [
    "Dump truck", "Excavator", "Bucket loader", "Bulldozer", "Mixer",
    "Crane manipulator", "Autocran", "Forklift", "Drilling rig",
]

IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
VAL_ALIASES = ("val", "valid", "validation")
SHORT = {"nano": "n", "small": "s", "medium": "m", "large": "l", "xlarge": "x"}


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Обучение D-FINE на YOLO-датасете")
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--out", type=Path, default=Path("./runs/dfine"))
    p.add_argument("--classes", type=Path, default=None)

    p.add_argument("--model", choices=tuple(SHORT), default="medium")
    p.add_argument("--pretrain", choices=("obj2coco", "coco", "obj365"), default="obj2coco")
    p.add_argument("--imgsz", type=int, default=640)

    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--accum", type=int, default=2)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--lr-backbone", type=float, default=1e-5)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--clip", type=float, default=0.1)
    p.add_argument("--warmup", type=float, default=0.03)
    p.add_argument("--patience", type=int, default=8)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no-amp", action="store_true")
    p.add_argument("--resume", type=Path, default=None)
    return p.parse_args(argv)


def find_splits(root: Path):
    found = {}
    for key, names in (("train", ("train",)), ("val", VAL_ALIASES)):
        for name in names:
            d = root / name
            if (d / "images").is_dir() and (d / "labels").is_dir():
                found[key] = d
                break
    missing = {"train", "val"} - set(found)
    if missing:
        inside = sorted(p.name for p in root.iterdir() if p.is_dir())[:10] if root.is_dir() else []
        raise SystemExit(f"не нашёл сплиты {sorted(missing)} в {root}\nвнутри лежит: {inside}")
    return found


def summarize(splits, classes):
    for key in ("train", "val"):
        d = splits[key]
        imgs = [p for p in (d / "images").glob("*") if p.suffix.lower() in IMG_EXTS]
        lbls = list((d / "labels").glob("*.txt"))
        boxes = bad = 0
        for lbl in lbls:
            for line in lbl.read_text().splitlines():
                q = line.split()
                if not q:
                    continue
                boxes += 1
                if len(q) != 5 or not (0 <= int(q[0]) < len(classes)):
                    bad += 1
        print(f"{key:<6} {len(imgs):>7} изображений  {len(lbls):>7} разметок  {boxes:>8} боксов"
              + (f"  ! битых строк: {bad}" if bad else ""))
        if key == "train" and not imgs:
            raise SystemExit("в train нет изображений")


def imread_any(path: Path):
    import cv2
    return cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)


class YoloDetDataset(Dataset):
    """YOLO-папка (images/ + labels/) -> батчи для D-FINE."""

    def __init__(self, split_dir: Path, processor, aug=None):
        self.img_dir, self.lbl_dir = split_dir / "images", split_dir / "labels"
        self.paths = sorted(p for p in self.img_dir.glob("*") if p.suffix.lower() in IMG_EXTS)
        self.processor, self.aug = processor, aug

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        import cv2
        path = self.paths[i]
        img = cv2.cvtColor(imread_any(path), cv2.COLOR_BGR2RGB)

        boxes, cls = [], []
        lbl = self.lbl_dir / f"{path.stem}.txt"
        if lbl.exists():
            for line in lbl.read_text().splitlines():
                q = line.split()
                if len(q) != 5:
                    continue
                xc, yc, bw, bh = (float(v) for v in q[1:])
                # разметка бывает на доли пикселя за краем кадра; albumentations
                # такое не прощает, поэтому обрезаем по границам
                x1, y1 = max(0.0, xc - bw / 2), max(0.0, yc - bh / 2)
                x2, y2 = min(1.0, xc + bw / 2), min(1.0, yc + bh / 2)
                if x2 - x1 > 1e-6 and y2 - y1 > 1e-6:
                    cls.append(int(q[0]))
                    boxes.append([(x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1])

        if self.aug is not None and boxes:
            out = self.aug(image=img, bboxes=boxes, cls=cls)
            img, boxes, cls = out["image"], list(out["bboxes"]), list(out["cls"])

        h, w = img.shape[:2]
        objs = []
        for (xc, yc, bw, bh), c in zip(boxes, cls):
            x, y = (xc - bw / 2) * w, (yc - bh / 2) * h
            objs.append({"bbox": [x, y, bw * w, bh * h], "category_id": int(c),
                         "area": bw * w * bh * h, "iscrowd": 0})

        enc = self.processor(images=img, annotations={"image_id": i, "annotations": objs},
                             return_tensors="pt")
        return {"pixel_values": enc["pixel_values"][0], "labels": enc["labels"][0]}


def collate(batch):
    return {"pixel_values": torch.stack([b["pixel_values"] for b in batch]),
            "labels": [b["labels"] for b in batch]}


def build_aug():
    import albumentations as A
    return A.Compose(
        [
            A.HorizontalFlip(p=0.5),
            A.Affine(scale=(0.7, 1.3), translate_percent=(-0.1, 0.1), p=0.5),
            A.RandomBrightnessContrast(p=0.5),
            A.HueSaturationValue(p=0.5),
        ],
        bbox_params=A.BboxParams(format="yolo", label_fields=["cls"], min_visibility=0.3),
    )


def load_model(args, classes):
    from transformers import AutoImageProcessor, AutoModelForObjectDetection

    id2label = {i: n for i, n in enumerate(classes)}
    candidates = [
        f"ustc-community/dfine-{args.model}-{args.pretrain}",
        f"ustc-community/dfine_{SHORT[args.model]}_{args.pretrain}",
        f"ustc-community/dfine-{args.model}-coco",
        f"ustc-community/dfine_{SHORT[args.model]}_coco",
    ]
    for repo in candidates:
        try:
            processor = AutoImageProcessor.from_pretrained(repo)
            model = AutoModelForObjectDetection.from_pretrained(
                repo, num_labels=len(classes), id2label=id2label,
                label2id={n: i for i, n in id2label.items()},
                ignore_mismatched_sizes=True,
            )
            processor.size = {"height": args.imgsz, "width": args.imgsz}
            print(f"чекпоинт: {repo}")
            return model, processor
        except Exception as e:
            print(f"{repo}: не вышло ({type(e).__name__})")
    raise SystemExit("ни один чекпоинт не загрузился, проверь интернет")


def build_coco_gt(split_dir: Path, classes):
    img_dir, lbl_dir = split_dir / "images", split_dir / "labels"
    paths = sorted(p for p in img_dir.glob("*") if p.suffix.lower() in IMG_EXTS)

    images, anns, ann_id = [], [], 1
    for img_id, p in enumerate(paths, 1):
        w, h = Image.open(p).size
        images.append({"id": img_id, "file_name": p.name, "width": w, "height": h})
        lbl = lbl_dir / f"{p.stem}.txt"
        if not lbl.exists():
            continue
        for line in lbl.read_text().splitlines():
            q = line.split()
            if len(q) != 5:
                continue
            c = int(q[0])
            xc, yc, bw, bh = (float(v) for v in q[1:])
            x, y, bw, bh = (xc - bw / 2) * w, (yc - bh / 2) * h, bw * w, bh * h
            anns.append({"id": ann_id, "image_id": img_id, "category_id": c + 1,
                         "bbox": [x, y, bw, bh], "area": bw * bh, "iscrowd": 0})
            ann_id += 1

    return ({"images": images, "annotations": anns,
             "categories": [{"id": i + 1, "name": n} for i, n in enumerate(classes)]}, paths)


@torch.no_grad()
def predict_all(model, processor, paths, device, batch, conf=0.001, amp=True, desc="val"):
    import cv2
    model.eval()
    dets = []
    for i in tqdm(range(0, len(paths), batch), desc=desc, leave=False):
        chunk = paths[i:i + batch]
        imgs = [cv2.cvtColor(imread_any(p), cv2.COLOR_BGR2RGB) for p in chunk]
        sizes = [im.shape[:2] for im in imgs]
        px = processor(images=imgs, return_tensors="pt")["pixel_values"].to(device)

        with torch.autocast("cuda", dtype=torch.float16, enabled=amp and device == "cuda"):
            out = model(pixel_values=px)

        for k, r in enumerate(processor.post_process_object_detection(
                out, target_sizes=sizes, threshold=conf)):
            img_id = i + k + 1
            for j in r["scores"].argsort(descending=True)[:100].tolist():
                x1, y1, x2, y2 = r["boxes"][j].tolist()
                dets.append({"image_id": img_id, "category_id": int(r["labels"][j]) + 1,
                             "bbox": [x1, y1, x2 - x1, y2 - y1],
                             "score": float(r["scores"][j])})
    return dets


def coco_eval(gt, dets):
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    if not dets:
        return None
    coco_gt = COCO()
    coco_gt.dataset = gt
    coco_gt.createIndex()
    E = COCOeval(coco_gt, coco_gt.loadRes(dets), "bbox")
    E.evaluate(); E.accumulate()
    return E


def coco_map(E):
    """(mAP50-95, mAP50) без печати."""
    if E is None:
        return 0.0, 0.0
    prec = E.eval["precision"]

    def _m(x):
        x = x[x > -1]
        return float(x.mean()) if x.size else 0.0

    return _m(prec[:, :, :, 0, 2]), _m(prec[0, :, :, 0, 2])


def per_class_table(E, gt, classes, out_dir: Path):
    prec, rec = E.eval["precision"], E.eval["recall"]
    cat_ids = list(E.params.catIds)
    counts = Counter(a["category_id"] - 1 for a in gt["annotations"])

    def _m(a):
        a = a[a > -1]
        return float(a.mean()) if a.size else np.nan

    rows = []
    for i, name in enumerate(classes):
        k = cat_ids.index(i + 1)
        rows.append({"id": i, "class": name, "n_val": counts.get(i, 0),
                     "R@0.5": _m(rec[0, k, 0, 2]),
                     "mAP50": _m(prec[0, :, k, 0, 2]),
                     "mAP50-95": _m(prec[:, :, k, 0, 2])})
    df = pd.DataFrame(rows).sort_values("mAP50", na_position="last")
    df.to_csv(out_dir / "metrics_per_class.csv", index=False, encoding="utf-8-sig")
    with pd.option_context("display.float_format", lambda v: f"{v:.3f}"):
        print("\nпо классам:")
        print(df.to_string(index=False, na_rep="—"))


def train_loop(model, processor, train_dl, gt, val_paths, args, device):
    from transformers import get_cosine_schedule_with_warmup

    amp = not args.no_amp and device == "cuda"
    best_path = args.out / "best.pt"

    backbone = {id(p) for n, p in model.named_parameters() if "backbone" in n and p.requires_grad}
    optimizer = torch.optim.AdamW(
        [{"params": [p for p in model.parameters() if p.requires_grad and id(p) not in backbone],
          "lr": args.lr},
         {"params": [p for p in model.parameters() if p.requires_grad and id(p) in backbone],
          "lr": args.lr_backbone}],
        weight_decay=args.weight_decay,
    )
    steps_per_epoch = math.ceil(len(train_dl) / args.accum)
    total_steps = steps_per_epoch * args.epochs
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, int(total_steps * args.warmup), total_steps)
    scaler = torch.amp.GradScaler("cuda", enabled=amp)

    start_epoch, best_map, bad = 1, -1.0, 0
    if args.resume and args.resume.exists():
        ck = torch.load(args.resume, map_location="cpu", weights_only=False)
        model.load_state_dict(ck["model"])
        optimizer.load_state_dict(ck["optimizer"])
        scheduler.load_state_dict(ck["scheduler"])
        start_epoch, best_map = ck["epoch"] + 1, ck.get("mAP50-95", -1.0)
        print(f"продолжаю с эпохи {start_epoch}, лучший mAP50-95 = {best_map:.3f}")

    history = []
    epochs_bar = tqdm(
        total=args.epochs, initial=start_epoch - 1, desc="эпохи", unit="эп",
        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} "
                   "[прошло {elapsed}, осталось {remaining}{postfix}]",
        dynamic_ncols=True, position=0, leave=True,
    )
    t0 = time.time()

    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        running = seen = skipped = 0
        bar = tqdm(train_dl, desc=f"эпоха {epoch}/{args.epochs}", position=1,
                   leave=False, dynamic_ncols=True)

        for step, batch in enumerate(bar):
            px = batch["pixel_values"].to(device, non_blocking=True)
            labels = [{k: v.to(device) for k, v in l.items()} for l in batch["labels"]]

            with torch.autocast("cuda", dtype=torch.float16, enabled=amp):
                loss = model(pixel_values=px, labels=labels).loss

            # лосс DETR в fp16 изредка даёт NaN; такой батч пропускаем,
            # иначе одним шагом уводим в NaN все веса
            if not torch.isfinite(loss):
                skipped += 1
                optimizer.zero_grad(set_to_none=True)
                continue

            scaler.scale(loss / args.accum).backward()

            if (step + 1) % args.accum == 0 or step + 1 == len(train_dl):
                scaler.unscale_(optimizer)
                gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
                if torch.isfinite(gnorm):
                    scaler.step(optimizer)
                else:
                    skipped += 1
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()

            running += float(loss.detach()); seen += 1
            bar.set_postfix(loss=f"{running / seen:.3f}", lr=f"{scheduler.get_last_lr()[0]:.2e}")

        bar.close()

        E = coco_eval(gt, predict_all(model, processor, val_paths, device, args.batch,
                                      amp=amp, desc=f"val {epoch}"))
        m5095, m50 = coco_map(E)
        loss_avg = running / max(seen, 1)
        history.append({"epoch": epoch, "loss": loss_avg, "mAP50": m50, "mAP50-95": m5095})
        pd.DataFrame(history).to_csv(args.out / "history.csv", index=False)

        epochs_bar.set_postfix({"loss": f"{loss_avg:.3f}", "mAP50": f"{m50:.3f}",
                                "mAP50-95": f"{m5095:.3f}"})
        epochs_bar.update(1)
        if skipped:
            tqdm.write(f"эпоха {epoch}: пропущено из-за NaN: {skipped}")

        if m5095 > best_map:
            best_map, bad = m5095, 0
            torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                        "scheduler": scheduler.state_dict(), "epoch": epoch,
                        "mAP50-95": m5095, "classes": len(gt["categories"]),
                        "imgsz": args.imgsz}, best_path)
            tqdm.write(f"эпоха {epoch}: новый лучший mAP50-95 = {m5095:.3f}")
        else:
            bad += 1
            if args.patience and bad >= args.patience:
                tqdm.write(f"{args.patience} эпох без улучшения, останавливаюсь")
                break

    epochs_bar.close()
    print(f"\nобучение заняло {(time.time() - t0) / 60:.1f} мин, лучший mAP50-95 = {best_map:.3f}")
    return best_path


def main(argv=None):
    args = parse_args(argv)

    classes = CLASSES
    if args.classes:
        classes = [l.strip() for l in args.classes.read_text(encoding="utf-8").splitlines()
                   if l.strip()]
    print(f"классов: {len(classes)}")

    splits = find_splits(args.data)
    summarize(splits, classes)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    args.out.mkdir(parents=True, exist_ok=True)

    model, processor = load_model(args, classes)
    model.to(device)
    print(f"модель: D-FINE {args.model}/{args.pretrain} @ {args.imgsz}px, "
          f"{sum(p.numel() for p in model.parameters()) / 1e6:.1f}M параметров")
    print(f"effective batch = {args.batch} x {args.accum} = {args.batch * args.accum}")

    train_dl = DataLoader(
        YoloDetDataset(splits["train"], processor, build_aug()),
        batch_size=args.batch, shuffle=True, collate_fn=collate,
        num_workers=args.workers, pin_memory=True, drop_last=True,
    )
    gt, val_paths = build_coco_gt(splits["val"], classes)
    print(f"val: {len(gt['images'])} изображений, {len(gt['annotations'])} боксов")

    best_path = train_loop(model, processor, train_dl, gt, val_paths, args, device)

    ck = torch.load(best_path, map_location="cpu", weights_only=False)
    model.load_state_dict(ck["model"])
    model.to(device)
    E = coco_eval(gt, predict_all(model, processor, val_paths, device, args.batch,
                                  amp=not args.no_amp and device == "cuda", desc="финальный val"))
    E.summarize()
    per_class_table(E, gt, classes, args.out)
    m5095, m50 = coco_map(E)
    print(f"\nD-FINE: mAP50 = {m50:.3f}, mAP50-95 = {m5095:.3f}")
    print(f"веса: {best_path}")


if __name__ == "__main__":
    sys.exit(main())
