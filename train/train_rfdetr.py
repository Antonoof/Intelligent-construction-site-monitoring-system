#!/usr/bin/env python3
"""RF-DETR: обучение на YOLO-датасете. Одна GPU.

    python train_rfdetr.py --data C:\\ml\\dataset
"""
from __future__ import annotations

import argparse
import shutil
import sys
from collections import Counter
import time
from pathlib import Path

from tqdm.auto import tqdm

# запасной список — используется, только если в датасете нет data.yaml и не передан --classes
CLASSES = [
    "Dump truck", "Excavator", "Bucket loader", "Bulldozer", "Mixer",
    "Crane manipulator", "Autocran", "Forklift", "Drilling rig",
]

IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
VAL_ALIASES = ("val", "valid", "validation")


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Обучение RF-DETR на YOLO-датасете")
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--out", type=Path, default=Path("./runs/rfdetr"))
    p.add_argument("--classes", type=Path, default=None,
                   help="txt со списком классов, по одному на строку")

    p.add_argument("--model", choices=("nano", "small", "medium", "large"), default="large")
    p.add_argument("--resolution", type=int, default=704, help="кратно 32")

    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--batch", default="8", help="число или auto")
    p.add_argument("--grad-accum", type=int, default=2)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--lr-encoder", type=float, default=1.5e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--patience", type=int, default=8, help="0 — не останавливать")
    p.add_argument("--skip-best-epochs", type=int, default=2)
    p.add_argument("--grad-checkpointing", action="store_true")
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--devices", type=int, default=1)
    p.add_argument("--resume", type=Path, default=None)

    a = p.parse_args(argv)
    if a.resolution % 32:
        p.error(f"--resolution должен делиться на 32, получено {a.resolution}")
    if a.batch != "auto":
        try:
            a.batch = int(a.batch)
        except ValueError:
            p.error("--batch: число или auto")
    return a


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
        raise SystemExit(
            f"не нашёл сплиты {sorted(missing)} в {root}\n"
            f"ожидаю {root}\\train\\images + {root}\\train\\labels "
            f"(val | valid | validation для валидации)\n"
            f"внутри лежит: {inside}"
        )
    return found


def read_yaml_classes(root: Path):
    """names из data.yaml датасета (список или словарь {id: name})."""
    for fn in ("data.yaml", "data.yml"):
        y = root / fn
        if not y.exists():
            continue
        import yaml
        names = (yaml.safe_load(y.read_text(encoding="utf-8")) or {}).get("names")
        if isinstance(names, dict):
            return [str(names[k]) for k in sorted(names, key=int)]
        if isinstance(names, list):
            return [str(n) for n in names]
    return None


def count_boxes(split_dir: Path):
    """Сколько боксов каждого класса и в каких файлах они лежат."""
    per_class, files = Counter(), {}
    for lbl in (split_dir / "labels").glob("*.txt"):
        ids = set()
        for line in lbl.read_text().splitlines():
            q = line.split()
            if q:
                c = int(float(q[0]))
                per_class[c] += 1
                ids.add(c)
        files[lbl] = ids
    return per_class, files


def exclude_untrained_from_val(root: Path, splits, classes):
    """Классы, которых нет в train, убираем из val — иначе у них AP=0 и общий mAP занижен.

    Кадры не удаляются, а переносятся в <датасет>/_excluded_val. При каждом запуске
    они сначала возвращаются обратно, так что если в train появятся такие классы,
    кадры автоматически вернутся в валидацию.
    """
    val = splits["val"]
    store = root / "_excluded_val"
    for sub in ("images", "labels"):
        if (store / sub).is_dir():
            for f in (store / sub).iterdir():
                shutil.move(str(f), str(val / sub / f.name))

    train_cnt, _ = count_boxes(splits["train"])
    val_cnt, val_files = count_boxes(val)
    drop = sorted(c for c in val_cnt if train_cnt[c] == 0)
    if not drop:
        return
    (store / "images").mkdir(parents=True, exist_ok=True)
    (store / "labels").mkdir(parents=True, exist_ok=True)
    moved = 0
    for lbl, ids in val_files.items():
        if ids & set(drop):
            for img in (val / "images").glob(lbl.stem + ".*"):
                if img.suffix.lower() in IMG_EXTS:
                    shutil.move(str(img), str(store / "images" / img.name))
            shutil.move(str(lbl), str(store / "labels" / lbl.name))
            moved += 1
    names = ", ".join(classes[c] if c < len(classes) else str(c) for c in drop)
    print(f"нет в train, исключены из валидации: {names} "
          f"({moved} кадров перенесено в {store})")


def prepare_dataset(root: Path, classes) -> Path:
    """Пишет data.yaml в корень датасета и печатает сводку."""
    splits = find_splits(root)
    exclude_untrained_from_val(root, splits, classes)

    (root / "data.yaml").write_text(
        f"train: {splits['train'].name}/images\n"
        f"val: {splits['val'].name}/images\n"
        f"test: {splits['val'].name}/images\n\n"
        f"nc: {len(classes)}\n"
        "names:\n" + "".join(f"  {i}: {n}\n" for i, n in enumerate(classes)),
        encoding="utf-8",
    )

    for key in ("train", "val"):
        d = splits[key]
        imgs = [p for p in (d / "images").glob("*") if p.suffix.lower() in IMG_EXTS]
        lbls = list((d / "labels").glob("*.txt"))
        boxes, bad = 0, 0
        for lbl in lbls:
            for line in lbl.read_text().splitlines():
                q = line.split()
                if not q:
                    continue
                boxes += 1
                if len(q) != 5 or not (0 <= int(float(q[0])) < len(classes)):
                    bad += 1
        print(f"{key:<6} {len(imgs):>7} изображений  {len(lbls):>7} разметок  {boxes:>8} боксов"
              + (f"  ! битых строк: {bad}" if bad else ""))
        if key == "train" and not imgs:
            raise SystemExit("в train нет изображений")

    return root


def make_epoch_bar(total_epochs):
    """Колбэк Lightning: полоса по эпохам — сколько прошло, сколько осталось."""
    from pytorch_lightning import Callback

    class EpochProgress(Callback):
        def __init__(self, total):
            super().__init__()
            self.total = total
            self.bar = None
            self.t0 = None

        def on_train_start(self, trainer, pl_module):
            self.t0 = time.time()
            self.bar = tqdm(
                total=self.total, desc="эпохи", unit="эп",
                bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} "
                           "[прошло {elapsed}, осталось {remaining}{postfix}]",
                dynamic_ncols=True, leave=True, position=0,
            )

        def on_train_epoch_end(self, trainer, pl_module):
            if self.bar is None:
                return
            post = {}
            for key, val in trainer.callback_metrics.items():
                k = str(key).lower()
                if "map" in k and "ema" not in k and len(post) < 2:
                    try:
                        post[str(key).split("/")[-1]] = f"{float(val):.3f}"
                    except (TypeError, ValueError):
                        pass
            if post:
                self.bar.set_postfix(post)
            self.bar.update(1)

        def on_train_end(self, trainer, pl_module):
            if self.bar is not None:
                self.bar.close()
                self.bar = None
            if self.t0:
                print(f"обучение заняло {(time.time() - self.t0) / 60:.1f} мин")

    return EpochProgress(total_epochs)


def attach_epoch_bar(total_epochs):
    """RF-DETR не принимает callbacks в train(), поэтому оборачиваем build_trainer.

    Импорт build_trainer внутри train() ленивый, так что подмена атрибута модуля
    срабатывает.
    """
    import rfdetr.training as rt

    original = rt.build_trainer

    def wrapped(*args, **kwargs):
        trainer = original(*args, **kwargs)
        try:
            trainer.callbacks.append(make_epoch_bar(total_epochs))
        except Exception as e:
            print("не смог повесить полосу прогресса по эпохам:", e)
        return trainer

    rt.build_trainer = wrapped


def main(argv=None):
    args = parse_args(argv)

    if args.classes:
        classes = [l.strip() for l in args.classes.read_text(encoding="utf-8").splitlines()
                   if l.strip()]
        src = args.classes
    else:
        classes = read_yaml_classes(args.data)
        src = args.data / "data.yaml"
        if not classes:
            classes, src = CLASSES, "встроенный список CLASSES"

    print(f"классов: {len(classes)} (из {src}): {', '.join(classes)}")
    dataset_dir = prepare_dataset(args.data, classes)

    from rfdetr import RFDETRLarge, RFDETRMedium, RFDETRNano, RFDETRSmall
    cls_obj = {"nano": RFDETRNano, "small": RFDETRSmall,
               "medium": RFDETRMedium, "large": RFDETRLarge}[args.model]

    print(f"модель: RF-DETR {args.model} @ {args.resolution}px")
    if isinstance(args.batch, int):
        print(f"effective batch = {args.batch} x {args.grad_accum} x {args.devices} "
              f"= {args.batch * args.grad_accum * args.devices}")

    attach_epoch_bar(args.epochs)

    model = cls_obj(gradient_checkpointing=args.grad_checkpointing)
    kwargs = dict(
        dataset_dir=str(dataset_dir),
        output_dir=str(args.out),
        epochs=args.epochs,
        batch_size=args.batch,
        grad_accum_steps=args.grad_accum,
        lr=args.lr,
        lr_encoder=args.lr_encoder,
        weight_decay=args.weight_decay,
        resolution=args.resolution,
        early_stopping=args.patience > 0,
        early_stopping_patience=max(args.patience, 1),
        skip_best_epochs=args.skip_best_epochs,
        log_per_class_metrics=True,
        checkpoint_interval=5,
        num_workers=args.num_workers,
        seed=args.seed,
        tensorboard=False,
        devices=args.devices,
        progress_bar="tqdm",
    )
    if args.resume:
        kwargs["resume"] = str(args.resume)

    args.out.mkdir(parents=True, exist_ok=True)
    model.train(**kwargs)

    best = next((args.out / n for n in ("checkpoint_best_total.pth", "checkpoint_best_ema.pth",
                                        "checkpoint_best_regular.pth")
                 if (args.out / n).exists()), None)
    print(f"\nготово. лучший чекпоинт: {best or 'не найден в ' + str(args.out)}")


if __name__ == "__main__":
    sys.exit(main())
