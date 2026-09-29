"""Дообучение детектора RF-DETR Large на датасете v2 (классы ТЗ: + каток, грузовик; бетононасос — если есть в данных).

Стартует с весов v1 из ML-части (weights/rfdetr_large_best_ema.pth): признаки техники уже выучены, голова
классов переинициализируется под новое число классов (RF-DETR делает это сам по разметке датасета).

  python training/06_train_detector.py --dataset datasets/oko_v2 --epochs 40
Результат: training/runs/detector_v2/…/checkpoint_best_total.pth → weights/rfdetr_large_v2_best.pth
Время: на RTX 4090 порядка часа на 10 тыс. снимков при 40 эпохах (оценка; зависит от объёма датасета).
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--dataset", default=str(ROOT / "datasets" / "oko_v2"))
    ap.add_argument("--init", default=str(ROOT / "weights" / "rfdetr_large_best_ema.pth"),
                    help="веса v1 (ML-часть); если файла нет — старт с весов COCO")
    ap.add_argument("--out", default=str(ROOT / "training" / "runs" / "detector_v2"))
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--grad-accum", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--resolution", type=int, default=0, help="0 — разрешение модели по умолчанию")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--save-as", default=str(ROOT / "weights" / "rfdetr_large_v2_best.pth"))
    a = ap.parse_args(argv)

    from rfdetr import RFDETRLarge

    init = Path(a.init)
    model = RFDETRLarge(pretrain_weights=str(init), trust_checkpoint=True) if init.exists() else RFDETRLarge()
    kw = dict(dataset_dir=a.dataset, epochs=a.epochs, batch_size=a.batch, grad_accum_steps=a.grad_accum, lr=a.lr,
              output_dir=a.out, device=a.device, early_stopping=True,
              notes={"project": "ОКО", "classes": "v2: 10 классов ML-части + каток, грузовик"})
    if a.resolution:
        kw["resolution"] = a.resolution
    model.train(**kw)

    best = None
    for name in ("checkpoint_best_total.pth", "checkpoint_best_ema.pth", "checkpoint_best_regular.pth"):
        cands = sorted(Path(a.out).rglob(name))
        if cands:
            best = cands[-1]
            break
    if best is None:
        raise SystemExit(f"не найден лучший чекпойнт в {a.out}")
    Path(a.save_as).parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(best, a.save_as)
    print(f"лучшие веса: {best} → {a.save_as}")
    print("дальше: python training/07_eval_conditions.py --weights", a.save_as)


if __name__ == "__main__":
    main()
