"""Экспорт детектора для ноутбуков без GPU (ТЗ, раздел 3.1) и замер скорости на CPU.

  python training/08_export_cpu.py                                              # v2, если обучена, иначе v1
  python training/08_export_cpu.py --weights weights/rfdetr_large_best_ema.pth  # явно v1 (10 классов)
→ weights/rfdetr_v2.onnx + weights/rfdetr_v2.classes.json; сервис подхватывает их при OKO_DETECTOR=onnx
  (или auto — если PyTorch не установлен). Для работы сервиса нужен только onnxruntime; для экспорта — PyTorch и rfdetr.

Для слабых ноутбуков можно обучить вариант поменьше (RF-DETR Small/Medium, те же скрипты 06–07) —
на CPU он в несколько раз быстрее Large ценой нескольких пунктов AP.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--weights", default=None,
                    help="по умолчанию weights/rfdetr_large_v2_best.pth, если есть, иначе v1 rfdetr_large_best_ema.pth")
    ap.add_argument("--variant", default="large", help="вариант RF-DETR для чекпойнтов старого формата")
    ap.add_argument("--classes", default="", help="порядок классов через запятую, если чекпойнт их не хранит")
    ap.add_argument("--out", default=str(ROOT / "weights" / "rfdetr_v2.onnx"))
    ap.add_argument("--bench-images", default=str(ROOT / "data" / "demo" / "housing" / "snapshots"))
    ap.add_argument("--runs", type=int, default=10)
    a = ap.parse_args(argv)

    if a.weights is None:
        v2 = ROOT / "weights" / "rfdetr_large_v2_best.pth"
        a.weights = str(v2 if v2.exists() else ROOT / "weights" / "rfdetr_large_best_ema.pth")
    print(f"веса: {a.weights}")

    import rfdetr
    try:   # современные чекпойнты хранят архитектуру и имена классов
        model = rfdetr.RFDETR.from_checkpoint(a.weights, device="cpu", trust_checkpoint=True)
    except Exception as e:   # старый формат — собираем модель по варианту (как backend/app/detection/rfdetr_backend.py)
        print(f"from_checkpoint не сработал ({e}), загружаем как RFDETR{a.variant.capitalize()}")
        cls = {"nano": "RFDETRNano", "small": "RFDETRSmall", "medium": "RFDETRMedium", "base": "RFDETRBase",
               "large": "RFDETRLarge"}[a.variant.lower()]
        try:
            model = getattr(rfdetr, cls)(pretrain_weights=a.weights, device="cpu", trust_checkpoint=True)
        except TypeError:   # версии rfdetr без параметра trust_checkpoint
            model = getattr(rfdetr, cls)(pretrain_weights=a.weights, device="cpu")
    names = [c.strip() for c in a.classes.split(",") if c.strip()]
    if not names:
        try:
            names = list(model.class_names)
        except Exception:
            names = []
    if not names or len(names) > 40:     # нет имён или COCO — порядок классов v1 из ML-части
        sys.path.insert(0, str(ROOT / "backend"))
        from app.detection.rfdetr_backend import V1_CLASSES
        names = V1_CLASSES
    tmp = Path(a.out).parent / "_export"
    path = model.export(format="onnx", output_dir=str(tmp))
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(path), out)
    shutil.rmtree(tmp, ignore_errors=True)
    out.with_suffix(".classes.json").write_text(json.dumps(names, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"ONNX: {out}  ({out.stat().st_size / 1e6:.0f} МБ), классы: {names}")

    # замер на CPU тем же кодом, что и в сервисе (backend/app/detection/onnx_backend.py)
    sys.path.insert(0, str(ROOT / "backend"))
    from PIL import Image
    from app.detection.onnx_backend import OnnxDetector
    det = OnnxDetector(out)
    imgs = sorted(Path(a.bench_images).glob("*.jpg"))[: a.runs] or [None]
    times = []
    for p in imgs:
        img = Image.open(p).convert("RGB") if p else Image.new("RGB", (1280, 720))
        t = time.perf_counter()
        det.detect(img)
        times.append(time.perf_counter() - t)
    print(f"CPU: {1000 * sum(times[1:] or times) / max(1, len(times[1:] or times)):.0f} мс на кадр "
          f"(без первого прогрева, {len(times)} кадров)")


if __name__ == "__main__":
    main()
