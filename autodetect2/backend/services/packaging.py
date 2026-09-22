from __future__ import annotations

import json
import zipfile
from pathlib import Path

import yaml

from ..config import AUTOFILE_DIR, MODEL_FAMILIES
from .project import Project, read_train_yaml, utc_now

TRAIN_TEMPLATE = AUTOFILE_DIR / "train"
INFER_TEMPLATE = AUTOFILE_DIR / "infer"


def _write_tree(archive: zipfile.ZipFile, source: Path, prefix: str = "") -> None:
    for path in sorted(source.rglob("*")):
        if path.is_dir() or "__pycache__" in path.parts:
            continue
        archive.write(path, str(Path(prefix) / path.relative_to(source)).replace("\\", "/"))


def _ensemble_summary(config: dict) -> list[str]:
    model = config.get("model", {})
    enabled = [m for m in model.get("ensemble", []) if m.get("enabled")]
    if not enabled:
        enabled = [{"family": model.get("family", "11"), "size": model.get("size", "x"), "imgsz": 960}]
    return [
        f"{MODEL_FAMILIES.get(str(m['family']), {}).get('name', 'yolo11')}{m.get('size', 'x')} @ {m.get('imgsz')}"
        for m in enabled
    ]


def train_readme(project: Project, config: dict) -> str:
    models = "\n".join(f"- `{item}`" for item in _ensemble_summary(config))
    train = config.get("train", {})
    counts = project.meta.get("counts", {})

    return f"""# AutoFile — обучение · {project.name}

Сгенерировано AutoDetect2 {utc_now()}.

## Что внутри
- `train.py` — обучение ансамбля YOLO по `train.yaml`
- `predict.py` — сборка `submission.csv` на всех данных лучшей моделью
- `data.yaml` — датасет ({counts.get('train', 0)} train / {counts.get('val', 0)} val, {project.meta.get('nc', 0)} классов)
- `train.yaml` — полная конфигурация обучения

## Модели
{models}

## Запуск
```bash
pip install -r requirements.txt
python train.py --dataset-root /path/to/dataset
```
`--dataset-root` не нужен, если пути из `data.yaml` уже валидны на этой машине:
пути переносятся по последним 1–3 сегментам (`.../train/images`).

## Результат
- `result/best_<i>.pt` — веса каждой модели ансамбля
- `submission.csv` — `image_id,class_id,confidence,x_center,y_center,width,height`
  (координаты нормированы), файл загружается обратно в AutoDetect2 через
  «Загрузить результаты».

## Ключевые параметры
| Параметр | Значение |
|---|---|
| epochs | {train.get('epochs')} |
| imgsz | {train.get('imgsz')} |
| batch | {train.get('batch')} |
| optimizer | {train.get('optimizer')} |
| lr0 / lrf | {train.get('lr0')} / {train.get('lrf')} |
| seed | {config.get('runtime', {}).get('seed')} |
"""


def infer_readme(project: Project, params: dict, models: list[str]) -> str:
    listed = "\n".join(f"- `{name}`" for name in models) or "- (положите `*.pt` в `models/`)"
    per_model = "\n".join(f"| {k} | {v:.4f} |" for k, v in params.get("conf", {}).items())

    return f"""# AutoFile — инференс · {project.name}

Сгенерировано AutoDetect2 {utc_now()}.

## Что внутри
- `infer.py` — ансамблевый инференс с Weighted Box Fusion
- `wbf_params.json` — параметры, найденные Optuna на валидации

## Модели
{listed}

## Запуск
```bash
pip install -r requirements.txt
python infer.py --images /path/to/test/images
```

## Параметры WBF
| Параметр | Значение |
|---|---|
| wbf_iou | {params.get('wbf_iou')} |
| skip_box_thr | {params.get('skip_box_thr')} |
| conf_type | {params.get('conf_type')} |
| best F1 | {params.get('score', 0):.4f} |

### Порог confidence по моделям
| Модель | conf |
|---|---|
{per_model}
"""


def build_train_bundle(project: Project, config: dict | None = None) -> Path:
    config = config or read_train_yaml(project)
    project.exports_dir.mkdir(parents=True, exist_ok=True)
    target = project.exports_dir / "AutoFile.zip"

    with zipfile.ZipFile(target, "w", zipfile.ZIP_DEFLATED) as archive:
        _write_tree(archive, TRAIN_TEMPLATE)

        archive.writestr("data.yaml", project.data_yaml.read_text(encoding="utf-8"))
        archive.writestr("train.yaml", yaml.safe_dump(config, allow_unicode=True, sort_keys=False))
        archive.writestr("README.md", train_readme(project, config))
        archive.writestr(
            "project.json",
            json.dumps(
                {
                    "project": project.name,
                    "id": project.id,
                    "classes": project.meta.get("names", []),
                    "counts": project.meta.get("counts", {}),
                    "generated_at": utc_now(),
                },
                indent=2,
                ensure_ascii=False,
            ),
        )
    return target


def build_infer_bundle(
    project: Project,
    params: dict,
    model_paths: list[Path],
    include_weights: bool = False,
) -> Path:
    project.exports_dir.mkdir(parents=True, exist_ok=True)
    target = project.exports_dir / "AutoFile_infer.zip"
    names = [p.name for p in model_paths]

    with zipfile.ZipFile(target, "w", zipfile.ZIP_DEFLATED) as archive:
        _write_tree(archive, INFER_TEMPLATE)
        archive.writestr("wbf_params.json", json.dumps(params, indent=2))
        archive.writestr("README.md", infer_readme(project, params, names))
        if include_weights:
            for path in model_paths:
                archive.write(path, f"models/{path.name}")
        else:
            archive.writestr("models/PUT_MODELS_HERE.txt", "\n".join(names) + "\n")
    return target
