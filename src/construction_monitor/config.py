"""Загрузка YAML-конфига и приведение путей к абсолютным."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "inference.yaml"


def resolve_path(p: str | Path) -> Path:
    """Относительный путь считается от корня репозитория, а не от текущей папки."""
    p = Path(p).expanduser()
    return p if p.is_absolute() else PROJECT_ROOT / p


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    path = resolve_path(path) if path else DEFAULT_CONFIG
    if not path.exists():
        raise SystemExit(f"не найден конфиг {path}")
    cfg = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    for section in ("model", "videos", "output", "benchmark"):
        cfg.setdefault(section, {})
    return cfg
