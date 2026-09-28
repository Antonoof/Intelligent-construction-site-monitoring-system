"""Настройки сервиса. Все значения читаются из переменных окружения (или файла .env рядом с backend/).

Ноутбук без GPU (по умолчанию): SQLite, детектор `auto` — RF-DETR на CPU, если есть PyTorch и веса,
иначе RF-DETR в ONNX, если есть модель, иначе демо-разметка. Docker: PostgreSQL и те же настройки через docker-compose.yml.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parents[1]
REPO_DIR = BACKEND_DIR.parent


def _load_dotenv() -> None:
    env = BACKEND_DIR / ".env"
    if not env.exists():
        return
    for line in env.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


_load_dotenv()


def _path(env: str, default: Path) -> Path:
    v = os.environ.get(env)
    return Path(v).expanduser().resolve() if v else default


@dataclass(frozen=True)
class Settings:
    # где лежат данные: БД SQLite, загруженные снимки, отчёты
    data_dir: Path = field(default_factory=lambda: _path("OKO_DATA_DIR", REPO_DIR / "data" / "runtime"))
    database_url: str = field(default_factory=lambda: os.environ.get("OKO_DATABASE_URL", ""))
    methodology_dir: Path = field(default_factory=lambda: _path("OKO_METHODOLOGY_DIR", REPO_DIR / "methodology"))
    frontend_dir: Path = field(default_factory=lambda: _path("OKO_FRONTEND_DIR", REPO_DIR / "frontend" / "app"))
    demo_dir: Path = field(default_factory=lambda: _path("OKO_DEMO_DIR", REPO_DIR / "data" / "demo"))

    # детектор: auto | rfdetr | onnx | http | demo
    detector: str = field(default_factory=lambda: os.environ.get("OKO_DETECTOR", "auto"))
    # веса RF-DETR из ML-части (weights/rfdetr_large_best_ema.pth)
    rfdetr_weights: Path = field(default_factory=lambda: _path("OKO_RFDETR_WEIGHTS", REPO_DIR / "weights" / "rfdetr_large_best_ema.pth"))
    rfdetr_variant: str = field(default_factory=lambda: os.environ.get("OKO_RFDETR_VARIANT", "large"))
    # порядок классов чекпойнта, если в нём нет имён (через запятую, названия или синонимы)
    rfdetr_class_names: str = field(default_factory=lambda: os.environ.get("OKO_RFDETR_CLASSES", ""))
    # собственные веса команды: разрешаем полную распаковку чекпойнта (старый формат с argparse.Namespace)
    rfdetr_trust: bool = field(default_factory=lambda: os.environ.get("OKO_RFDETR_TRUST", "1") not in ("0", "false", "no"))
    # RF-DETR в ONNX для CPU (training/08_export_cpu.py); рядом — <имя>.classes.json
    onnx_model: Path = field(default_factory=lambda: _path("OKO_ONNX_MODEL", REPO_DIR / "weights" / "rfdetr_v2.onnx"))
    device: str = field(default_factory=lambda: os.environ.get("OKO_DEVICE", "auto"))  # auto | cpu | cuda
    detector_threshold: float = field(default_factory=lambda: float(os.environ.get("OKO_DETECTOR_THRESHOLD", "0.3")))
    # удалённый AI-сервис на GPU (архитектура: backend ⇄ AI-сервис)
    ai_url: str = field(default_factory=lambda: os.environ.get("OKO_AI_URL", "http://ai:8001"))

    # загрузка демо-проекта при первом старте
    seed_demo: bool = field(default_factory=lambda: os.environ.get("OKO_SEED_DEMO", "1") not in ("0", "false", "no"))
    max_upload_mb: int = field(default_factory=lambda: int(os.environ.get("OKO_MAX_UPLOAD_MB", "25")))

    def db_url(self) -> str:
        if self.database_url:
            return self.database_url
        self.data_dir.mkdir(parents=True, exist_ok=True)
        return f"sqlite:///{self.data_dir / 'oko.db'}"

    @property
    def snapshots_dir(self) -> Path:
        p = self.data_dir / "snapshots"
        p.mkdir(parents=True, exist_ok=True)
        return p


settings = Settings()
