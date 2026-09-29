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

    # ---- слой ИИ-анализа (app/ai): модель готовности, локальная VLM и LLM (YandexGPT / Claude / ChatGPT)
    # модель готовности ML-части (DINOv2 + голова): auto — weights/readiness/<прогон>/head.pt | путь | off
    readiness: str = field(default_factory=lambda: os.environ.get("OKO_READINESS", "auto"))
    readiness_dtype: str = field(default_factory=lambda: os.environ.get("OKO_READINESS_DTYPE", "float32"))
    # VLM: off | auto (самая крупная Qwen3-VL, которая помещается в память) | id модели Hugging Face
    vlm: str = field(default_factory=lambda: os.environ.get("OKO_VLM", "off"))
    vlm_device: str = field(default_factory=lambda: os.environ.get("OKO_VLM_DEVICE", "auto"))   # auto | cpu | cuda
    vlm_max_tokens: int = field(default_factory=lambda: int(os.environ.get("OKO_VLM_MAX_TOKENS", "700")))
    vlm_reserve_gb: float = field(default_factory=lambda: float(os.environ.get("OKO_VLM_RESERVE_GB", "1.5")))
    # длинная сторона кадра для VLM: меньше — быстрее на CPU (768 px ≈ 600 токенов изображения у Qwen3-VL)
    vlm_image_px: int = field(default_factory=lambda: int(os.environ.get("OKO_VLM_IMAGE_PX", "768")))
    # LLM: off | yandex (YandexGPT, Yandex AI Studio) | anthropic (Claude) | openai (ChatGPT и совместимые шлюзы)
    llm_provider: str = field(default_factory=lambda: os.environ.get("OKO_LLM_PROVIDER", "off"))
    # yandex: yandexgpt-5.1 (по умолчанию), yandexgpt-5-lite, aliceai-llm, qwen3.6-35b-a3b (видит изображения)
    llm_model: str = field(default_factory=lambda: os.environ.get("OKO_LLM_MODEL", ""))
    # yandex: API-ключ сервисного аккаунта; без ключа на ВМ Yandex Cloud берётся IAM-токен её сервисного аккаунта
    llm_api_key: str = field(default_factory=lambda: os.environ.get("OKO_LLM_API_KEY") or os.environ.get(
        {"yandex": "YC_API_KEY", "anthropic": "ANTHROPIC_API_KEY", "openai": "OPENAI_API_KEY"}.get(
            os.environ.get("OKO_LLM_PROVIDER", "off").lower(), "OKO_LLM_API_KEY"), ""))
    # yandex: каталог Yandex Cloud, в котором работает AI Studio (yc config get folder-id)
    yc_folder_id: str = field(default_factory=lambda: os.environ.get("OKO_YC_FOLDER_ID") or os.environ.get("YC_FOLDER_ID", ""))
    # адрес API или прокси: из России api.anthropic.com и api.openai.com обычно недоступны напрямую
    llm_base_url: str = field(default_factory=lambda: os.environ.get("OKO_LLM_BASE_URL", ""))
    # отправлять ли в LLM изображения: auto — если модель их понимает (Claude, GPT, Qwen3.6 в AI Studio)
    llm_images: str = field(default_factory=lambda: os.environ.get("OKO_LLM_IMAGES", "auto"))
    llm_timeout: float = field(default_factory=lambda: float(os.environ.get("OKO_LLM_TIMEOUT", "300")))
    # лимит токенов ответа LLM; если ответ обрезан на лимите, запрос повторяется один раз с вдвое большим
    llm_max_tokens: int = field(default_factory=lambda: int(os.environ.get("OKO_LLM_MAX_TOKENS", "8000")))
    # автоматический анализ каждого нового снимка (платные вызовы LLM) — по умолчанию только по кнопке
    ai_auto: bool = field(default_factory=lambda: os.environ.get("OKO_AI_AUTO", "0") in ("1", "true", "yes"))
    # минимальная уверенность LLM, чтобы её исправление (ложная рамка, пропущенная техника) можно было применить
    ai_apply_min_conf: float = field(default_factory=lambda: float(os.environ.get("OKO_AI_APPLY_MIN_CONF", "0.6")))
    # manual — исправления LLM применяет инженер кнопкой; auto — сразу после анализа (с возможностью отмены)
    ai_apply: str = field(default_factory=lambda: os.environ.get("OKO_AI_APPLY", "manual"))

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
