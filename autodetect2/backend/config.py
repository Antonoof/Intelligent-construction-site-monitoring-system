from __future__ import annotations

import os
from pathlib import Path

APP_NAME = "AutoDetect2"
APP_VERSION = "2.1.0"

PACKAGE_ROOT = Path(__file__).resolve().parent.parent
REPO_ROOT = PACKAGE_ROOT.parent

FRONTEND_DIR = PACKAGE_ROOT / "frontend"
TEMPLATES_DIR = FRONTEND_DIR / "templates"
STATIC_DIR = FRONTEND_DIR / "static"
AUTOFILE_DIR = PACKAGE_ROOT / "autofile"

WORKSPACE_DIR = Path(os.getenv("AUTODETECT2_WORKSPACE") or REPO_ROOT / "workspace")
PROJECTS_DIR = WORKSPACE_DIR / "projects"
UPLOADS_DIR = WORKSPACE_DIR / "uploads"
INBOX_DIR = WORKSPACE_DIR / "inbox"

# Опорный кэш эмбеддингов — необязательный и всегда внешний по отношению к
# репозиторию: датасеты меняются, а 150 МБ векторов от прошлого проекта только
# мешают и никогда не совпадают по именам файлов с новыми данными.
REFERENCE_DIR = Path(os.getenv("AUTODETECT2_REFERENCE") or WORKSPACE_DIR / "reference")

IMAGE_SUFFIXES = frozenset({".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"})
LABEL_SUFFIX = ".txt"

EMBED_DIM = 128
MAX_EMBED_IMAGES = 60_000

# Классы проекта «умный мониторинг стройплощадки». Порядок — это id класса,
# менять его задним числом нельзя: он вшит в каждый .txt с разметкой.
CLASS_NAMES = [
    "Dump truck",         # 0
    "Excavator",          # 1
    "Motor grader",       # 2
    "Tower crane",        # 3
    "Bulldozer",          # 4
    "Bucket loader",      # 5
    "Mixer",              # 6
    "Crane manipulator",  # 7
    "Autocran",           # 8
    "Drilling rig",       # 9
]

MODEL_FAMILIES = {
    "8": {"name": "yolov8", "imgsz": 1024, "label": "YOLOv8"},
    "11": {"name": "yolo11", "imgsz": 960, "label": "YOLO11"},
    "12": {"name": "yolo12", "imgsz": 768, "label": "YOLO12"},
    "26": {"name": "yolo26", "imgsz": 768, "label": "YOLO26"},
}
MODEL_SIZES = ("n", "m", "l", "x")

# Состояние кадра в конвейере разметки. Хранится в state.json проекта.
STATUS_NEW = "new"            # не размечено, никем не занято
STATUS_ASSIGNED = "assigned"  # отдано в задание — висит в ожидании
STATUS_DONE = "done"          # размечено и подтверждено
STATUS_REVIEW = "review"      # помечено как спорное, нужен второй взгляд


def ensure_workspace() -> None:
    PROJECTS_DIR.mkdir(parents=True, exist_ok=True)
    UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
    INBOX_DIR.mkdir(parents=True, exist_ok=True)
