#!/usr/bin/env bash
# Запуск на ноутбуке без Docker (Linux / macOS): виртуальное окружение, зависимости, сервис на :8000.
set -euo pipefail
cd "$(dirname "$0")/.."
python3 -m venv .venv
source .venv/bin/activate
pip install -q -r backend/requirements.txt
cd backend
echo "ОКО: http://localhost:8000  ·  API: http://localhost:8000/docs"
exec uvicorn app.main:app --host 0.0.0.0 --port 8000
