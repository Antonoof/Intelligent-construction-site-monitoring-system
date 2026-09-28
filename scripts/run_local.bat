@echo off
rem Запуск на ноутбуке без Docker (Windows): виртуальное окружение, зависимости, сервис на :8000.
cd /d "%~dp0\.."
python -m venv .venv
call .venv\Scripts\activate.bat
pip install -q -r backend\requirements.txt
cd backend
echo OKO: http://localhost:8000   API: http://localhost:8000/docs
uvicorn app.main:app --host 0.0.0.0 --port 8000
