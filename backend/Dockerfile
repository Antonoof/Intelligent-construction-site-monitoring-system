# ОКО — основной сервис (API + веб-интерфейс + движок правил). CPU, без PyTorch.
# Сборка из корня репозитория:  docker build -f backend/Dockerfile -t oko-app .
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1
RUN apt-get update && apt-get install -y --no-install-recommends fonts-dejavu-core curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /srv/backend
COPY backend/requirements.txt ./requirements.txt
RUN pip install -r requirements.txt

COPY backend/app ./app
COPY backend/tests ./tests
COPY methodology /srv/methodology
COPY frontend/app /srv/frontend/app
COPY data/demo /srv/data/demo

ENV OKO_METHODOLOGY_DIR=/srv/methodology \
    OKO_FRONTEND_DIR=/srv/frontend/app \
    OKO_DEMO_DIR=/srv/data/demo \
    OKO_DATA_DIR=/data \
    OKO_RFDETR_WEIGHTS=/srv/weights/rfdetr_large_best_ema.pth \
    OKO_ONNX_MODEL=/srv/weights/rfdetr_v2.onnx
VOLUME ["/data"]
EXPOSE 8000
HEALTHCHECK --interval=15s --timeout=5s --start-period=20s CMD curl -fs http://localhost:8000/api/health || exit 1
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
