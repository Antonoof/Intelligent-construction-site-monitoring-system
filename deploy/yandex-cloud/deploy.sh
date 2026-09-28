#!/usr/bin/env bash
# Выкладка ОКО с RF-DETR на ВМ Yandex Cloud: копирует код и веса, собирает образ, запускает, ждёт готовности.
# Повторный запуск обновляет демо до текущего кода (данные в томах сохраняются).
#
#   deploy/yandex-cloud/deploy.sh <публичный_IP> [пользователь=oko]
set -euo pipefail

HOST=${1:?укажите публичный IP ВМ: deploy/yandex-cloud/deploy.sh 51.250.0.1}
USER_=${2:-oko}
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
SSH="ssh -o StrictHostKeyChecking=accept-new $USER_@$HOST"

# веса: одна .pth в weights/ — её имя уйдёт в .env, если своего .env нет
shopt -s nullglob
PTH=("$ROOT"/weights/*.pth)
[ ${#PTH[@]} -gt 0 ] || { echo "Нет весов: положите файл .pth в $ROOT/weights/"; exit 1; }
if [ ! -f "$ROOT/deploy/yandex-cloud/.env" ] && [ ${#PTH[@]} -gt 1 ]; then
  echo "В weights/ несколько .pth — укажите нужный в deploy/yandex-cloud/.env (OKO_WEIGHTS_FILE)"; exit 1
fi
WEIGHTS=$(basename "${PTH[0]}")

echo "→ жду, пока cloud-init поставит Docker на $HOST"
$SSH 'cloud-init status --wait >/dev/null 2>&1 || true; until sudo docker compose version >/dev/null 2>&1; do sleep 5; done'

echo "→ копирую код и веса (без документов и кешей)"
tar -C "$ROOT" \
  --exclude=.git --exclude=.venv --exclude='data/runtime' --exclude='__pycache__' --exclude='*.pyc' \
  --exclude='docs/*.pptx' --exclude='docs/*.pdf' --exclude='docs/*.docx' \
  -czf - . | $SSH 'mkdir -p ~/oko && tar -xzf - -C ~/oko'
$SSH "cd ~/oko && [ -f deploy/yandex-cloud/.env ] || echo 'OKO_WEIGHTS_FILE=$WEIGHTS' > deploy/yandex-cloud/.env"

echo "→ сборка и запуск (первый раз 5–15 минут: PyTorch и rfdetr)"
$SSH 'cd ~/oko && sudo docker compose -f deploy/yandex-cloud/docker-compose.yml up -d --build'

echo "→ жду, пока загрузится модель"
for _ in $(seq 1 60); do
  if H=$(curl -fs --max-time 5 "http://$HOST/api/health"); then
    echo "$H"
    echo
    echo "Готово: http://$HOST  ·  API: http://$HOST/docs"
    exit 0
  fi
  sleep 10
done
echo "Сервис не ответил за 10 минут. Логи:"
$SSH 'cd ~/oko && sudo docker compose -f deploy/yandex-cloud/docker-compose.yml logs --tail 80 app'
exit 1
