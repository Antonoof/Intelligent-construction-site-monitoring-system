#!/usr/bin/env bash
# Выкладка ОКО на ВМ Yandex Cloud: копирует код и веса (RF-DETR и модель готовности), собирает образ, запускает,
# ждёт готовности и в фоне скачивает веса DINOv2 и VLM. YandexGPT: каталог берётся из yc config.
# Повторный запуск обновляет демо до текущего кода (данные в томах сохраняются).
#
#   deploy/yandex-cloud/deploy.sh <публичный_IP> [пользователь=oko]
set -euo pipefail

HOST=${1:?укажите публичный IP ВМ: deploy/yandex-cloud/deploy.sh 51.250.0.1}
USER_=${2:-oko}
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
# ConnectTimeout и ServerAlive — чтобы скрипт падал с сообщением, а не висел, если ВМ не отвечает
SSH="ssh -o StrictHostKeyChecking=accept-new -o ConnectTimeout=15 -o ServerAliveInterval=15 -o ServerAliveCountMax=8 $USER_@$HOST"

# веса: одна .pth в weights/ — её имя уйдёт в .env, если своего .env нет
shopt -s nullglob
PTH=("$ROOT"/weights/*.pth)
[ ${#PTH[@]} -gt 0 ] || { echo "Нет весов: положите файл .pth в $ROOT/weights/"; exit 1; }
if [ ! -f "$ROOT/deploy/yandex-cloud/.env" ] && [ ${#PTH[@]} -gt 1 ]; then
  echo "В weights/ несколько .pth — укажите нужный в deploy/yandex-cloud/.env (OKO_WEIGHTS_FILE)"; exit 1
fi
WEIGHTS=$(basename "${PTH[0]}")

echo "→ проверяю ВМ $HOST"
if ! $SSH 'echo "   $(uptime -p), свободно памяти $(awk "/MemAvailable/{printf \"%.1f\", \$2/1048576}" /proc/meminfo) ГБ"'; then
  echo "ВМ не отвечает по SSH. Проверьте, что она запущена: yc compute instance list"
  echo "Если запущена, но не отвечает (нехватка памяти): yc compute instance restart <имя ВМ>"
  exit 1
fi
# Docker уже стоит (повторная выкладка) — cloud-init не ждём; sudo -n не ждёт пароль, а сразу сообщает об ошибке
if ! $SSH 'sudo -n docker compose version >/dev/null 2>&1'; then
  echo "→ жду, пока cloud-init поставит Docker (до 10 минут)"
  $SSH 'sudo -n timeout 300 cloud-init status --wait >/dev/null 2>&1 || true
        for i in $(seq 1 60); do sudo -n docker compose version >/dev/null 2>&1 && exit 0; sleep 5; done
        echo "Docker не появился. Состояние cloud-init:"; sudo -n cloud-init status --long; exit 1'
fi

echo "→ копирую код и веса (без документов и кешей)"
# COPYFILE_DISABLE=1 — tar на macOS не добавляет служебные файлы ._имя с метаданными Finder
COPYFILE_DISABLE=1 tar -C "$ROOT" \
  --exclude=.git --exclude=.venv --exclude='data/runtime' --exclude='__pycache__' --exclude='*.pyc' \
  --exclude='._*' --exclude='.DS_Store' \
  --exclude='docs/*.pptx' --exclude='docs/*.pdf' --exclude='docs/*.docx' \
  -czf - . | $SSH 'mkdir -p ~/oko && tar --warning=no-unknown-keyword -xzf - -C ~/oko \
                   && find ~/oko \( -name "._*" -o -name ".DS_Store" \) -type f -delete'
$SSH "cd ~/oko && [ -f deploy/yandex-cloud/.env ] || echo 'OKO_WEIGHTS_FILE=$WEIGHTS' > deploy/yandex-cloud/.env"
# каталог Yandex Cloud для YandexGPT: из настроек yc, если в .env его ещё нет
FOLDER=$(yc config get folder-id 2>/dev/null || true)
if [ -n "$FOLDER" ]; then
  $SSH "cd ~/oko/deploy/yandex-cloud && { grep -q '^OKO_YC_FOLDER_ID=.' .env || { sed -i '/^OKO_YC_FOLDER_ID=/d' .env; echo 'OKO_YC_FOLDER_ID=$FOLDER' >> .env; }; }"
fi
MEM=$($SSH "awk '/MemTotal/{printf \"%d\", \$2/1024/1024 + 0.5}' /proc/meminfo")
if [ "${MEM:-0}" -lt 15 ]; then
  echo "   ВНИМАНИЕ: на ВМ $MEM ГБ памяти, а RF-DETR + модель готовности + VLM требуют 16 ГБ."
  echo "   Увеличить: deploy/yandex-cloud/upgrade-vm.sh <имя ВМ>. Пока слои, которым не хватит памяти, будут пропущены."
fi

echo "→ сборка и запуск (первый раз 5–15 минут: PyTorch и rfdetr)"
$SSH 'cd ~/oko && sudo docker compose -f deploy/yandex-cloud/docker-compose.yml up -d --build'

echo "→ жду, пока загрузится модель"
for _ in $(seq 1 60); do
  if H=$(curl -fs --max-time 5 "http://$HOST/api/health"); then
    echo "$H"
    echo
    echo "Готово: http://$HOST  ·  API: http://$HOST/docs"
    # веса DINOv2 (модель готовности) и VLM — в фоне, чтобы первый «Анализ ИИ» не ждал загрузки (~9 ГБ)
    $SSH 'cd ~/oko && sudo docker compose -f deploy/yandex-cloud/docker-compose.yml exec -d app \
          sh -c "python -m app.ai.prefetch > /data/prefetch.log 2>&1"' || true
    echo "Модели ИИ-слоя скачиваются в фоне несколько минут. Проверить:"
    echo "  ssh $USER_@$HOST 'sudo docker compose -f ~/oko/deploy/yandex-cloud/docker-compose.yml exec app tail -3 /data/prefetch.log'"
    exit 0
  fi
  sleep 10
done
echo "Сервис не ответил за 10 минут. Логи:"
$SSH 'cd ~/oko && sudo docker compose -f deploy/yandex-cloud/docker-compose.yml logs --tail 80 app'
exit 1
