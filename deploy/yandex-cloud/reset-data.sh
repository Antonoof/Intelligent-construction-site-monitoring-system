#!/usr/bin/env bash
# Сброс стенда к исходному демо: удаляет всё, что внесено на стенде, — проекты, загруженные снимки, отклонения,
# вердикты, ИИ-анализы и их кеш, — и при запуске сервис заново создаёт демо-проект.
# Остаются код, собранный образ, .env и скачанные веса моделей (/data/hf, ≈10 ГБ) — заново ничего не качается.
#
#   deploy/yandex-cloud/reset-data.sh <публичный_IP> [пользователь=oko]
#   OKO_YES=1 deploy/yandex-cloud/reset-data.sh <IP>     # без вопроса «удалить?»
set -euo pipefail

HOST=${1:?укажите публичный IP ВМ: deploy/yandex-cloud/reset-data.sh 51.250.0.1}
USER_=${2:-oko}
SSH="ssh -o StrictHostKeyChecking=accept-new -o ConnectTimeout=15 -o ServerAliveInterval=15 -o ServerAliveCountMax=8 $USER_@$HOST"

if [ "${OKO_YES:-}" != 1 ]; then
  read -r -p "Удалить все данные стенда $HOST (проекты, снимки, вердикты, ИИ-анализы) и вернуть демо? [y/N] " a
  case "$a" in y|Y|yes|д|Д|да) ;; *) echo "Отменено"; exit 1 ;; esac
fi

$SSH 'set -e
  cd ~/oko
  C="sudo -n docker compose -f deploy/yandex-cloud/docker-compose.yml"
  echo "→ останавливаю сервис и базу"
  $C down
  echo "→ удаляю загруженные снимки и кеш ИИ-анализа (веса моделей в /data/hf остаются)"
  $C run --rm --no-deps -T app sh -c "rm -rf /data/snapshots /data/ai_cache /data/oko.db"
  echo "→ удаляю базу данных"
  sudo -n docker volume rm -f oko_pgdata >/dev/null
  echo "→ запускаю: пустая база, демо-проект создаётся при старте"
  $C up -d'

echo "→ жду, пока сервис поднимется"
for _ in $(seq 1 60); do
  if H=$(curl -fs --max-time 5 "http://$HOST/api/health"); then
    echo "$H"
    echo
    echo "Готово: стенд как после первой выкладки — http://$HOST"
    echo "Модели ИИ-слоя загружаются в память ещё несколько минут; «Анализ ИИ», нажатый раньше, дождётся их."
    exit 0
  fi
  sleep 10
done
echo "Сервис не ответил за 10 минут. Логи:"
$SSH 'cd ~/oko && sudo docker compose -f deploy/yandex-cloud/docker-compose.yml logs --tail 80 app'
exit 1
