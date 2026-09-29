#!/usr/bin/env bash
# Готовит уже созданную ВМ к ИИ-анализу: закрепляет публичный IP, привязывает сервисный аккаунт для YandexGPT
# и увеличивает память до 16 ГБ (RF-DETR + модель готовности + VLM). ВМ на минуту останавливается.
#
#   deploy/yandex-cloud/upgrade-vm.sh [имя_ВМ=oko-demo] [память_ГБ=16] [диск_ГБ=60]
# После — снова deploy/yandex-cloud/deploy.sh <IP>.
set -euo pipefail

NAME=${1:-oko-demo}
MEMORY=${2:-16}
DISK=${3:-60}
HERE=$(cd "$(dirname "$0")" && pwd)

IP=$(yc compute instance get --name "$NAME" --format yaml | awk '/one_to_one_nat:/{f=1} f && /address:/{print $2; exit}')
echo "→ ВМ $NAME, публичный IP $IP"

# без резервирования адрес после остановки ВМ сменится
ADDR_ID=$(yc vpc address list --format yaml | awk -v ip="$IP" '/^- id:/{id=$3} $1=="address:" && $2==ip {print id; exit}')
if [ -n "$ADDR_ID" ]; then
  yc vpc address update "$ADDR_ID" --reserved=true >/dev/null && echo "   IP $IP закреплён"
fi

# shellcheck source=service-account.sh
. "$HERE/service-account.sh"
SA_ID=$(ensure_service_account)

echo "→ остановка ВМ, $MEMORY ГБ памяти, сервисный аккаунт"
yc compute instance stop --name "$NAME" >/dev/null
yc compute instance update --name "$NAME" --memory "$MEMORY" --service-account-id "$SA_ID" >/dev/null

BOOT=$(yc compute instance get --name "$NAME" --format yaml | awk '/boot_disk:/{f=1} f && /disk_id:/{print $2; exit}')
CUR=$(yc compute disk get "$BOOT" --format yaml | awk '/^size:/{gsub(/"/, "", $2); print $2; exit}')
WANT=$((DISK * 1024 * 1024 * 1024))
if [ -n "$CUR" ] && [ "$CUR" -lt "$WANT" ]; then
  yc compute disk update "$BOOT" --size "$DISK" >/dev/null && echo "   диск увеличен до $DISK ГБ (раздел расширится при загрузке)"
fi

echo "→ запуск ВМ"
yc compute instance start --name "$NAME" >/dev/null
IP=$(yc compute instance get --name "$NAME" --format yaml | awk '/one_to_one_nat:/{f=1} f && /address:/{print $2; exit}')
cat <<EOF

Готово: $NAME, $MEMORY ГБ, IP $IP. Через минуту обновите сервис:
  deploy/yandex-cloud/deploy.sh $IP
EOF
