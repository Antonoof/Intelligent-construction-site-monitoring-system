#!/usr/bin/env bash
# Создаёт в Yandex Cloud ВМ для публичного демо ОКО: группа безопасности (22, 80, 443) и ВМ Ubuntu 24.04
# 4 vCPU / 8 ГБ / 40 ГБ SSD с публичным IP; Docker ставится через cloud-init.
#
# Нужен настроенный yc CLI (yc init) и публичный SSH-ключ (~/.ssh/id_ed25519.pub или ~/.ssh/id_rsa.pub).
#   deploy/yandex-cloud/create-vm.sh [имя_ВМ] [зона]
# Переменные: SSH_KEY — путь к публичному ключу, NETWORK — облачная сеть (default), SUBNET — подсеть (default-<зона>).
set -euo pipefail

NAME=${1:-oko-demo}
ZONE=${2:-ru-central1-a}
NETWORK=${NETWORK:-default}
SUBNET=${SUBNET:-default-$ZONE}
HERE=$(cd "$(dirname "$0")" && pwd)

KEY=${SSH_KEY:-}
for k in "$KEY" "$HOME/.ssh/id_ed25519.pub" "$HOME/.ssh/id_rsa.pub"; do
  if [ -n "$k" ] && [ -f "$k" ]; then KEY=$k; break; fi
done
[ -f "$KEY" ] || { echo "Нет публичного SSH-ключа. Создайте: ssh-keygen -t ed25519"; exit 1; }

yaml_id() { awk '/^id:/{print $2; exit}'; }

echo "→ группа безопасности oko-demo-sg"
SG_ID=$(yc vpc security-group get --name oko-demo-sg --format yaml 2>/dev/null | yaml_id || true)
if [ -z "$SG_ID" ]; then
  NET_ID=$(yc vpc network get --name "$NETWORK" --format yaml | yaml_id)
  SG_ID=$(yc vpc security-group create --name oko-demo-sg --network-id "$NET_ID" \
    --description "OKO demo: SSH, HTTP, HTTPS" \
    --rule "description=SSH,direction=ingress,port=22,protocol=tcp,v4-cidrs=[0.0.0.0/0]" \
    --rule "description=HTTP,direction=ingress,port=80,protocol=tcp,v4-cidrs=[0.0.0.0/0]" \
    --rule "description=HTTPS,direction=ingress,port=443,protocol=tcp,v4-cidrs=[0.0.0.0/0]" \
    --rule "description=any outgoing,direction=egress,from-port=1,to-port=65535,protocol=any,v4-cidrs=[0.0.0.0/0]" \
    --format yaml | yaml_id)
fi
echo "   $SG_ID"

USERDATA=$(mktemp)
trap 'rm -f "$USERDATA"' EXIT
sed "s|__SSH_PUBLIC_KEY__|$(tr -d '\n' < "$KEY")|" "$HERE/cloud-init.yaml" > "$USERDATA"

echo "→ ВМ $NAME в зоне $ZONE"
yc compute instance create \
  --name "$NAME" --hostname "$NAME" --zone "$ZONE" \
  --platform standard-v3 --cores 4 --memory 8 \
  --create-boot-disk image-folder-id=standard-images,image-family=ubuntu-2404-lts,size=40,type=network-ssd \
  --network-interface subnet-name="$SUBNET",nat-ip-version=ipv4,security-group-ids="$SG_ID" \
  --metadata-from-file user-data="$USERDATA" >/dev/null

IP=$(yc compute instance get --name "$NAME" --format yaml | awk '/one_to_one_nat:/{f=1} f && /address:/{print $2; exit}')
cat <<EOF

Готово. Публичный IP: $IP
Docker ставится ещё 2–3 минуты. Дальше:
  deploy/yandex-cloud/deploy.sh $IP

Чтобы IP не менялся после остановки ВМ, сделайте его статическим:
  yc vpc address list
  yc vpc address update --reserved=true <id адреса $IP>
EOF
