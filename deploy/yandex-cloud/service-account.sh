# Общая функция для create-vm.sh и upgrade-vm.sh: сервисный аккаунт oko-ai с ролью ai.languageModels.user
# в текущем каталоге. ВМ с этим аккаунтом получает IAM-токен из сервиса метаданных — YandexGPT работает без ключа.
# Печатает id аккаунта в stdout, сообщения — в stderr.

ensure_service_account() {
  local sa=${SA_NAME:-oko-ai} folder sa_id
  folder=$(yc config get folder-id)
  echo "→ сервисный аккаунт $sa (YandexGPT: роль ai.languageModels.user в каталоге $folder)" >&2
  sa_id=$(yc iam service-account get --name "$sa" --format yaml 2>/dev/null | awk '/^id:/{print $2; exit}' || true)
  if [ -z "$sa_id" ]; then
    sa_id=$(yc iam service-account create --name "$sa" --description "OKO: вызовы YandexGPT" --format yaml \
            | awk '/^id:/{print $2; exit}')
  fi
  yc resource-manager folder add-access-binding "$folder" --role ai.languageModels.user \
    --subject "serviceAccount:$sa_id" >/dev/null 2>&1 || true   # уже выдана — не ошибка
  echo "   $sa_id" >&2
  echo "$sa_id"
}
