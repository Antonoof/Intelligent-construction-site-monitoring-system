# Публичное демо ОКО с RF-DETR в Yandex Cloud

Одна ВМ Ubuntu 24.04 (4 vCPU, 8 ГБ, без GPU) с Docker: PostgreSQL и сервис ОКО с детектором RF-DETR на CPU.
Демо открыто по адресу `http://<публичный IP>`, Swagger — `/docs`.

| Файл | Что делает |
|---|---|
| `create-vm.sh` | создаёт группу безопасности (порты 22, 80, 443) и ВМ с публичным IP; Docker ставится через cloud-init |
| `cloud-init.yaml` | пользователь `oko` с вашим SSH-ключом, Docker, зеркало Docker Hub, swap 4 ГБ |
| `deploy.sh` | копирует код и веса на ВМ, собирает образ и запускает; повторный запуск — обновление |
| `docker-compose.yml` | PostgreSQL + сервис из `backend/Dockerfile.ai` (PyTorch CPU + rfdetr), порт 80 |
| `.env.example` | имя файла весов, порядок классов, вариант модели, порог уверенности |

Скрипты — для bash (Linux, macOS, WSL или Git Bash на Windows).

## 1. Подготовка

1. Аккаунт Yandex Cloud с платёжным аккаунтом и каталогом.
2. [yc CLI](https://yandex.cloud/ru/docs/cli/quickstart): установить и выполнить `yc init`.
3. SSH-ключ: `ssh-keygen -t ed25519`, если его ещё нет.
4. Веса детектора — в `weights/` в корне репозитория, например `weights/rfdetr_large_best_ema.pth`.
   Если в чекпойнте нет имён классов, задайте их порядок в `deploy/yandex-cloud/.env`
   (скопируйте `.env.example`), см. [weights/README.md](../../weights/README.md).

## 2. Создать ВМ

```bash
deploy/yandex-cloud/create-vm.sh            # имя oko-demo, зона ru-central1-a
```

Скрипт печатает публичный IP. Сделайте его статическим, чтобы ссылка для жюри не поменялась после
остановки ВМ:

```bash
yc vpc address list
yc vpc address update --reserved=true <id адреса>
```

Без CLI то же самое делается в консоли: Compute Cloud → «Создать ВМ» → Ubuntu 24.04, Intel Ice Lake,
4 vCPU, 8 ГБ, диск SSD 40 ГБ, публичный адрес «Автоматически», группа безопасности с входящими портами 22 и 80;
в «Метаданных» — `user-data` с содержимым `cloud-init.yaml`, где `__SSH_PUBLIC_KEY__` заменён вашим публичным ключом.

## 3. Выложить демо

```bash
deploy/yandex-cloud/deploy.sh <публичный IP>
```

Первая сборка — 5–15 минут: PyTorch и rfdetr. Скрипт ждёт `/api/health` и печатает версию детектора:
`"detector": "rfdetr-large-10cls:…"` — модель подключена.

## 4. Эксплуатация

| Задача | Команда (на ВМ, в `~/oko`) |
|---|---|
| Логи сервиса | `sudo docker compose -f deploy/yandex-cloud/docker-compose.yml logs -f app` |
| Перезапуск | `sudo docker compose -f deploy/yandex-cloud/docker-compose.yml restart app` |
| Сбросить данные к демо | `sudo docker compose -f deploy/yandex-cloud/docker-compose.yml down -v`, затем `deploy.sh` |
| Обновить код или веса | `deploy.sh <IP>` с ноутбука |
| Остановить ВМ после показа | `yc compute instance stop <имя ВМ>` (статический IP сохраняется) |

## Если что-то не так

- **Образы не скачиваются (403, 429, timeout).** Docker Hub из российских облаков работает нестабильно; cloud-init
  прописывает зеркало `mirror.gcr.io` в `/etc/docker/daemon.json`. Можно добавить другое зеркало и выполнить
  `sudo systemctl restart docker`.
- **Модель не загружается: ошибка сети при старте.** RF-DETR при сборке модели может скачать конфигурацию
  DINOv2 с Hugging Face (кеш — в томе `/data/hf`, скачивается один раз). Если huggingface.co недоступен, задайте
  `HF_ENDPOINT` в `.env`.
- **Рамки подписаны не теми классами** — порядок классов не совпадает с обучением: `OKO_RFDETR_CLASSES` в `.env`.
- **Медленно.** RF-DETR Large на CPU — секунды на снимок; галочка «детекция по фрагментам 3×3» запускает модель
  10 раз. Для быстрой детекции — ВМ с GPU и профиль `gpu` из корневого `docker-compose.yml`.
- **Безопасность.** Демо открыто всем, у кого есть ссылка: загрузки снимков и вердикты доступны без входа.
  После защиты остановите ВМ; для постоянной работы нужны вход и HTTPS (раздел 8 `backend/readme.md`).
