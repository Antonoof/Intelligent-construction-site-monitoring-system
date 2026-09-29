# Публичное демо ОКО с RF-DETR и ИИ-анализом в Yandex Cloud

Одна ВМ Ubuntu 24.04 (4 vCPU, 16 ГБ, без GPU) с Docker: PostgreSQL и сервис ОКО — детектор RF-DETR на CPU
и слой ИИ-анализа: модель готовности ML-части (DINOv2 + обученная голова), локальная VLM (Qwen3-VL) и
YandexGPT в Yandex AI Studio, которая сверяет все слои, исправляет ошибки моделей и даёт прогноз.
Демо открыто по адресу `http://<публичный IP>`, Swagger — `/docs`.

| Файл | Что делает |
|---|---|
| `create-vm.sh` | группа безопасности (22, 80, 443), сервисный аккаунт для YandexGPT, ВМ 4 vCPU / 16 ГБ / 60 ГБ с публичным IP |
| `upgrade-vm.sh` | уже созданная ВМ → 16 ГБ, сервисный аккаунт, закреплённый IP (ВМ на минуту останавливается) |
| `service-account.sh` | общая функция: аккаунт `oko-ai` с ролью `ai.languageModels.user` в каталоге |
| `cloud-init.yaml` | пользователь `oko` с вашим SSH-ключом, Docker, зеркало Docker Hub, swap 4 ГБ |
| `deploy.sh` | копирует код и веса, собирает образ, запускает, в фоне скачивает DINOv2 и VLM; повторный запуск — обновление |
| `docker-compose.yml` | PostgreSQL + сервис из `backend/Dockerfile.ai` (PyTorch CPU, rfdetr, transformers), порт 80 |
| `.env.example` | веса детектора, YandexGPT (каталог, модель), модель готовности, VLM, автоанализ |

Скрипты — для bash (Linux, macOS, WSL или Git Bash на Windows).

## 1. Подготовка

1. Аккаунт Yandex Cloud с платёжным аккаунтом и каталогом.
2. [yc CLI](https://yandex.cloud/ru/docs/cli/quickstart): установить и выполнить `yc init`.
3. SSH-ключ: `ssh-keygen -t ed25519`, если его ещё нет.
4. Веса в `weights/` в корне репозитория:
   - детектор — `weights/rfdetr_large_best_ema.pth`;
   - модель готовности — `weights/readiness/<прогон>/head.pt` (+ `train_embed.npy`, `train_meta.csv`), уже в архиве;
     свежий прогон из `training/runs/` копируется туда же, имя — в `weights/readiness/latest.txt`.

## 2. Создать ВМ

```bash
deploy/yandex-cloud/create-vm.sh            # имя oko-demo, зона ru-central1-a, 16 ГБ
```

Скрипт создаёт сервисный аккаунт `oko-ai` с ролью `ai.languageModels.user` и привязывает его к ВМ: сервис
получает IAM-токен из сервиса метаданных ВМ, API-ключ для YandexGPT не нужен. Сделайте IP статическим:

```bash
yc vpc address list
yc vpc address update --reserved=true <id адреса>
```

**ВМ уже создана (8 ГБ, без сервисного аккаунта)** — одна команда, IP закрепляется автоматически:

```bash
deploy/yandex-cloud/upgrade-vm.sh oko-demo  # 16 ГБ, диск 60 ГБ, сервисный аккаунт
```

## 3. Выложить демо

```bash
deploy/yandex-cloud/deploy.sh <публичный IP>
```

Скрипт подставляет каталог (`yc config get folder-id`) в `.env`, собирает образ (первый раз 5–15 минут), ждёт
`/api/health` и в фоне скачивает веса DINOv2 ViT-g и VLM (≈9 ГБ, несколько минут). В строке состояния внизу
страницы — какие ИИ-слои включены.

Без сервисного аккаунта на ВМ (или вне Yandex Cloud) — API-ключ: консоль → Identity and Access Management →
Сервисные аккаунты → `oko-ai` → «Создать API-ключ» (область действия `yc.ai.languageModels.execute`);
секретную часть — в `OKO_LLM_API_KEY` файла `deploy/yandex-cloud/.env`.

## 4. ИИ-анализ в интерфейсе

- **Карточка снимка → «Анализ ИИ»** (и то же на вкладке «Проверить снимок» для своего фото): модель готовности оценивает стадию и готовность и сравнивает с планом по графику;
  VLM описывает кадр и отмечает технику рамками; рамки детектора и VLM сверяются; YandexGPT получает все слои и
  выносит вердикт по каждой рамке и каждому отклонению, добавляет пропущенную технику, даёт прогноз и
  рекомендации. «Применить исправления» — ложные рамки исключаются, пропущенная техника добавляется, отклонения
  пересчитываются; «Отменить» возвращает как было.
- **Сводка → «Анализ ИИ за день»**: статус площадки, риски срыва сроков, прогноз задержек по этапам, ошибки моделей.

| Настройка `.env` | По умолчанию | |
|---|---|---|
| `OKO_LLM_MODEL` | `yandexgpt-5.1` | `yandexgpt-5-lite` дешевле; `qwen3.6-35b-a3b` видит сами изображения |
| `OKO_VLM` | `auto` | самая крупная Qwen3-VL, которая помещается в память; `off` — без VLM |
| `OKO_READINESS` | `auto` | модель готовности из `weights/readiness/`; `off` — без неё |
| `OKO_AI_AUTO` | `0` | `1` — анализировать каждый новый снимок (платные вызовы YandexGPT) |
| `OKO_AI_APPLY` | `manual` | `auto` — применять исправления сразу (с возможностью отмены) |

Память на 16 ГБ: RF-DETR ≈1,5 ГБ, DINOv2 ViT-g ≈5 ГБ, Qwen3-VL-2B ≈6 ГБ. Слой, которому не хватает памяти,
пропускается с объяснением в карточке. На CPU анализ снимка — 1–2 минуты (DINOv2 ≈30 с, VLM ≈1 мин), YandexGPT —
секунды.

## 5. Эксплуатация

| Задача | Команда (на ВМ, в `~/oko`) |
|---|---|
| Логи сервиса | `sudo docker compose -f deploy/yandex-cloud/docker-compose.yml logs -f app` |
| Загрузка весов ИИ-слоя | `sudo docker compose -f deploy/yandex-cloud/docker-compose.yml exec app tail /data/prefetch.log` |
| Перезапуск | `sudo docker compose -f deploy/yandex-cloud/docker-compose.yml restart app` |
| Сбросить данные к демо | `sudo docker compose -f deploy/yandex-cloud/docker-compose.yml down -v`, затем `deploy.sh` |
| Обновить код или веса | `deploy.sh <IP>` с ноутбука |
| Остановить ВМ после показа | `yc compute instance stop <имя ВМ>` (статический IP сохраняется) |

## Если что-то не так

- **Нет блока «Анализ ИИ».** Браузер показывает старый `app.js`: обновите страницу с очисткой кеша
  (Ctrl+Shift+R, на Mac — Cmd+Shift+R). Внизу страницы должна быть строка «ИИ-анализ: …». Начиная с этой версии
  интерфейс отдаётся с `Cache-Control: no-cache`, и после обновления сервиса страница берёт новые файлы сама.

- **YandexGPT: 401/403.** Нет роли `ai.languageModels.user` у сервисного аккаунта ВМ или ВМ без аккаунта —
  `upgrade-vm.sh`, либо API-ключ в `OKO_LLM_API_KEY`. **400 «model not found»** — проверьте `OKO_YC_FOLDER_ID`
  и имя модели.
- **«модели готовности нужно ≈5.8 ГБ, доступно …»** — мало памяти: `upgrade-vm.sh` или
  `OKO_READINESS_DTYPE=bfloat16`. VLM в этом случае выбирает модель поменьше или отключается.
- **Образы не скачиваются (403, 429, timeout).** Docker Hub из российских облаков работает нестабильно; cloud-init
  прописывает зеркало `mirror.gcr.io` в `/etc/docker/daemon.json`.
- **Веса с Hugging Face не скачиваются.** Кеш — в томе `/data/hf`, скачивается один раз. Если huggingface.co
  недоступен, задайте `HF_ENDPOINT` (зеркало) в `.env`.
- **Рамки подписаны не теми классами** — порядок классов не совпадает с обучением: `OKO_RFDETR_CLASSES` в `.env`.
- **Безопасность.** Демо открыто всем, у кого есть ссылка: загрузки снимков, вердикты и запуск ИИ-анализа
  (платные вызовы YandexGPT) доступны без входа. После защиты остановите ВМ; для постоянной работы нужны вход и
  HTTPS (раздел 8 `backend/readme.md`).
