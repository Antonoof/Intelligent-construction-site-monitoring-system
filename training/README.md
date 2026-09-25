# Обучение и предсказание

Замороженные **DINOv2 ViT-g** (сцена) и **RF-DETR Large** (техника) + обучаемая голова: по кадру она выдаёт
готовность объекта 0–100% и стадию S1–S5. Все команды запускаются из корня репозитория.

## Установка

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install -r training/requirements.txt
```

Веса детектора положите в `weights/rfdetr_large_best_ema.pth`, таймлапсы — в `videos for train/`.

## Шаги

| шаг | команда | RTX 4090 | результат |
|---|---|---|---|
| 0 | `python training/00_prepare_frames.py` | ~4 мин | 4 692 кадра + метки |
| 1 | `python training/01_build_features.py` | ~12 мин | кеш признаков, 2.1 ГБ |
| 2 | `python training/02_train_head.py` | < 1 мин | `training/runs/<дата>/` |
| 3 | `python training/03_predict.py --demo 10` | ~10 с/кадр | `predictions/<дата>/` |
| 4 | `python training/04_report.py` | ~10 с | итоговые графики, примеры и «до/после» в `docs/assets/` |

```bash
python training/03_predict.py "videos for train/video3.mp4" --every 3   # кадр каждые 3 с видео
python training/03_predict.py photo.jpg --expected 0.6                  # своё фото, план 60%
python training/03_predict.py --no-vlm --no-openvocab                   # быстро: без Grounding DINO и Qwen3-VL
```

## Настройки — `config.yaml`

- `stages` — границы стадий как доли готовности; `videos.<файл>.stage_until` — свои границы для видео.
- `videos.<файл>.plan_finish`, `total_days` — план стройки: `plan_finish: 0.8` значит «по плану закончить к 80% срока».
- `backbone.name` — энкодер сцены. По умолчанию `facebook/dinov2-giant`, поддерживается и DINOv3 (`facebook/dinov3-*`).
- `detector.tiles`, `videos.<файл>.tiles` — детекция техники по фрагментам кадра для общих планов с высоты.
- `head.holdout_video` — видео, которое не участвует в обучении.

## Файлы

```
00_prepare_frames.py   видео → кадры + метки
01_build_features.py   кадры → кеш признаков DINOv2 и RF-DETR
02_train_head.py       кеш → голова
03_predict.py          кадры → панели, карты внимания, JSON
04_report.py           запуск → итоговые графики, примеры и проверка кадров для README
common.py              метки, энкодер, детектор, голова
render.py              отрисовка панели
```
