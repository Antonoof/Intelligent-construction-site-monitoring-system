"""Кеш выводов детерминированных слоёв ИИ-анализа по содержимому кадра.

Модель готовности, Grounding DINO и VLM (жадное декодирование) на одном и том же кадре с теми же настройками дают
тот же ответ. Поэтому их вывод сохраняется на диск (data_dir/ai_cache) с ключом из SHA-256 кадра, имени модели и
всех параметров, от которых зависит ответ (подсказки, порог, рамки детектора, контекст для VLM). «Повторить анализ»,
повторная быстрая проверка того же фото и анализ снимка общего плана, уже оценённого в фоне при загрузке, заново
вызывают только LLM — остальные слои берутся из кеша без потери качества.

OKO_AI_CACHE=0 — выключить.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import tempfile
from pathlib import Path

from ..config import settings

log = logging.getLogger("oko.ai.cache")
VERSION = 1                     # меняется, если меняется формат вывода слоя


def key(*parts) -> str:
    raw = json.dumps([VERSION, *parts], ensure_ascii=False, sort_keys=True, default=str)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _path(layer: str, k: str) -> Path:
    return settings.data_dir / "ai_cache" / layer / k[:2] / f"{k}.json"


def get(layer: str, k: str) -> dict | None:
    if not settings.ai_cache:
        return None
    p = _path(layer, k)
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def put(layer: str, k: str, value: dict) -> None:
    if not settings.ai_cache:
        return
    p = _path(layer, k)
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=p.parent, suffix=".tmp")
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(value, f, ensure_ascii=False)
        os.replace(tmp, p)                  # запись целиком или никак: параллельный читатель не увидит половину
    except OSError as e:
        log.warning("кеш %s не записан: %s", layer, e)


def cached(layer: str, k: str, compute):
    """Вывод слоя из кеша или compute() с сохранением. Возвращает (вывод, взят_из_кеша)."""
    hit = get(layer, k)
    if hit is not None:
        return hit, True
    value = compute()
    put(layer, k, value)
    return value, False
