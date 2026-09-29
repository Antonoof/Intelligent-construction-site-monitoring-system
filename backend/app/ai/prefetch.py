"""Заранее скачать веса моделей ИИ-слоя в кеш Hugging Face (HF_HOME), чтобы первый анализ не ждал загрузки.

    python -m app.ai.prefetch          # DINOv2 (модель готовности) и VLM, которую выберет OKO_VLM=auto

deploy/yandex-cloud/deploy.sh запускает это в фоне после старта контейнера: DINOv2 ViT-g ≈ 4.5 ГБ,
Qwen3-VL-2B ≈ 4.5 ГБ — на канале ВМ это несколько минут.
"""
from __future__ import annotations

import logging

from huggingface_hub import snapshot_download

from .readiness import get_readiness
from .vlm import get_vlm

log = logging.getLogger("oko.prefetch")
PATTERNS = ["*.json", "*.safetensors", "*.txt", "*.model", "*.jinja", "tokenizer*", "*.py"]


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    names = []
    rd = get_readiness()
    if rd.enabled:
        import torch
        ck = torch.load(rd.run / "head.pt", map_location="cpu", weights_only=True)
        names.append(ck["backbone"])
    vlm = get_vlm()
    plan = vlm.plan()
    if plan.get("model"):
        names.append(plan["model"])
    elif vlm.enabled:
        log.warning("VLM не помещается в память: %s", plan.get("reason"))
    for name in names:
        log.info("скачиваю %s", name)
        snapshot_download(name, allow_patterns=PATTERNS)
    log.info("готово: %s", ", ".join(names) or "нечего скачивать")


if __name__ == "__main__":
    main()
