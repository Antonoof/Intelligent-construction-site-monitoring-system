"""Локальная визуально-языковая модель (VLM): что видно на снимке стройплощадки.

В ML-части независимую проверку делала Qwen3-VL 8B в bf16 (~17 ГБ видеопамяти). На сервере без такой
видеокарты модель выбирается под доступную память — от крупной к лёгкой, с тем же интерфейсом:

    Qwen3-VL-8B-Instruct (≈23 ГБ) → Qwen3-VL-4B-Instruct (≈12 ГБ) → Qwen3-VL-2B-Instruct (≈6 ГБ)
    → SmolVLM2-500M-Video-Instruct (≈2 ГБ, крайний случай: описывает сцену беднее и по-русски хуже)

OKO_VLM=auto — выбрать автоматически; OKO_VLM=<id модели Hugging Face> — задать явно; OKO_VLM=off — выключить.
Модель загружается при первом анализе, а не при старте сервиса, и остаётся в памяти.
Ответ — JSON: сцена, техника с количеством, стадия S1–S5, признаки работ, условия съёмки.
"""
from __future__ import annotations

import json
import logging
import os
import re
import threading
import time
from dataclasses import dataclass

from PIL import Image

log = logging.getLogger("oko.vlm")

# (id модели, параметров, млрд) — от крупной к лёгкой
LADDER: list[tuple[str, float]] = [
    ("Qwen/Qwen3-VL-8B-Instruct", 8.8),
    ("Qwen/Qwen3-VL-4B-Instruct", 4.4),
    ("Qwen/Qwen3-VL-2B-Instruct", 2.1),
    ("HuggingFaceTB/SmolVLM2-500M-Video-Instruct", 0.5),
]

PROMPT = """Ты — инженер строительного контроля. Перед тобой снимок камеры стройплощадки.
Опиши, что на нём видно, и ответь ТОЛЬКО JSON без пояснений. В equipment — по одной записи на каждую
единицу техники, box — рамка [x1, y1, x2, y2] в тысячных долях кадра (0–1000, от левого верхнего угла):
{
 "scene": "1–2 предложения: что происходит на площадке",
 "equipment": [{"type": "экскаватор | самосвал | каток | кран-манипулятор | автобетоносмеситель | бульдозер | грузовик | автокран | погрузчик | грейдер | башенный кран | буровая установка | автобетононасос | другое", "box": [0, 0, 1000, 1000], "state": "работает | стоит"}],
 "stage": "S1 подготовка и котлован | S2 фундамент | S3 каркас | S4 фасад и сети | S5 отделка и благоустройство",
 "activity": "работы ведутся | работы не ведутся | не определить",
 "people": 0,
 "conditions": "день | сумерки | ночь; ясно | дождь | снег | туман",
 "notes": "что мешает оценке или требует внимания"
}"""


def need_gb(params_b: float, bytes_per_param: float = 2.0) -> float:
    """Оценка памяти под модель в bf16: веса + ~25 % на кеш и активации + визуальный энкодер."""
    return params_b * bytes_per_param * 1.25 + 0.8


def choose_from(available_gb: float, ladder: list[tuple[str, float]] = LADDER) -> tuple[str | None, str]:
    """Самая крупная модель лестницы, которая помещается в available_gb; иначе None."""
    for name, params in ladder:
        if need_gb(params) <= available_gb:
            return name, f"нужно ≈{need_gb(params):.1f} ГБ, доступно {available_gb:.1f} ГБ"
    smallest = ladder[-1]
    return None, f"даже {smallest[0]} (≈{need_gb(smallest[1]):.1f} ГБ) не помещается в {available_gb:.1f} ГБ"


def _ram_available_gb() -> float:
    try:
        with open("/proc/meminfo", encoding="ascii") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) / 1024 / 1024
    except OSError:
        pass
    try:
        return os.sysconf("SC_AVPHYS_PAGES") * os.sysconf("SC_PAGE_SIZE") / 1024 ** 3
    except (ValueError, OSError, AttributeError):
        return 8.0


def _ram_total_gb() -> float:
    try:
        with open("/proc/meminfo", encoding="ascii") as f:
            for line in f:
                if line.startswith("MemTotal:"):
                    return int(line.split()[1]) / 1024 / 1024
    except OSError:
        pass
    return _ram_available_gb()


def ram_budget_gb(reserve_gb: float) -> tuple[float, str]:
    """Память под VLM на CPU по бюджету, а не только по MemAvailable: веса safetensors читаются через mmap и
    числятся «доступными» (страничный кеш), хотя заняты. Бюджет = вся память − детектор − модель готовности − запас."""
    parts, budget = [], _ram_total_gb() - reserve_gb
    try:
        from ..detection import get_detector
        if get_detector().name.startswith("rfdetr"):
            budget -= 1.5
            parts.append("RF-DETR 1.5")
    except Exception:
        pass
    try:
        from .readiness import get_readiness
        rd = get_readiness().footprint_gb()
        if rd:
            budget -= rd
            parts.append(f"модель готовности {rd:.1f}")
    except Exception:
        pass
    return budget, ", ".join(parts)


def resolve_device(pref: str) -> tuple[str, float]:
    """Устройство и доступная на нём память, ГБ."""
    try:
        import torch
        if pref in ("auto", "cuda") and torch.cuda.is_available():
            free, _total = torch.cuda.mem_get_info(0)
            return "cuda", free / 1024 ** 3
    except ImportError:
        pass
    return "cpu", _ram_available_gb()


def parse_json(text: str) -> dict:
    """JSON из ответа модели: целиком, из блока ```json или самый большой объект {...} в тексте; рассуждения
    <think>…</think> отбрасываются. Не нашёлся — {"raw": текст}."""
    raw = text
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S)
    text = re.sub(r"^.*?</think>", "", text, flags=re.S).strip()      # начало рассуждения обрезано
    for candidate in (text, *re.findall(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.S)):
        try:
            v = json.loads(candidate)
            if isinstance(v, dict):
                return v
        except json.JSONDecodeError:
            continue
    dec, best, best_len = json.JSONDecoder(), None, 0
    for mt in re.finditer(r"\{", text):
        try:
            v, end = dec.raw_decode(text, mt.start())
        except json.JSONDecodeError:
            continue
        if isinstance(v, dict) and end - mt.start() > best_len:
            best, best_len = v, end - mt.start()
    return best if best is not None else {"raw": raw.strip()}


def normalize_boxes(out: dict, size: tuple[int, int], per_mille: bool = True) -> dict:
    """Рамки VLM → доли кадра 0..1. Qwen3-VL отвечает в тысячных долях кадра (0–1000, per_mille); другие модели —
    в пикселях поданного им кадра; значения ≤ 1 — уже доли. Рамки без координат остаются без box."""
    w, h = size
    for it in out.get("equipment") or []:
        if not isinstance(it, dict):
            continue
        b = it.get("box") or it.get("bbox_2d") or it.get("bbox")
        it.pop("bbox_2d", None)
        it.pop("bbox", None)
        try:
            b = [float(v) for v in b]
        except (TypeError, ValueError):
            it.pop("box", None)
            continue
        if len(b) != 4:
            it.pop("box", None)
            continue
        scale = (1, 1) if max(b) <= 1.0 else (1000, 1000) if per_mille and max(b) <= 1000 else (w, h)
        x1, y1, x2, y2 = (min(1.0, max(0.0, v / s)) for v, s in zip(b, (scale[0], scale[1], scale[0], scale[1])))
        if x2 - x1 < 0.005 or y2 - y1 < 0.005 or (x1, y1, x2, y2) == (0.0, 0.0, 1.0, 1.0):
            it.pop("box", None)                 # вырожденная рамка или «весь кадр» из примера в запросе
            continue
        it["box"] = [round(x1, 3), round(y1, 3), round(x2, 3), round(y2, 3)]
    return out


@dataclass
class VLMResult:
    model: str
    device: str
    output: dict
    seconds: float


class LocalVLM:
    def __init__(self, setting: str, device_pref: str = "auto", max_tokens: int = 400, reserve_gb: float = 1.5,
                 image_px: int = 768):
        self.setting = setting
        self.device_pref = device_pref
        self.max_tokens = max_tokens
        self.reserve_gb = reserve_gb
        self.image_px = image_px
        self._lock = threading.Lock()
        self._model = None
        self._processor = None
        self.model_name: str | None = None
        self.device: str | None = None
        self.reason = ""

    @property
    def enabled(self) -> bool:
        return self.setting.lower() not in ("", "off", "0", "false", "no")

    def plan(self) -> dict:
        """Какая модель будет загружена и почему — без загрузки (для /api/ai/status)."""
        if not self.enabled:
            return {"enabled": False}
        if self.model_name:
            return {"enabled": True, "model": self.model_name, "device": self.device, "loaded": True,
                    "reason": self.reason}
        device, avail = resolve_device(self.device_pref)
        if self.setting.lower() == "auto":
            try:                                  # модель готовности загрузится раньше VLM — оставляем ей место
                from .readiness import get_readiness
                avail -= get_readiness().pending_gb()
            except Exception:
                pass
            avail -= self.reserve_gb
            note = ""
            if device == "cpu":
                budget, parts = ram_budget_gb(self.reserve_gb)
                if budget < avail:
                    avail, note = budget, f" (бюджет ОЗУ за вычетом: {parts})" if parts else ""
            name, why = choose_from(avail)
            why += note
        else:
            name, why = self.setting, "задана в OKO_VLM"
        return {"enabled": True, "model": name, "device": device, "loaded": False, "reason": why}

    def _load(self) -> None:
        import torch
        from transformers import AutoProcessor
        try:
            from transformers import AutoModelForImageTextToText as AutoVLM
        except ImportError:                      # старые версии transformers
            from transformers import AutoModelForVision2Seq as AutoVLM
        plan = self.plan()
        if not plan.get("model"):
            raise RuntimeError(f"VLM не помещается в память: {plan.get('reason')}")
        name, device = plan["model"], plan["device"]
        dtype = torch.bfloat16 if device == "cpu" or torch.cuda.is_bf16_supported() else torch.float16
        t0 = time.perf_counter()
        kw = {"low_cpu_mem_usage": True}
        try:
            import accelerate  # noqa: F401 — с ним веса сразу грузятся на видеокарту, минуя копию в ОЗУ
            kw["device_map"] = device
        except ImportError:
            pass
        try:
            model = AutoVLM.from_pretrained(name, dtype=dtype, **kw)
        except TypeError:                        # transformers < 4.56: параметр назывался torch_dtype
            model = AutoVLM.from_pretrained(name, torch_dtype=dtype, **kw)
        self._model = (model if "device_map" in kw else model.to(device)).eval()
        self._processor = AutoProcessor.from_pretrained(name)
        self.model_name, self.device, self.reason = name, device, plan["reason"]
        log.info("VLM %s загружена на %s за %.0f с (%s)", name, device, time.perf_counter() - t0, plan["reason"])

    def describe(self, img: Image.Image, hint: str = "") -> VLMResult:
        """Описание снимка. hint — короткий контекст (камера, этапы графика), не обязателен."""
        import torch
        with self._lock:
            if self._model is None:
                self._load()
            im = img.convert("RGB")
            im.thumbnail((self.image_px, self.image_px))   # меньше токенов изображения — быстрее на CPU
            text = PROMPT + (f"\nКонтекст: {hint}" if hint else "")
            messages = [{"role": "user", "content": [{"type": "image", "image": im}, {"type": "text", "text": text}]}]
            t0 = time.perf_counter()
            inputs = self._processor.apply_chat_template(messages, tokenize=True, add_generation_prompt=True,
                                                         return_dict=True, return_tensors="pt").to(self.device)
            with torch.inference_mode():
                out = self._model.generate(**inputs, max_new_tokens=self.max_tokens, do_sample=False)
            answer = self._processor.batch_decode(out[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)[0]
            per_mille = "qwen3" in self.model_name.lower()      # Qwen3-VL: 0–1000; Qwen2.5-VL, SmolVLM — пиксели
            return VLMResult(self.model_name, self.device, normalize_boxes(parse_json(answer), im.size, per_mille),
                             time.perf_counter() - t0)


_vlm: LocalVLM | None = None


def get_vlm() -> LocalVLM:
    global _vlm
    if _vlm is None:
        from ..config import settings
        _vlm = LocalVLM(settings.vlm, settings.vlm_device, settings.vlm_max_tokens, settings.vlm_reserve_gb,
                        settings.vlm_image_px)
    return _vlm


def set_vlm(v: LocalVLM | None) -> None:
    """Подмена VLM (тесты)."""
    global _vlm
    _vlm = v
