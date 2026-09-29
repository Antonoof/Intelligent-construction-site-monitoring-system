"""Grounding DINO — объекты по текстовым подсказкам, как stage_open_vocab в training/03_predict.py ML-части.

Два вида запросов (подсказки — methodology/open_vocab.yaml):
  objects   — прочие объекты: рабочие, леса, опалубка, строящееся здание, котлован, ограждение. Каждая фраза —
              отдельный запрос, чтобы подпись рамки однозначно совпадала с фразой; у фразы может быть свой порог.
  equipment — техника одним запросом («excavator. dump truck. …»): проверка RF-DETR. Рамка, которая совпала с рамкой
              детектора (IoU ≥ 0.3), его подтверждает; рамка без пары — кандидат в пропущенную технику для LLM.

Фильтры, как в ML-части: рамка больше 60 % кадра отбрасывается (модель «нашла» фразу во всей сцене); прочий объект,
который совпал с техникой RF-DETR (IoU > 0.5) или с более уверенной рамкой (IoU > 0.6), не повторяется.

OKO_OPENVOCAB: auto (IDEA-Research/grounding-dino-base, если есть PyTorch и transformers) | id модели | off.
Модель ≈ 0.9 ГБ в float32, на CPU 4 vCPU — несколько секунд на запрос; загружается при первом анализе.
"""
from __future__ import annotations

import importlib.util
import logging
from contextlib import contextmanager
import threading
import time
from pathlib import Path

import yaml
from PIL import Image

from ..config import settings

log = logging.getLogger("oko.openvocab")

DEFAULTS = {
    "model": "IDEA-Research/grounding-dino-base", "threshold": 0.4, "max_area": 0.6,
    "objects": {"worker": "рабочий", "scaffolding": "леса", "formwork": "опалубка",
                "unfinished building": {"ru": "строящееся здание", "threshold": 0.55},
                "excavation pit": "котлован", "fence": "ограждение"},
    "equipment": {"threshold": 0.35, "prompts": {"excavator": "excavator", "dump truck": "dump_truck",
                                                 "wheel loader": "loader", "bulldozer": "bulldozer",
                                                 "concrete mixer": "concrete_mixer", "road roller": "roller",
                                                 "grader": "grader", "drilling rig": "drilling_rig"}},
}
PARAMS_B = {"grounding-dino-base": 0.233, "grounding-dino-tiny": 0.172}


def load_config(directory: Path | None = None) -> dict:
    p = (directory or settings.methodology_dir) / "open_vocab.yaml"
    cfg = dict(DEFAULTS)
    if p.exists():
        cfg.update(yaml.safe_load(p.read_text(encoding="utf-8")) or {})
    return cfg


def iou(a, b) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def phrase_for(label: str, prompts: list[str]) -> str | None:
    """Фраза запроса по ответу модели: Grounding DINO возвращает фразу целиком или её часть («truck»).
    Слова в фразах не повторяются, поэтому совпадение слова даёт фразу однозначно."""
    label = (label or "").strip().lower()
    if label in prompts:
        return label
    words = set(label.split())
    for p in prompts:
        if words & set(p.split()):
            return p
    return None


def filter_objects(found: list[dict], equipment_boxes: list[list[float]], area: float, max_area: float) -> list[dict]:
    """Фильтры ML-части: рамка на весь кадр, повтор техники RF-DETR, повтор более уверенной рамки."""
    out: list[dict] = []
    for b in sorted(found, key=lambda x: -x["conf"]):
        x1, y1, x2, y2 = b["xyxy"]
        if (x2 - x1) * (y2 - y1) > max_area * area:
            continue
        if any(iou(b["xyxy"], e) > 0.5 for e in equipment_boxes) or any(iou(b["xyxy"], k["xyxy"]) > 0.6 for k in out):
            continue
        out.append(b)
    return out


def match_equipment(found: list[dict], detections: list[dict], min_iou: float = 0.3
                    ) -> tuple[list[int], list[dict], list[dict]]:
    """Техника Grounding DINO против рамок RF-DETR (доли кадра): id подтверждённых рамок детектора,
    рамки без пары (кандидаты в пропущенную технику) и спорный класс (та же рамка, другой класс)."""
    confirmed, only, conflicts = set(), [], []
    for f in sorted(found, key=lambda x: -x["conf"]):
        best, best_iou = None, 0.0
        for d in detections:
            v = iou(f["box"], d["box"])
            if v > best_iou:
                best, best_iou = d, v
        if best is not None and best_iou >= min_iou:
            if best["cls"] == f["cls"]:
                confirmed.add(best["id"])
            elif best["id"] not in {c["detection_id"] for c in conflicts}:
                conflicts.append({"detection_id": best["id"], "detector_cls": best["cls"], "open_vocab_cls": f["cls"],
                                  "conf": f["conf"], "iou": round(best_iou, 2)})
            continue
        if not any(iou(f["box"], o["box"]) > 0.6 for o in only):
            only.append(f)
    return sorted(confirmed), only, conflicts


class OpenVocab:
    def __init__(self, setting: str, device_pref: str = "cpu"):
        self.setting = setting or "off"
        self.device_pref = device_pref
        self.cfg = load_config()
        self._lock = threading.Lock()
        self._model = self._proc = None
        self.device = None
        off = self.setting.lower() in ("", "off", "0", "false", "no")
        self.model_name = None if off else (self.cfg["model"] if self.setting.lower() == "auto" else self.setting)
        self.reason = "выключен" if off else ""
        if self.model_name and not all(importlib.util.find_spec(x) for x in ("torch", "transformers")):
            self.model_name, self.reason = None, "нет PyTorch и transformers (образ Dockerfile.ai)"

    @property
    def enabled(self) -> bool:
        return self.model_name is not None

    def footprint_gb(self) -> float:
        if not self.enabled:
            return 0.0
        params = next((v for k, v in PARAMS_B.items() if k in self.model_name), 0.233)
        return params * 4 * 1.3 + 0.3

    def plan(self) -> dict:
        if not self.enabled:
            return {"enabled": False, "reason": self.reason}
        return {"enabled": True, "model": self.model_name, "loaded": self._model is not None,
                "objects": list(self.cfg["objects"]), "equipment": list(self.cfg["equipment"]["prompts"])}

    def _load(self) -> None:
        from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

        from .vlm import resolve_device
        device, avail = resolve_device(self.device_pref)
        if avail < self.footprint_gb() + 0.5:
            raise RuntimeError(f"Grounding DINO нужно ≈{self.footprint_gb():.1f} ГБ, доступно {avail:.1f} ГБ")
        t0 = time.perf_counter()
        self._proc = AutoProcessor.from_pretrained(self.model_name)
        self._model = AutoModelForZeroShotObjectDetection.from_pretrained(self.model_name).to(device).eval()
        self.device = device
        log.info("Grounding DINO %s загружен на %s за %.0f с", self.model_name, device, time.perf_counter() - t0)

    def warm(self) -> None:
        """Загрузить модель заранее (после старта сервиса), чтобы первый анализ её не ждал."""
        with self._lock:
            if self.enabled and self._model is None:
                self._load()

    @contextmanager
    def _shared_backbone(self):
        """Признаки изображения (Swin) не зависят от текста: на одном кадре все запросы получают один и тот же
        pixel_values, поэтому backbone считается один раз, а для остальных фраз берётся готовый результат —
        ответ модели тот же, время слоя меньше."""
        try:
            bb = self._model.model.backbone
        except AttributeError:                       # другая версия transformers — без ускорения, но корректно
            yield
            return
        orig, memo = bb.forward, {}

        def forward(pixel_values, pixel_mask, *a, **kw):
            k = (pixel_values.data_ptr(), tuple(pixel_values.shape), pixel_mask.data_ptr() if pixel_mask is not None else 0)
            if k not in memo:
                memo[k] = orig(pixel_values, pixel_mask, *a, **kw)
            return memo[k]
        bb.forward = forward
        try:
            yield
        finally:
            del bb.forward                           # вернуть метод класса
            memo.clear()

    def _query(self, img: Image.Image, text: str, thr: float, pixels=None) -> list[tuple[list[float], float, str]]:
        import torch
        inputs = self._proc(images=img, text=text, return_tensors="pt").to(self.device)
        if pixels is not None:                       # тот же тензор кадра для всех фраз — backbone из памяти
            inputs["pixel_values"] = pixels[0]
            if pixels[1] is not None:
                inputs["pixel_mask"] = pixels[1]
        with torch.inference_mode():
            out = self._model(**inputs)
        kw = dict(text_threshold=thr, target_sizes=[img.size[::-1]])
        try:
            res = self._proc.post_process_grounded_object_detection(out, inputs.input_ids, threshold=thr, **kw)[0]
        except TypeError:                            # старые версии transformers
            res = self._proc.post_process_grounded_object_detection(out, inputs.input_ids, box_threshold=thr, **kw)[0]
        labels = res.get("text_labels") or res.get("labels") or [""] * len(res["scores"])
        return [(list(map(float, b)), float(s), str(lab)) for b, s, lab in
                zip(res["boxes"].tolist(), res["scores"].tolist(), labels)]

    def detect(self, img: Image.Image, equipment_boxes_px: list[list[float]]) -> dict:
        """Прочие объекты и техника по подсказкам. Рамки — в пикселях кадра (xyxy) и в долях (box)."""
        with self._lock:
            if self._model is None:
                self._load()
            img = img.convert("RGB")
            W, H = img.size
            area = W * H
            cfg = self.cfg
            t0 = time.perf_counter()

            def item(box, conf, **kw):
                return {"xyxy": [round(v, 1) for v in box], "box": [round(box[0] / W, 4), round(box[1] / H, 4),
                                                                     round(box[2] / W, 4), round(box[3] / H, 4)],
                        "conf": round(conf, 3), **kw}

            first = self._proc(images=img, text="object.", return_tensors="pt").to(self.device)
            pixels = (first["pixel_values"], first.get("pixel_mask"))
            found, equipment = [], []
            with self._shared_backbone():
                for en, val in cfg["objects"].items():
                    ru, thr = (val, cfg["threshold"]) if isinstance(val, str) else \
                        (val["ru"], float(val.get("threshold", cfg["threshold"])))
                    for box, conf, _ in self._query(img, f"{en}.", thr, pixels):
                        found.append(item(box, conf, label=ru, prompt=en))

                eq_cfg = cfg.get("equipment") or {}
                prompts = {k.lower(): v for k, v in (eq_cfg.get("prompts") or {}).items()}
                if prompts:
                    thr = float(eq_cfg.get("threshold", cfg["threshold"]))
                    for box, conf, lab in self._query(img, ". ".join(prompts) + ".", thr, pixels):
                        ph = phrase_for(lab, list(prompts))
                        if ph is None or (box[2] - box[0]) * (box[3] - box[1]) > float(cfg.get("max_area", 0.6)) * area:
                            continue
                        equipment.append(item(box, conf, cls=prompts[ph], prompt=ph))
            objects = filter_objects(found, equipment_boxes_px, area, float(cfg.get("max_area", 0.6)))
            return {"model": self.model_name, "objects": objects, "equipment": equipment,
                    "seconds": round(time.perf_counter() - t0, 1), "device": self.device}

    def cache_key(self, sha: str, equipment_boxes_px: list[list[float]]) -> str:
        from . import cache
        c = self.cfg
        return cache.key("open_vocab", self.model_name, c.get("threshold"), c.get("max_area"), c.get("objects"),
                         c.get("equipment"), sha, [[round(float(v), 1) for v in b] for b in equipment_boxes_px])


_ov: OpenVocab | None = None


def get_openvocab() -> OpenVocab:
    global _ov
    if _ov is None:
        _ov = OpenVocab(settings.openvocab, settings.device)
    return _ov


def set_openvocab(v) -> None:
    """Подмена (тесты)."""
    global _ov
    _ov = v


def summarize(ov: dict, detections: list[dict], zone_of=None) -> dict:
    """Итог слоя для контекста LLM и интерфейса: прочие объекты с зонами, сверка техники с RF-DETR."""
    from collections import Counter
    confirmed, only, conflicts = match_equipment(ov.get("equipment") or [], detections)
    objects = []
    for o in ov.get("objects") or []:
        objects.append({**o, "zone": zone_of(o["xyxy"]) if zone_of else None})
    return {"model": ov.get("model"), "seconds": ov.get("seconds"),
            "objects": objects, "counts": dict(Counter(o["label"] for o in objects)),
            "equipment_counts": dict(Counter(e["cls"] for e in ov.get("equipment") or [])),
            "detector_confirmed": confirmed, "only_open_vocab": only, "class_conflict": conflicts}
