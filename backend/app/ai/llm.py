"""Внешняя LLM — последний слой: проверяет выводы всех моделей и правил и даёт итоговый анализ.

Провайдеры (OKO_LLM_PROVIDER):
  yandex    — YandexGPT в Yandex AI Studio (OpenAI-совместимый API https://ai.api.cloud.yandex.net/v1):
              модель gpt://<каталог>/yandexgpt-5.1, ответ по JSON-схеме (response_format=json_schema).
              Работает из Yandex Cloud без прокси. Авторизация — API-ключ сервисного аккаунта (OKO_LLM_API_KEY)
              или, если ключа нет, IAM-токен сервисного аккаунта ВМ из сервиса метаданных.
              YandexGPT не принимает изображения: о кадре она судит по описанию локальной VLM и рамкам
              детектора; модель qwen3.6-35b-a3b в AI Studio изображения принимает — тогда они отправляются.
              Рассуждения у моделей, которые рассуждают по умолчанию (Qwen3.x, gpt-oss), выключаются параметром
              reasoning_effort (OKO_LLM_REASONING): подсказку /no_think Qwen3.6 не понимает, а рассуждения
              втрое удлиняют ответ. Если API параметр не принимает, запрос повторяется без него.
  anthropic — Claude, Messages API; ответ через инструмент со схемой JSON.
  openai    — ChatGPT и любой OpenAI-совместимый шлюз (Chat Completions, response_format=json_object).

Зависимостей, кроме httpx, нет: запросы собираются вручную.
"""
from __future__ import annotations

import base64
import json
import time
from dataclasses import dataclass, field

import httpx

from .vlm import parse_json

DEFAULT_MODELS = {"yandex": "yandexgpt-5.1", "anthropic": "claude-sonnet-5", "openai": "gpt-5"}
DEFAULT_BASE = {"yandex": "https://ai.api.cloud.yandex.net/v1", "anthropic": "https://api.anthropic.com",
                "openai": "https://api.openai.com/v1"}
PROVIDER_NAMES = {"yandex": "YandexGPT", "anthropic": "Claude", "openai": "ChatGPT"}
YC_METADATA_TOKEN = "http://169.254.169.254/computeMetadata/v1/instance/service-accounts/default/token"
YANDEX_VISION = ("qwen3.6", "qwen3-vl", "gemma")          # модели AI Studio, которые принимают изображения

SYSTEM = (
    "Ты — старший инженер строительного контроля и эксперт по компьютерному зрению. Сервис ОКО сверяет снимки "
    "камер стройплощадки с календарным графиком: детектор RF-DETR находит технику, модель готовности "
    "(DINOv2 + обученная голова) оценивает стадию и готовность объекта, локальная VLM описывает сцену и отмечает "
    "технику рамками, движок правил выводит отклонения по методике «этап → техника». Твоя задача — сопоставить "
    "выводы всех слоёв, найти и исправить ошибки моделей, дать итоговую оценку, прогноз и рекомендации. "
    "Правила: опирайся на данные и на то, что действительно видно; не выдумывай технику; если слои расходятся "
    "и данных для решения нет — ставь verdict «uncertain» или «doubtful» и низкую уверенность. Отвечай по-русски, "
    "кратко и по делу."
)


def snapshot_schema(class_keys: list[str]) -> dict:
    stage = {"type": "string", "enum": ["S1", "S2", "S3", "S4", "S5", "unknown"]}
    return {
        "type": "object",
        "properties": {
            "summary": {"type": "string", "description": "1–2 предложения: что происходит и главный вывод"},
            "scene": {"type": "object", "properties": {
                "conditions": {"type": "string"},
                "activity": {"type": "string", "enum": ["работы ведутся", "работы не ведутся", "не определить"]},
                "people": {"type": "integer"}}},
            "detections": {"type": "array", "description": "вердикт по КАЖДОЙ рамке детектора (по id)",
                           "items": {"type": "object", "properties": {
                               "id": {"type": "integer"},
                               "verdict": {"type": "string",
                                           "enum": ["confirmed", "false_positive", "wrong_class", "uncertain"]},
                               "correct_class": {"type": "string", "enum": class_keys},
                               "comment": {"type": "string", "description": "до 12 слов; для confirmed не нужен"}},
                               "required": ["id", "verdict"]}},
            "missed": {"type": "array", "description": "техника, которую детектор пропустил",
                       "items": {"type": "object", "properties": {
                           "cls": {"type": "string", "enum": class_keys},
                           "box": {"type": "array", "items": {"type": "number"}},
                           "confidence": {"type": "number"},
                           "comment": {"type": "string", "description": "до 12 слов"}},
                           "required": ["cls", "box", "confidence"]}},
            "stage": {"type": "object", "properties": {
                "value": stage, "readiness": {"type": "number", "description": "готовность объекта, %"},
                "confidence": {"type": "number"}, "comment": {"type": "string", "description": "до 15 слов"}},
                "required": ["value"]},
            "deviations": {"type": "array", "description": "вердикт по каждому отклонению правил (по id)",
                           "items": {"type": "object", "properties": {
                               "id": {"type": "integer"},
                               "verdict": {"type": "string", "enum": ["confirmed", "doubtful", "rejected"]},
                               "comment": {"type": "string", "description": "до 12 слов"}},
                               "required": ["id", "verdict"]}},
            "new_findings": {"type": "array", "description": "отклонения, которые правила не нашли; не больше 3",
                             "items": {"type": "object", "properties": {
                                 "title": {"type": "string"},
                                 "severity": {"type": "string", "enum": ["critical", "warning", "info"]},
                                 "zone": {"type": "string"},
                                 "reason": {"type": "string", "description": "до 15 слов"}},
                                 "required": ["title", "severity", "reason"]}},
            "forecast": {"type": "string", "description": "1–2 предложения: успевают ли этапы зоны, чем грозит"},
            "recommendations": {"type": "array", "description": "1–3 действия, до 15 слов каждое",
                                "items": {"type": "string"}},
            "confidence": {"type": "number", "description": "уверенность в анализе 0..1"},
        },
        "required": ["summary", "detections", "missed", "stage", "deviations", "recommendations", "confidence"],
    }


DAY_SCHEMA = {
    "type": "object",
    "properties": {
        "summary": {"type": "string", "description": "3–5 предложений: состояние площадки за день"},
        "status": {"type": "string", "enum": ["по графику", "есть риски", "отставание"]},
        "zones": {"type": "array", "items": {"type": "object", "properties": {
            "zone": {"type": "string"}, "status": {"type": "string"}, "comment": {"type": "string"}},
            "required": ["zone", "status"]}},
        "risks": {"type": "array", "items": {"type": "object", "properties": {
            "title": {"type": "string"}, "probability": {"type": "number"},
            "impact": {"type": "string", "enum": ["высокое", "среднее", "низкое"]},
            "zone": {"type": "string"}, "task": {"type": "string"},
            "reason": {"type": "string"}, "mitigation": {"type": "string"}},
            "required": ["title", "probability", "impact", "reason"]}},
        "forecast": {"type": "array", "items": {"type": "object", "properties": {
            "task": {"type": "string"}, "expected_delay_days": {"type": "number"}, "reason": {"type": "string"}},
            "required": ["task", "expected_delay_days"]}},
        "model_errors": {"type": "array", "description": "где модели и правила, вероятно, ошиблись",
                         "items": {"type": "string"}},
        "recommendations": {"type": "array", "items": {"type": "string"}},
        "confidence": {"type": "number"},
    },
    "required": ["summary", "status", "risks", "forecast", "recommendations", "confidence"],
}


@dataclass
class LLMResult:
    provider: str
    model: str
    output: dict
    tokens_in: int = 0
    tokens_out: int = 0
    seconds: float = 0.0
    raw: dict = field(default_factory=dict)


class LLMError(RuntimeError):
    pass


class LLMClient:
    def __init__(self, provider: str, model: str = "", api_key: str = "", base_url: str = "",
                 timeout: float = 180.0, transport: httpx.BaseTransport | None = None, folder_id: str = "",
                 images: str = "auto", max_tokens: int = 8000, reasoning: str = "off"):
        self.provider = (provider or "off").lower()
        self.model = model or DEFAULT_MODELS.get(self.provider, "")
        self.api_key = api_key
        self.base_url = (base_url or DEFAULT_BASE.get(self.provider, "")).rstrip("/")
        self.timeout = timeout
        self.transport = transport
        self.folder_id = folder_id
        self.images_mode = (images or "auto").lower()
        self.max_tokens = max_tokens
        self.reasoning_mode = (reasoning or "off").lower()
        self._iam: tuple[str, float] | None = None     # IAM-токен ВМ и время его истечения
        self._effort_ok: bool | None = None            # принимает ли API reasoning_effort (None — ещё не ясно)
        self._fmt: tuple | None = None                 # (формат ответа,) — который модель уже приняла

    @property
    def enabled(self) -> bool:
        return self.provider in DEFAULT_MODELS

    @property
    def name(self) -> str:
        if self.provider == "yandex":             # в AI Studio не только YandexGPT: Alice AI, Qwen, gpt-oss
            m = self.model.lower()
            return "YandexGPT" if "yandexgpt" in m else "Alice AI" if "aliceai" in m else "Yandex AI Studio"
        return PROVIDER_NAMES.get(self.provider, self.provider)

    @property
    def reasoning(self) -> bool:
        """Модель по умолчанию рассуждает (Qwen3, gpt-oss): рассуждения съедают лимит токенов ответа."""
        m = self.model.lower()
        return any(k in m for k in ("qwen3", "gpt-oss", "deepseek"))

    @property
    def effort(self) -> str | None:
        """reasoning_effort для запроса: у рассуждающих моделей по умолчанию рассуждения выключены (none; у gpt-oss
        выключить нельзя — low). None — параметр не отправляется."""
        mode = self.reasoning_mode
        if not self.reasoning or mode in ("model", "auto", "default", "") or self._effort_ok is False:
            return None
        if mode in ("off", "none", "0", "false", "no"):
            return "low" if "gpt-oss" in self.model.lower() else "none"
        return mode

    @property
    def vision(self) -> bool:
        """Принимает ли модель изображения (иначе о кадре судит по описанию VLM и рамкам детектора)."""
        if self.images_mode in ("1", "true", "yes", "on"):
            return True
        if self.images_mode in ("0", "false", "no", "off"):
            return False
        if self.provider == "yandex":
            return any(k in self.model.lower() for k in YANDEX_VISION)
        return self.enabled

    @property
    def model_uri(self) -> str:
        if self.provider != "yandex" or self.model.startswith("gpt://"):
            return self.model
        return f"gpt://{self.folder_id}/{self.model}"

    def info(self) -> dict:
        if not self.enabled:
            return {"provider": "off", "name": "", "model": "", "base_url": "", "key": False, "vision": False}
        out = {"provider": self.provider, "name": self.name, "model": self.model, "base_url": self.base_url,
               "key": bool(self.api_key), "vision": self.vision}
        if self.provider == "yandex":
            out["folder_id"] = self.folder_id
            out["auth"] = "API-ключ" if self.api_key else "IAM-токен сервисного аккаунта ВМ"
        return out

    def ask(self, instructions: str, context: dict, schema: dict, tool_name: str,
            images: list[bytes] | None = None) -> LLMResult:
        if not self.enabled:
            raise LLMError("LLM не настроена: задайте OKO_LLM_PROVIDER=yandex|anthropic|openai")
        if self.provider == "yandex" and not self.folder_id and not self.model.startswith("gpt://"):
            raise LLMError("YandexGPT: задайте каталог OKO_YC_FOLDER_ID (yc config get folder-id)")
        if not self.api_key and self.provider != "yandex":
            raise LLMError("нет ключа API: задайте OKO_LLM_API_KEY")
        images = (images or []) if self.vision else []
        text = instructions + "\n\nДанные моделей и правил (JSON):\n" + json.dumps(context, ensure_ascii=False)
        t0 = time.perf_counter()
        with httpx.Client(timeout=self.timeout, transport=self.transport) as http:
            if self.provider == "anthropic":
                res = self._anthropic(http, text, schema, tool_name, images)
            elif self.provider == "yandex":
                res = self._chat(http, text, schema, tool_name, images, self._yandex_headers(http),
                                 formats=("json_schema", "json_object", None), token_param="max_tokens")
            else:
                param = "max_completion_tokens" if "api.openai.com" in self.base_url else "max_tokens"
                res = self._chat(http, text, schema, tool_name, images,
                                 {"Authorization": f"Bearer {self.api_key}"}, formats=("json_object",),
                                 token_param=param)
        res.seconds = time.perf_counter() - t0
        return res

    # ---------------- Claude: Messages API, ответ — вызов инструмента со схемой
    def _anthropic(self, http: httpx.Client, text: str, schema: dict, tool_name: str, images: list[bytes]) -> LLMResult:
        content = [{"type": "image", "source": {"type": "base64", "media_type": "image/jpeg",
                                                "data": base64.b64encode(im).decode()}} for im in images]
        content.append({"type": "text", "text": text})
        body = {"model": self.model, "max_tokens": 4096, "system": SYSTEM,
                "messages": [{"role": "user", "content": content}],
                "tools": [{"name": tool_name, "description": "Итог проверки в строгом формате", "input_schema": schema}],
                "tool_choice": {"type": "tool", "name": tool_name}}
        r = http.post(f"{self.base_url}/v1/messages", json=body,
                      headers={"x-api-key": self.api_key, "anthropic-version": "2023-06-01",
                               "content-type": "application/json"})
        if r.status_code >= 400:
            raise LLMError(f"Claude API {r.status_code}: {r.text[:500]}")
        data = r.json()
        out = next((b.get("input") for b in data.get("content", []) if b.get("type") == "tool_use"), None)
        if out is None:   # модель ответила текстом — пробуем достать JSON
            out = parse_json("".join(b.get("text", "") for b in data.get("content", []) if b.get("type") == "text"))
        u = data.get("usage", {})
        return LLMResult("anthropic", data.get("model", self.model), out, u.get("input_tokens", 0),
                         u.get("output_tokens", 0), raw={"id": data.get("id"), "stop_reason": data.get("stop_reason")})

    # ---------------- YandexGPT: авторизация
    def _yandex_headers(self, http: httpx.Client) -> dict:
        h = {"OpenAI-Project": self.folder_id, "x-folder-id": self.folder_id} if self.folder_id else {}
        if self.api_key:
            # IAM-токен (t1.…) передаётся как Bearer, API-ключ — как Api-Key
            h["Authorization"] = f"Bearer {self.api_key}" if self.api_key.startswith("t1.") else f"Api-Key {self.api_key}"
            return h
        if self._iam is None or self._iam[1] < time.time() + 60:
            try:
                r = http.get(YC_METADATA_TOKEN, headers={"Metadata-Flavor": "Google"}, timeout=5)
                r.raise_for_status()
                d = r.json()
                self._iam = (d["access_token"], time.time() + float(d.get("expires_in", 3600)))
            except (httpx.HTTPError, KeyError, ValueError) as e:
                raise LLMError("YandexGPT: нет OKO_LLM_API_KEY, а IAM-токен ВМ получить не удалось — привяжите к ВМ "
                               f"сервисный аккаунт с ролью ai.languageModels.user или задайте API-ключ ({e})")
        h["Authorization"] = f"Bearer {self._iam[0]}"
        return h

    # ---------------- OpenAI-совместимый Chat Completions (YandexGPT, ChatGPT, шлюзы)
    def _chat(self, http: httpx.Client, text: str, schema: dict, tool_name: str, images: list[bytes],
              headers: dict, formats: tuple, token_param: str) -> LLMResult:
        prompt = text + "\n\nОтветь ОДНИМ JSON-объектом строго по схеме:\n" + json.dumps(schema, ensure_ascii=False)
        if images:
            content = [{"type": "text", "text": prompt}] + [
                {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + base64.b64encode(im).decode()}}
                for im in images]
        else:
            content = prompt                      # текстовым моделям — обычная строка
        if self.reasoning and self.reasoning_mode in ("off", "none") and "qwen3" in self.model.lower():
            # подсказку /no_think понимает Qwen3; Qwen3.5+ её игнорирует — им рассуждения выключает reasoning_effort
            if isinstance(content, str):
                content += "\n\n/no_think"
            else:
                content[0]["text"] += "\n\n/no_think"
        base = {"model": self.model_uri, "temperature": 0.2,
                "messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": content}]}
        if self.provider == "openai" and "api.openai.com" in self.base_url:
            base.pop("temperature")               # новые модели OpenAI принимают только значение по умолчанию
        if self._fmt and self._fmt[0] in formats:  # формат, который модель уже принимала, — без лишних отказов
            formats = formats[formats.index(self._fmt[0]):]
        budget, tin, tout, rtok = self.max_tokens, 0, 0, 0
        for attempt in range(2):
            r, fmt, body = self._post_first_accepted(http, headers, base, formats, schema, tool_name,
                                                     token_param, budget)
            if r.status_code >= 400:
                raise LLMError(f"{self.name} API {r.status_code}: {r.text[:500]}")
            formats = (fmt,)                      # формат, который модель приняла, — и для повтора
            self._fmt = (fmt,)
            data = r.json()
            choice = (data.get("choices") or [{}])[0]
            msg = choice.get("message") or {}
            text_out = msg.get("content") or ""
            u = data.get("usage") or {}
            tin += int(u.get("prompt_tokens") or 0)
            tout += int(u.get("completion_tokens") or 0)
            rtok += int((u.get("completion_tokens_details") or {}).get("reasoning_tokens") or 0)
            out = parse_json(text_out)
            finish = choice.get("finish_reason")
            # ответ обрезан на лимите (модель долго рассуждала) — один повтор с вдвое большим лимитом
            if "raw" not in out or finish != "length" or attempt:
                break
            budget = min(budget * 2, 32000)
        return LLMResult(self.provider, self.model, out, tin, tout,
                         raw={"id": data.get("id"), "format": fmt, "finish_reason": finish, "max_tokens": budget,
                              "reasoning": bool(msg.get("reasoning_content")) or "<think>" in text_out or rtok > 0,
                              "reasoning_tokens": rtok, "reasoning_effort": body.get("reasoning_effort")})

    def _post_first_accepted(self, http: httpx.Client, headers: dict, base: dict, formats: tuple, schema: dict,
                             tool_name: str, token_param: str, budget: int) -> tuple:
        """Запрос в первом формате ответа, который модель принимает (json_schema → json_object → текст), с
        reasoning_effort. Не принят параметр рассуждений (400/422 с его упоминанием или во всех форматах) —
        те же форматы без него, и дальше клиент его не отправляет. Возвращает (ответ, формат, тело запроса)."""
        efforts = [self.effort, None] if self.effort else [None]
        r = fmt = body = None
        for effort in efforts:
            for i, fmt in enumerate(formats):
                body = {**base, token_param: budget}
                if fmt == "json_schema":
                    body["response_format"] = {"type": "json_schema", "json_schema": {"name": tool_name, "schema": schema}}
                elif fmt == "json_object":
                    body["response_format"] = {"type": "json_object"}
                if effort:
                    body["reasoning_effort"] = effort
                r = http.post(f"{self.base_url}/chat/completions", json=body, headers=headers)
                if r.status_code in (400, 422):
                    if effort and "reasoning" in r.text.lower():
                        break                     # не принят параметр рассуждений — те же форматы без него
                    if i < len(formats) - 1:
                        continue                  # не принят формат ответа — формат попроще
                break
            if r.status_code < 400:
                if effort:
                    self._effort_ok = True
                elif efforts[0]:
                    self._effort_ok = False       # без параметра рассуждений принято — больше его не отправляем
                break
            if r.status_code not in (400, 422):
                break                             # авторизация, лимиты, сбой сервиса — повтор без параметра не поможет
        return r, fmt, body


_override: LLMClient | None = None
_client: LLMClient | None = None


def get_llm() -> LLMClient:
    """Клиент по настройкам; один на процесс — чтобы кешировать IAM-токен ВМ."""
    global _client
    if _override is not None:
        return _override
    if _client is None:
        from ..config import settings
        _client = LLMClient(settings.llm_provider, settings.llm_model, settings.llm_api_key, settings.llm_base_url,
                            settings.llm_timeout, folder_id=settings.yc_folder_id, images=settings.llm_images,
                            max_tokens=settings.llm_max_tokens, reasoning=settings.llm_reasoning)
    return _client


def set_llm(client: LLMClient | None) -> None:
    """Подмена клиента LLM (тесты: LLMClient с httpx.MockTransport)."""
    global _override
    _override = client
