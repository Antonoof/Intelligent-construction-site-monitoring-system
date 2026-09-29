"""ИИ-анализ: модель готовности (без torch — чистые функции), выбор VLM под память, клиенты YandexGPT / Claude /
ChatGPT, сверка слоёв, применение и отмена исправлений.

Внешние вызовы подменяются: LLM — httpx.MockTransport, VLM и модель готовности — заглушки с тем же интерфейсом.
"""
import json
import os
import tempfile
import unittest
from types import SimpleNamespace
from unittest import mock

os.environ.setdefault("OKO_DATA_DIR", tempfile.mkdtemp(prefix="oko-ai-"))
os.environ.setdefault("OKO_DETECTOR", "demo")
os.environ.setdefault("OKO_SEED_DEMO", "1")

import httpx  # noqa: E402
import numpy as np  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from app.ai import jobs  # noqa: E402
from app.ai import llm as L  # noqa: E402
from app.ai import readiness as R  # noqa: E402
from app.ai import vlm as V  # noqa: E402
from app.ai.fusion import cross_check, fuse_snapshot  # noqa: E402
from app.config import REPO_DIR  # noqa: E402
from app.main import app  # noqa: E402
from app.methodology import get_methodology  # noqa: E402

DAY = "2026-09-24"
FOLDER = "b1gtestfolder"


# ------------------------------------------------------------------ модель готовности

def reference_det_features(det, n_classes, width, height):
    """Дословно training/common.py::det_features — эталон для сверки."""
    f = np.zeros((n_classes, 4), np.float32)
    for (x1, y1, x2, y2), c, s in zip(det.xyxy, det.class_id, det.confidence):
        c = int(c)
        f[c, 0] += 1
        f[c, 1] = max(f[c, 1], float(s))
        f[c, 2] += (x2 - x1) * (y2 - y1) / (width * height)
        f[c, 3] += (y1 + y2) / 2 / height
    n = f[:, 0].copy()
    f[:, 3] = np.where(n > 0, f[:, 3] / np.maximum(n, 1), 0)
    f[:, 0] = np.log1p(n)
    return f.ravel()


class ReadinessTest(unittest.TestCase):
    CKPT_CLASSES = ['Dump truck', 'Excavator', 'Motor grader', 'Tower crane', 'Bulldozer', 'Bucket loader', 'Mixer',
                    'Crane manipulator', 'Autocran', 'Drilling rig']

    def test_class_order_maps_to_keys(self):
        m = get_methodology()
        keys = [m.normalize_class(n) for n in self.CKPT_CLASSES]
        self.assertNotIn(None, keys)
        self.assertEqual(keys[:2], ["dump_truck", "excavator"])

    def test_det_features_match_training(self):
        m = get_methodology()
        keys = [m.normalize_class(n) for n in self.CKPT_CLASSES]
        boxes = [(100, 200, 400, 500), (500, 300, 700, 650), (50, 60, 150, 160)]
        cls_idx, conf = [1, 0, 1], [0.91, 0.55, 0.42]
        det = SimpleNamespace(xyxy=np.array(boxes, np.float32), class_id=np.array(cls_idx), confidence=np.array(conf))
        ref = reference_det_features(det, 10, 1280, 720)
        ours = R.det_features([(keys[c], s, b) for b, c, s in zip(boxes, cls_idx, conf)] + [("roller", 0.9, boxes[0])],
                              keys, 1280, 720)
        np.testing.assert_allclose(ours, ref, rtol=1e-6)      # каток голова не знает — он просто не учитывается
        self.assertEqual(ours.shape, (40,))

    def test_plan_status_thresholds(self):
        self.assertEqual(R.plan_status(0.05)[0], "ahead")
        self.assertEqual(R.plan_status(0.0)[0], "on_track")
        self.assertEqual(R.plan_status(-0.05)[0], "risk")
        self.assertEqual(R.plan_status(-0.2), ("late", "отстаём"))

    def test_run_found_in_weights(self):
        run = R.find_run("auto")
        self.assertIsNotNone(run)
        self.assertTrue((run / "head.pt").exists())
        self.assertEqual(R.find_run(str(REPO_DIR / "weights" / "readiness")), run)
        self.assertIsNone(R.find_run("off"))


class CacheTest(unittest.TestCase):
    def test_cache_roundtrip(self):
        from app.ai import cache
        k = cache.key("t", "sha", [1, 2])
        self.assertEqual(k, cache.key("t", "sha", [1, 2]))
        self.assertNotEqual(k, cache.key("t", "sha", [1, 3]))
        calls = []
        v1, hit1 = cache.cached("test", k, lambda: calls.append(1) or {"a": 1})
        v2, hit2 = cache.cached("test", k, lambda: calls.append(1) or {"a": 2})
        self.assertEqual((v1, hit1, v2, hit2, len(calls)), ({"a": 1}, False, {"a": 1}, True, 1))

# ------------------------------------------------------------------ VLM

class VLMTest(unittest.TestCase):
    def test_choose_by_memory(self):
        self.assertEqual(V.choose_from(24)[0], "Qwen/Qwen3-VL-8B-Instruct")
        self.assertEqual(V.choose_from(12)[0], "Qwen/Qwen3-VL-4B-Instruct")
        self.assertEqual(V.choose_from(6.5)[0], "Qwen/Qwen3-VL-2B-Instruct")
        self.assertEqual(V.choose_from(3)[0], "HuggingFaceTB/SmolVLM2-500M-Video-Instruct")
        name, why = V.choose_from(1.0)
        self.assertIsNone(name)
        self.assertIn("не помещается", why)

    def test_readiness_dtype_by_cpu(self):
        """auto: bfloat16 — только где он аппаратный (как при обучении на GPU), иначе float32 — быстрее на CPU."""
        with mock.patch.object(R, "cpu_has_bf16", return_value=False):
            self.assertEqual(R.auto_dtype("cpu"), "float32")             # Intel Ice Lake (Yandex Cloud standard-v3)
        with mock.patch.object(R, "cpu_has_bf16", return_value=True):
            self.assertEqual(R.auto_dtype("cpu"), "bfloat16")            # Sapphire Rapids (AMX), Zen 4
        with mock.patch("builtins.open", mock.mock_open(read_data="processor : 0\nflags : fpu avx512f amx_bf16\n")):
            self.assertTrue(R.cpu_has_bf16())
        with mock.patch("builtins.open", mock.mock_open(read_data="flags : fpu avx2 avx512f avx512_vnni\n")):
            self.assertFalse(R.cpu_has_bf16())

    def test_budget_16gb_vm(self):
        """ВМ 16 ГБ (MemTotal ≈15.6): после RF-DETR и модели готовности (в bfloat16 и во float32) остаётся место
        для Qwen3-VL-2B — не SmolVLM2, как было на стенде с Grounding DINO."""
        from unittest import mock
        det = SimpleNamespace(name="rfdetr-large")
        picked = {}
        for dtype in ("bfloat16", "float32"):
            rd = SimpleNamespace(footprint_gb=lambda d=dtype: R.ReadinessModel.footprint_gb(
                SimpleNamespace(enabled=True, dtype_name=d)))
            with mock.patch.object(V, "_ram_total_gb", return_value=15.6), \
                    mock.patch("app.detection.get_detector", return_value=det), \
                    mock.patch.object(R, "get_readiness", return_value=rd):
                budget, parts = V.ram_budget_gb(1.5)
            picked[dtype] = V.choose_from(budget)[0]
            self.assertEqual(parts.split(",")[0], "RF-DETR 1.0")
        self.assertEqual(picked, {"bfloat16": "Qwen/Qwen3-VL-2B-Instruct", "float32": "Qwen/Qwen3-VL-2B-Instruct"})
        from app.config import Settings
        env = {k: v for k, v in os.environ.items() if k != "OKO_READINESS_DTYPE"}
        with mock.patch.dict(os.environ, env, clear=True):
            self.assertEqual(Settings().readiness_dtype, "auto")        # под процессор: bf16 или float32

    def test_parse_json_and_boxes(self):
        self.assertEqual(V.parse_json('Ответ:\n```json\n{"a": 1}\n```'), {"a": 1})
        self.assertIn("raw", V.parse_json("не JSON"))
        out = V.normalize_boxes({"equipment": [{"type": "экскаватор", "box": [100, 500, 400, 900]},
                                               {"type": "самосвал", "bbox_2d": [0.5, 0.5, 0.7, 0.8]},
                                               {"type": "каток", "box": [0, 0, 1000, 1000]},
                                               {"type": "кран", "box": "слева"}]}, (768, 432))
        eq = out["equipment"]
        self.assertEqual(eq[0]["box"], [0.1, 0.5, 0.4, 0.9])
        self.assertEqual(eq[1]["box"], [0.5, 0.5, 0.7, 0.8])
        self.assertNotIn("box", eq[2])          # «весь кадр» из примера в запросе — не рамка
        self.assertNotIn("box", eq[3])


# ------------------------------------------------------------------ клиенты LLM

def chat_response(out: dict, model: str = "yandexgpt-5.1") -> httpx.Response:
    return httpx.Response(200, json={"id": "c1", "model": model, "usage": {"prompt_tokens": 900, "completion_tokens": 120},
                                     "choices": [{"message": {"role": "assistant",
                                                              "content": json.dumps(out, ensure_ascii=False)}}]})


class LLMClientTest(unittest.TestCase):
    def test_yandexgpt_structured_output(self):
        seen = []

        def handler(req: httpx.Request) -> httpx.Response:
            seen.append((str(req.url), dict(req.headers), json.loads(req.content)))
            return chat_response({"summary": "ок"})

        c = L.LLMClient("yandex", api_key="AQVNkey", folder_id=FOLDER, transport=httpx.MockTransport(handler))
        self.assertFalse(c.vision)
        res = c.ask("проверь", {"x": 1}, {"type": "object"}, "snapshot_review", [b"\xff\xd8jpeg"])
        url, headers, body = seen[0]
        self.assertEqual(url, "https://ai.api.cloud.yandex.net/v1/chat/completions")
        self.assertEqual(headers["authorization"], "Api-Key AQVNkey")
        self.assertEqual(headers["openai-project"], FOLDER)
        self.assertEqual(body["model"], f"gpt://{FOLDER}/yandexgpt-5.1")
        self.assertEqual(body["response_format"]["type"], "json_schema")
        self.assertIsInstance(body["messages"][1]["content"], str)      # YandexGPT — только текст, без изображений
        self.assertEqual((res.output, res.tokens_in, res.tokens_out), ({"summary": "ок"}, 900, 120))

    def test_yandex_schema_fallback_and_vm_iam_token(self):
        seen = []

        def handler(req: httpx.Request) -> httpx.Response:
            if "169.254.169.254" in str(req.url):
                self.assertEqual(req.headers["metadata-flavor"], "Google")
                return httpx.Response(200, json={"access_token": "t1.iam", "expires_in": 3600})
            body = json.loads(req.content)
            seen.append((req.headers["authorization"], body.get("response_format", {}).get("type")))
            if body.get("response_format", {}).get("type") == "json_schema":
                return httpx.Response(400, json={"error": "schema not supported"})
            return chat_response({"summary": "ок"})

        c = L.LLMClient("yandex", model="qwen3.6-35b-a3b", folder_id=FOLDER, transport=httpx.MockTransport(handler))
        self.assertTrue(c.vision)                                       # Qwen3.6 в AI Studio видит изображения
        res = c.ask("проверь", {}, {"type": "object"}, "day_review", [b"img"])
        self.assertEqual(res.output["summary"], "ок")
        self.assertEqual(seen, [("Bearer t1.iam", "json_schema"), ("Bearer t1.iam", "json_object")])
        c.ask("проверь", {}, {"type": "object"}, "day_review", [b"img"])          # принятый формат запоминается
        self.assertEqual(seen[2:], [("Bearer t1.iam", "json_object")])

    def test_reasoning_model_truncated_then_retried(self):
        """Qwen3 в AI Studio рассуждает: ответ обрезан на лимите → один повтор с вдвое большим лимитом."""
        seen = []

        def handler(req: httpx.Request) -> httpx.Response:
            body = json.loads(req.content)
            seen.append(body)
            if len(seen) == 1:
                return httpx.Response(200, json={"model": "gpt://qwen3.6-35b-a3b/latest",
                                                 "usage": {"prompt_tokens": 5147, "completion_tokens": 4096},
                                                 "choices": [{"finish_reason": "length", "message": {
                                                     "content": "<think>Посмотрим на рамки {id: 1…"}}]})
            return httpx.Response(200, json={"usage": {"prompt_tokens": 5147, "completion_tokens": 900},
                                             "choices": [{"finish_reason": "stop", "message": {
                                                 "content": "<think>кратко</think>\n```json\n{\"summary\": \"ок\"}\n```"}}]})

        c = L.LLMClient("yandex", model="qwen3.6-35b-a3b", api_key="k", folder_id=FOLDER, max_tokens=4096,
                        transport=httpx.MockTransport(handler))
        self.assertEqual(c.name, "Yandex AI Studio")
        res = c.ask("проверь", {}, {"type": "object"}, "snapshot_review")
        self.assertEqual(res.output, {"summary": "ок"})
        self.assertEqual([b["max_tokens"] for b in seen], [4096, 8192])
        self.assertTrue(seen[0]["messages"][1]["content"].endswith("/no_think"))
        self.assertEqual(seen[0]["reasoning_effort"], "none")
        self.assertEqual((res.tokens_in, res.tokens_out, res.model), (10294, 4996, "qwen3.6-35b-a3b"))

    def test_reasoning_off_by_default(self):
        """Qwen3.6 не понимает /no_think — рассуждения выключает reasoning_effort; gpt-oss выключить нельзя — low."""
        def body_for(model, **kw):
            seen = []

            def handler(req):
                seen.append(json.loads(req.content))
                return chat_response({"summary": "ок"}, model)
            L.LLMClient("yandex", model=model, api_key="k", folder_id=FOLDER, transport=httpx.MockTransport(handler),
                        **kw).ask("проверь", {}, {"type": "object"}, "snapshot_review")
            return seen[-1]
        self.assertEqual(body_for("qwen3.6-35b-a3b")["reasoning_effort"], "none")
        self.assertEqual(body_for("gpt-oss-120b")["reasoning_effort"], "low")
        self.assertNotIn("reasoning_effort", body_for("yandexgpt-5.1"))              # YandexGPT не рассуждает
        self.assertNotIn("reasoning_effort", body_for("qwen3.6-35b-a3b", reasoning="model"))
        self.assertEqual(body_for("qwen3.6-35b-a3b", reasoning="high")["reasoning_effort"], "high")

    def test_reasoning_param_not_supported(self):
        """API не принимает reasoning_effort — повтор без него, и дальше без него (лишних отказов нет)."""
        seen = []

        def handler(req):
            body = json.loads(req.content)
            seen.append(("reasoning_effort" in body, body.get("response_format", {}).get("type")))
            if "reasoning_effort" in body:
                return httpx.Response(400, json={"error": {"message": "unknown field: reasoning_effort"}})
            return httpx.Response(200, json={"usage": {"prompt_tokens": 10, "completion_tokens": 5,
                                                       "completion_tokens_details": {"reasoning_tokens": 3}},
                                             "choices": [{"message": {"content": '{"summary": "ок"}'}}]})

        c = L.LLMClient("yandex", model="qwen3.6-35b-a3b", api_key="k", folder_id=FOLDER,
                        transport=httpx.MockTransport(handler))
        res = c.ask("проверь", {}, {"type": "object"}, "snapshot_review")
        self.assertEqual(res.output, {"summary": "ок"})
        self.assertEqual((res.raw["reasoning"], res.raw["reasoning_tokens"]), (True, 3))
        self.assertEqual(seen, [(True, "json_schema"), (False, "json_schema")])
        c.ask("проверь", {}, {"type": "object"}, "snapshot_review")
        self.assertEqual(seen[2:], [(False, "json_schema")])

    def test_reasoning_param_rejected_silently(self):
        """Отказ без упоминания параметра: сначала все форматы с ним, затем без него — и схема ответа сохраняется."""
        seen = []

        def handler(req):
            body = json.loads(req.content)
            seen.append(("reasoning_effort" in body, body.get("response_format", {}).get("type")))
            if "reasoning_effort" in body:
                return httpx.Response(400, json={"error": "invalid argument"})
            return chat_response({"summary": "ок"}, "qwen3.6-35b-a3b")

        c = L.LLMClient("yandex", model="qwen3.6-35b-a3b", api_key="k", folder_id=FOLDER,
                        transport=httpx.MockTransport(handler))
        self.assertEqual(c.ask("проверь", {}, {"type": "object"}, "x").output, {"summary": "ок"})
        self.assertEqual(seen, [(True, "json_schema"), (True, "json_object"), (True, None), (False, "json_schema")])
        c.ask("проверь", {}, {"type": "object"}, "x")
        self.assertEqual(seen[4:], [(False, "json_schema")])

    def test_parse_json_variants(self):
        self.assertEqual(V.parse_json('<think>{черновик}</think>{"a": 1}'), {"a": 1})
        self.assertEqual(V.parse_json('Итог: {"a": {"b": 2}} — готово'), {"a": {"b": 2}})
        self.assertEqual(V.parse_json('рассуждение без конца</think>\n{"a": 3}'), {"a": 3})
        self.assertIn("raw", V.parse_json("<think>ещё думаю {"))

    def test_anthropic_forced_tool(self):
        seen = {}

        def handler(req: httpx.Request) -> httpx.Response:
            seen["url"], seen["key"], seen["body"] = str(req.url), req.headers.get("x-api-key"), json.loads(req.content)
            return httpx.Response(200, json={
                "id": "msg_1", "model": "claude-sonnet-5", "stop_reason": "tool_use",
                "content": [{"type": "tool_use", "id": "t1", "name": "snapshot_review", "input": {"summary": "ок"}}],
                "usage": {"input_tokens": 120, "output_tokens": 30}})

        c = L.LLMClient("anthropic", api_key="k", transport=httpx.MockTransport(handler))
        res = c.ask("проверь", {"x": 1}, {"type": "object"}, "snapshot_review", [b"\xff\xd8jpeg"])
        self.assertEqual(res.output, {"summary": "ок"})
        self.assertEqual(seen["url"], "https://api.anthropic.com/v1/messages")
        self.assertEqual(seen["body"]["tool_choice"], {"type": "tool", "name": "snapshot_review"})
        self.assertEqual(seen["body"]["messages"][0]["content"][0]["type"], "image")

    def test_openai_compatible_gateway(self):
        seen = {}

        def handler(req: httpx.Request) -> httpx.Response:
            seen["url"], seen["body"] = str(req.url), json.loads(req.content)
            return chat_response({"summary": "ок"}, "gpt-5")

        c = L.LLMClient("openai", api_key="k", base_url="https://proxy.example/v1/", transport=httpx.MockTransport(handler))
        c.ask("проверь", {}, {"type": "object"}, "day_review")
        self.assertEqual(seen["url"], "https://proxy.example/v1/chat/completions")
        self.assertEqual(seen["body"]["response_format"], {"type": "json_object"})
        self.assertIn("max_tokens", seen["body"])

    def test_not_configured(self):
        with self.assertRaises(L.LLMError):
            L.LLMClient("off").ask("", {}, {}, "x")
        with self.assertRaises(L.LLMError):
            L.LLMClient("yandex").ask("", {}, {}, "x")                  # нет каталога
        with self.assertRaises(L.LLMError):
            L.LLMClient("anthropic").ask("", {}, {}, "x")               # нет ключа


# ------------------------------------------------------------------ сверка слоёв

def _ctx(dets, devs=(), planned=None, assessment=None):
    return {"image": {"width": 1000, "height": 500}, "detector": {"model": "demo"}, "detections": list(dets),
            "deviations": list(devs), "stage": {"planned": planned or {"Z1": ["S1"]}, "readiness_model": assessment}}


class FusionTest(unittest.TestCase):
    D1 = {"id": 1, "cls": "excavator", "name": "Экскаватор", "conf": 0.9, "zone": "Z1", "above_threshold": True,
          "box": [0.1, 0.4, 0.3, 0.8]}
    D2 = {"id": 2, "cls": "truck", "name": "Грузовик", "conf": 0.7, "zone": "Z1", "above_threshold": True,
          "box": [0.5, 0.4, 0.7, 0.8]}
    D3 = {"id": 3, "cls": "roller", "name": "Каток", "conf": 0.6, "zone": "Z1", "above_threshold": True,
          "box": [0.8, 0.5, 0.9, 0.7]}

    def test_cross_check_detector_vs_vlm(self):
        vlm = {"equipment": [{"type": "экскаватор", "box": [0.11, 0.41, 0.3, 0.8]},
                             {"type": "самосвал", "box": [0.52, 0.42, 0.7, 0.8]},
                             {"type": "бульдозер", "box": [0.0, 0.0, 0.05, 0.1]}]}
        cc = cross_check(get_methodology(), [self.D1, self.D2, self.D3], vlm)
        self.assertEqual([a["detection_id"] for a in cc["agree"]], [1])
        self.assertEqual(cc["class_conflict"][0]["detection_id"], 2)
        self.assertEqual(cc["class_conflict"][0]["vlm_cls"], "dump_truck")
        self.assertEqual([v["cls"] for v in cc["vlm_only"]], ["bulldozer"])
        self.assertEqual(cc["detector_only"], [3])
        self.assertIsNone(cross_check(get_methodology(), [self.D1], {"equipment": [{"type": "экскаватор"}]}))

    def test_consensus_and_corrections(self):
        llm = {"summary": "Копают котлован", "confidence": 0.8,
               "detections": [{"id": 1, "verdict": "confirmed"},
                              {"id": 2, "verdict": "wrong_class", "correct_class": "dump_truck"},
                              {"id": "#3", "verdict": "false_positive", "comment": "куча грунта"}],
               "missed": [{"cls": "dump_truck", "box": [600, 200, 800, 400], "confidence": 0.9},
                          {"cls": "bulldozer", "box": [0.1, 0.1, 0.2, 0.2], "confidence": 0.3},
                          {"cls": "unicorn", "box": [0.1, 0.1, 0.2, 0.2], "confidence": 0.9}],
               "stage": {"value": "S1", "readiness": 4},
               "deviations": [{"id": 7, "verdict": "rejected", "comment": "самосвал есть"}],
               "recommendations": ["вызвать самосвалы"]}
        vlm = {"equipment": [{"type": "экскаватор", "count": 1}, {"type": "самосвал", "count": 2}],
               "stage": "S1 подготовка и котлован"}
        dev = {"id": 7, "rule": "INCOMPLETE_SET", "title": "Неполный комплект", "severity": "warning", "zone": "Z1"}
        f = fuse_snapshot(_ctx([self.D1, self.D2, self.D3], [dev],
                               assessment={"stage": "S1", "readiness": 6, "expected": 9, "status_ru": "риск отставания"}),
                          vlm, llm, min_conf=0.6)
        eq = {r["cls"]: r for r in f["equipment"]}
        self.assertEqual((eq["excavator"]["detector"], eq["excavator"]["llm"], eq["excavator"]["agree"]), (1, 1, True))
        self.assertEqual((eq["dump_truck"]["detector"], eq["dump_truck"]["vlm"], eq["dump_truck"]["llm"]), (0, 2, 2))
        self.assertEqual(eq["roller"]["llm"], 0)
        st = f["stage"]
        self.assertEqual((st["final"], st["agree"], st["matches_plan"], st["model_status"]), ("S1", True, True,
                                                                                               "риск отставания"))
        self.assertEqual(f["deviations"][0]["verdict"], "rejected")
        acts = [(c["action"], c["apply"]) for c in f["corrections"]]
        self.assertEqual(acts, [("reclass", True), ("reject", True), ("add", True), ("add", False)])
        added = next(c for c in f["corrections"] if c["action"] == "add")
        self.assertEqual(added["box"], [0.6, 0.4, 0.8, 0.8])       # пиксели → доли кадра

    def test_vision_llm_boxes_per_mille(self):
        llm = {"summary": "ок", "confidence": 0.8, "detections": [], "stage": {"value": "S1"}, "deviations": [],
               "missed": [{"cls": "dump_truck", "box": [600, 400, 800, 800], "confidence": 0.9}], "recommendations": []}
        f = fuse_snapshot(_ctx([]), None, llm, 0.6, meta={"llm_vision": True})
        self.assertEqual(f["missed"][0]["box"], [0.6, 0.4, 0.8, 0.8])            # 0–1000 → доли кадра
        f = fuse_snapshot(_ctx([]), None, llm, 0.6)                              # текстовая LLM: пиксели 1000×500
        self.assertEqual(f["missed"][0]["box"], [0.6, 0.8, 0.8, 1.0])
        llm["missed"][0]["box"] = [0.6, 0.4, 0.8, 0.8]                           # доли — как есть в обоих режимах
        self.assertEqual(fuse_snapshot(_ctx([]), None, llm, 0.6, meta={"llm_vision": True})["missed"][0]["box"],
                         [0.6, 0.4, 0.8, 0.8])

    def test_low_confidence_blocks_corrections(self):
        llm = {"summary": "плохо видно", "confidence": 0.3, "detections": [{"id": 1, "verdict": "false_positive"}],
               "missed": [], "stage": {"value": "unknown"}, "deviations": [], "recommendations": []}
        f = fuse_snapshot(_ctx([self.D1]), None, llm, min_conf=0.6)
        self.assertEqual([c["apply"] for c in f["corrections"]], [False])
        self.assertIsNone(f["stage"]["final"])

    def test_llm_confidence_formats(self):
        """Уверенность LLM приходит по-разному; нет её — None (не 0 %), исправления рамок не применяются."""
        from app.ai.fusion import conf01
        for v, want in ((0.85, 0.85), ("0,85", 0.85), (85, 0.85), ("85%", 0.85), ("высокая", 0.8),
                        ({"value": 0.7}, 0.7), (1, 1.0), (0, 0.0), (None, None), ("не знаю", None), (True, None)):
            self.assertEqual(conf01(v), want, v)
        llm = {"summary": "ok", "detections": [{"id": 1, "verdict": "false_positive"}],
               "missed": [{"cls": "dump_truck", "box": [0.6, 0.4, 0.8, 0.8], "confidence": "90%"}],
               "stage": {"value": "S1"}, "deviations": [], "recommendations": []}
        f = fuse_snapshot(_ctx([self.D1]), None, llm, min_conf=0.6)
        self.assertIsNone(f["confidence"])
        self.assertEqual([(c["action"], c["apply"]) for c in f["corrections"]], [("reject", False), ("add", True)])
        llm["stage"]["confidence"] = 0.8                     # общей нет — берётся уверенность в стадии
        f = fuse_snapshot(_ctx([self.D1]), None, llm, min_conf=0.6)
        self.assertEqual((f["confidence"], f["corrections"][0]["apply"]), (0.8, True))

    def test_loose_llm_json(self):
        """Без строгой схемы модель может вернуть строку вместо объекта или списка — сведение не падает."""
        llm = {"summary": ["Экскаватор", "работает"], "confidence": "0.7", "stage": "S1", "scene": "день",
               "detections": {"id": 1, "verdict": "confirmed"}, "missed": "нет", "deviations": None,
               "recommendations": "вызвать самосвалы"}
        f = fuse_snapshot(_ctx([self.D1]), None, llm, 0.6)
        self.assertEqual((f["stage"]["llm"], f["summary"]), ("S1", "Экскаватор работает"))
        self.assertEqual(f["detections"][0]["verdict"], "confirmed")
        self.assertEqual(f["recommendations"], ["вызвать самосвалы"])
        from app.ai.fusion import clean_day_output
        d = clean_day_output({"summary": "ок", "status": "есть риски", "risks": "срыв вывоза грунта",
                              "forecast": [{"task": "котлован", "expected_delay_days": "2"}], "recommendations": "x"})
        self.assertEqual((d["risks"][0]["title"], d["forecast"][0]["expected_delay_days"], d["recommendations"]),
                         ("срыв вывоза грунта", 2.0, ["x"]))

    def test_without_llm(self):
        f = fuse_snapshot(_ctx([self.D1]), {"scene": "экскаватор копает", "equipment": [{"type": "экскаватор"}]},
                          None, 0.6)
        self.assertEqual(f["summary"], "экскаватор копает")
        self.assertEqual(f["detections"][0]["verdict"], "not_checked")
        self.assertEqual(f["corrections"], [])


# ------------------------------------------------------------------ весь конвейер через API

class FakeVLM:
    enabled = True
    calls = 0

    def cache_key(self, sha, hint):
        from app.ai import cache
        return cache.key("fake-vlm", sha, hint)

    def plan(self):
        return {"enabled": True, "model": "fake-vlm", "device": "cpu", "loaded": True, "reason": "тест"}

    def describe(self, img, hint=""):
        FakeVLM.calls += 1
        return V.VLMResult("fake-vlm", "cpu", {
            "scene": "экскаватор разрабатывает котлован, у бровки самосвал", "stage": "S1",
            "equipment": [{"type": "самосвал", "box": [0.7, 0.55, 0.95, 0.9], "state": "стоит"}],
            "activity": "работы ведутся", "people": 1}, 0.1)


class FakeReadiness:
    enabled = True
    calls = 0

    def plan(self):
        return {"enabled": True, "run": "test", "loaded": True}

    def predict(self, img, dets, class_key):
        FakeReadiness.calls += 1
        return {"stage": "S1", "stage_name": "Подготовка и котлован", "stage_prob": 0.93, "readiness": 7.5,
                "attention": [[0.1] * 16] * 9, "model": "fake-readiness"}


class AIReviewApiTest(unittest.TestCase):
    requests: list = []

    @classmethod
    def handler(cls, req: httpx.Request) -> httpx.Response:
        body = json.loads(req.content)
        cls.requests.append({"headers": dict(req.headers), "body": body})
        tool = body["response_format"]["json_schema"]["name"]
        text = body["messages"][1]["content"]
        ctx, _ = json.JSONDecoder().raw_decode(text.split("(JSON):\n", 1)[1])
        if tool == "snapshot_review":
            vo = ctx["cross_check"]["vlm_only"][0]
            out = {"summary": "Экскаватор в котловане; самосвал у бровки детектор пропустил (есть у VLM).",
                   "confidence": 0.85, "scene": {"activity": "работы ведутся", "people": 1},
                   "detections": [{"id": d["id"], "verdict": "confirmed"} for d in ctx["detections"]],
                   "missed": [{"cls": vo["cls"], "box": vo["box"], "confidence": 0.8, "comment": "по рамке VLM"}],
                   "stage": {"value": "S1", "readiness": 6, "confidence": 0.7},
                   "deviations": [{"id": d["id"], "verdict": "doubtful", "comment": "самосвал на кадре есть"}
                                  for d in ctx["deviations"]],
                   "new_findings": [{"title": "Человек в зоне работы экскаватора", "severity": "warning",
                                     "reason": "VLM видит рабочего у ковша"}],
                   "forecast": "Без вывоза грунта разработка котлована сдвинется на 1–2 дня.",
                   "recommendations": ["Проверить путевые листы самосвалов"]}
        else:
            out = {"summary": "На площадке есть риск по котловану.", "status": "есть риски", "confidence": 0.7,
                   "risks": [{"title": "Срыв вывоза грунта", "probability": 0.4, "impact": "среднее",
                              "reason": "экскаватор без самосвалов"}],
                   "forecast": [{"task": "Разработка котлована", "expected_delay_days": 1}],
                   "model_errors": ["детектор пропустил самосвал на CAM-01"], "recommendations": ["вызвать самосвалы"]}
        return chat_response(out)

    @classmethod
    def setUpClass(cls):
        cls.ctx = TestClient(app)
        cls.c = cls.ctx.__enter__()
        jobs.SYNC = True
        cls.enable()

    @classmethod
    def enable(cls):
        V.set_vlm(FakeVLM())
        R.set_readiness(FakeReadiness())
        L.set_llm(L.LLMClient("yandex", api_key="test", folder_id=FOLDER, transport=httpx.MockTransport(cls.handler)))

    @classmethod
    def tearDownClass(cls):
        jobs.SYNC = False
        V.set_vlm(None)
        R.set_readiness(None)
        L.set_llm(None)
        cls.ctx.__exit__(None, None, None)

    def _tz_deviation(self):
        s = self.c.get("/api/projects/1/summary", params={"day": DAY}).json()
        return next(d for d in s["deviations"] if d["rule"] == "INCOMPLETE_SET")

    def test_snapshot_review_apply_revert_and_day(self):
        st = self.c.get("/api/ai/status").json()
        self.assertTrue(st["enabled"])
        self.assertEqual((st["llm"]["name"], st["llm"]["vision"]), ("YandexGPT", False))

        tz = self._tz_deviation()
        sid = tz["evidence"][0]["snapshot_id"]
        before = self.c.get(f"/api/snapshots/{sid}").json()
        self.assertIsNone(before["assessment"])
        self.assertIsNone(self.c.get(f"/api/snapshots/{sid}/ai-review").json())

        r = self.c.post(f"/api/snapshots/{sid}/ai-review")
        self.assertEqual(r.status_code, 202)
        rv = r.json()
        self.assertEqual(rv["status"], "done", rv.get("error"))
        f = rv["final"]
        self.assertEqual((f["layers"]["vlm_model"], f["layers"]["readiness_model"], f["layers"]["llm_name"]),
                         ("fake-vlm", "fake-readiness", "YandexGPT"))
        # модель готовности: стадия, готовность и план по графику сохранены в снимок
        a = self.c.get(f"/api/snapshots/{sid}").json()["assessment"]
        self.assertEqual((a["stage"], a["readiness"]), ("S1", 7.5))
        self.assertIn("expected", a)
        self.assertIn(a["status"], ("ahead", "on_track", "risk", "late"))
        self.assertEqual((f["stage"]["model"], f["stage"]["vlm"], f["stage"]["llm"], f["stage"]["final"]),
                         ("S1", "S1", "S1", "S1"))
        self.assertNotIn("open_vocab", f)                    # слоя Grounding DINO в сервисе нет
        # сверка детектора и VLM: самосвал видит только VLM, YandexGPT подтверждает его рамкой VLM
        self.assertEqual(f["cross_check"]["vlm_only"][0]["cls"], "dump_truck")
        dump = next(e for e in f["equipment"] if e["cls"] == "dump_truck")
        self.assertEqual((dump["detector"], dump["vlm"], dump["llm"], dump["agree"]), (0, 1, 1, False))
        self.assertTrue(any(d["id"] == tz["id"] and d["verdict"] == "doubtful" for d in f["deviations"]))

        # в YandexGPT ушли все слои текстом, без изображений; карта внимания в LLM не отправляется
        req = self.requests[-1]
        self.assertEqual(req["headers"]["openai-project"], FOLDER)
        self.assertIsInstance(req["body"]["messages"][1]["content"], str)
        full = self.c.get(f"/api/ai/reviews/{rv['id']}", params={"full": 1}).json()
        for key in ("detections", "schedule", "deviations", "stage", "vlm", "cross_check", "classes", "zones"):
            self.assertIn(key, full["context"])
        self.assertNotIn("attention", full["context"]["stage"]["readiness_model"])

        # применить: пропущенный самосвал добавляется рамкой source=llm в зоне камеры, правила пересчитываются
        ap = self.c.post(f"/api/ai/reviews/{rv['id']}/apply").json()
        self.assertEqual(len(ap["applied"]["added"]), 1)
        after = self.c.get(f"/api/snapshots/{sid}").json()
        self.assertEqual(len(after["boxes"]), len(before["boxes"]) + 1)
        llm_box = next(b for b in after["boxes"] if b["source"] == "llm")
        self.assertEqual(llm_box["cls"], "dump_truck")
        self.assertIsNotNone(llm_box["zone"])
        self.assertEqual(self.c.post(f"/api/ai/reviews/{rv['id']}/apply").status_code, 409)

        # анализ дня видит итог анализа снимка и оценку модели готовности
        d = self.c.post("/api/projects/1/ai-summary", params={"day": DAY})
        self.assertEqual(d.status_code, 202)
        dv = d.json()
        self.assertEqual(dv["status"], "done", dv.get("error"))
        self.assertEqual(dv["final"]["status"], "есть риски")
        self.assertGreaterEqual(dv["final"]["layers"]["snapshots_reviewed"], 1)
        self.assertGreaterEqual(dv["final"]["layers"]["readiness_frames"], 1)
        self.assertEqual(self.c.get("/api/projects/1/ai-summary", params={"day": DAY}).json()["id"], dv["id"])

        # отменить: рамка удаляется, отклонение из ТЗ снова с двумя снимками-доказательствами
        rv2 = self.c.post(f"/api/ai/reviews/{rv['id']}/revert").json()
        self.assertTrue(rv2["applied"]["reverted"])
        self.assertEqual(len(self.c.get(f"/api/snapshots/{sid}").json()["boxes"]), len(before["boxes"]))
        self.assertEqual(len(self._tz_deviation()["evidence"]), 2)

    def test_quick_check_ai(self):
        """«Проверить снимок»: снимок без проекта проходит тот же конвейер ИИ, результат — в памяти."""
        from pathlib import Path
        shot = Path(__file__).resolve().parents[2] / "data" / "demo" / "housing" / "snapshots" / "CAM-01_2026-09-24_10-30.jpg"
        r = self.c.post("/api/analyze/ai", data={"work_types": "12.3.1", "planned": ""},
                        files={"file": (shot.name, shot.read_bytes(), "image/jpeg")})
        self.assertEqual(r.status_code, 202)
        j = r.json()
        self.assertEqual(j["status"], "done", j.get("error"))
        f = j["final"]
        self.assertEqual(f["assessment"]["stage"], "S1")
        self.assertEqual(f["stage"]["planned"], ["S1"])
        self.assertEqual(f["cross_check"]["vlm_only"][0]["cls"], "dump_truck")
        self.assertEqual(f["detections"][0]["verdict"], "confirmed")
        self.assertNotIn("context", j)
        full = self.c.get(f"/api/analyze/ai/{j['id']}", params={"full": 1}).json()
        self.assertEqual(full["context"]["schedule"][0]["tasks"][0]["wbs"], "12.3.1")
        self.assertEqual(self.c.get("/api/analyze/ai/q000").status_code, 404)

    def test_vision_llm_skips_vlm_single_request(self):
        """qwen3.6 сама смотрит на кадр: локальная VLM не запускается и не грузится; анализ ставится тем же
        запросом, что и детекция (ai=1); рамки LLM — в тысячных долях кадра."""
        from pathlib import Path
        from app.config import settings
        seen = []

        def handler(req):
            body = json.loads(req.content)
            seen.append(body)
            content = body["messages"][1]["content"]
            ctx, _ = json.JSONDecoder().raw_decode(content[0]["text"].split("(JSON):\n", 1)[1])
            out = {"summary": "Самосвал у бровки детектор пропустил.", "confidence": 0.8, "stage": {"value": "S1"},
                   "detections": [{"id": d["id"], "verdict": "confirmed"} for d in ctx["detections"]],
                   "missed": [{"cls": "dump_truck", "box": [700, 550, 950, 900], "confidence": 0.8}],
                   "deviations": [], "recommendations": [], "_vlm_in_ctx": "vlm" in ctx}
            return chat_response(out, "qwen3.6-35b-a3b")

        L.set_llm(L.LLMClient("yandex", model="qwen3.6-35b-a3b", api_key="test", folder_id=FOLDER,
                              transport=httpx.MockTransport(handler)))
        shot = Path(__file__).resolve().parents[2] / "data" / "demo" / "housing" / "snapshots" / "CAM-01_2026-09-24_10-30.jpg"
        try:
            st = self.c.get("/api/ai/status").json()
            self.assertIn("сама смотрит на кадр", st["vlm"]["skipped"])
            self.assertIsNone(self.c.get("/api/health").json()["ai"]["vlm"])
            n_vlm = FakeVLM.calls
            r = self.c.post("/api/analyze", data={"work_types": "12.3.1", "ai": "1"},
                            files={"file": (shot.name, shot.read_bytes(), "image/jpeg")}).json()
            self.assertEqual(r["check"]["zone"], "FRAME")                        # детекция и проверки — как обычно
            j = r["ai_job"]
            self.assertEqual(j["status"], "done", j.get("error"))
            f = j["final"]
            self.assertEqual(FakeVLM.calls, n_vlm)
            self.assertIn("vlm_skipped", f["layers"])
            self.assertNotIn("vlm", f["layers"].get("timings", {}))
            self.assertEqual(f["missed"][0]["box"], [0.7, 0.55, 0.95, 0.9])
            self.assertEqual(len(seen[0]["messages"][1]["content"]), 3)          # задание + кадр + кадр с рамками
            self.assertEqual(seen[0]["reasoning_effort"], "none")
            self.assertEqual(f["layers"]["llm_reasoning"], {"effort": "none", "used": False, "tokens": 0})
            self.assertIn("тысячных", seen[0]["messages"][1]["content"][0]["text"])
            ctx = self.c.get(f"/api/analyze/ai/{j['id']}", params={"full": 1}).json()["context"]
            self.assertEqual(ctx["vlm"], {"available": False})                     # в LLM — без описания VLM
            import dataclasses                                                    # OKO_VLM_ALWAYS=1
            with mock.patch.object(jobs, "settings", dataclasses.replace(settings, vlm_always=True)):
                self.assertNotIn("skipped", self.c.get("/api/ai/status").json()["vlm"])
                j2 = self.c.post("/api/analyze/ai", data={"work_types": "12.3.1"},
                                 files={"file": (shot.name, shot.read_bytes(), "image/jpeg")}).json()
                self.assertEqual(j2["final"]["layers"]["vlm_model"], "fake-vlm")
        finally:
            self.enable()

    def _vision_llm(self, seen: list):
        """LLM, которая сама видит кадр (qwen3.6): запоминает контекст каждого запроса."""
        def handler(req):
            body = json.loads(req.content)
            ctx, _ = json.JSONDecoder().raw_decode(body["messages"][1]["content"][0]["text"].split("(JSON):\n", 1)[1])
            seen.append(ctx)
            return chat_response({"summary": "ок", "confidence": 0.8, "stage": {"value": "S1"},
                                  "detections": [{"id": d["id"], "verdict": "confirmed"} for d in ctx["detections"]],
                                  "missed": [], "deviations": [], "recommendations": []}, "qwen3.6-35b-a3b")
        L.set_llm(L.LLMClient("yandex", model="qwen3.6-35b-a3b", api_key="test", folder_id=FOLDER,
                              transport=httpx.MockTransport(handler)))

    @staticmethod
    def _clear_readiness_cache():
        import shutil
        from app.config import settings
        shutil.rmtree(settings.data_dir / "ai_cache" / "readiness", ignore_errors=True)

    def test_parallel_readiness_and_llm(self):
        """LLM видит кадр, оценки кадра зоны ещё нет: модель готовности и LLM работают одновременно, оценка модели —
        в сведении; кадр общего плана — по очереди (его оценка меняет отклонения); повтор — оценка из кеша, по очереди."""
        seen = []
        self._vision_llm(seen)
        self._clear_readiness_cache()
        try:
            over = self.c.get("/api/projects/1/snapshots", params={"day": DAY, "camera": "CAM-06"}).json()[0]
            zone = self.c.get("/api/projects/1/snapshots", params={"day": DAY, "camera": "CAM-03"}).json()[0]
            r1 = self.c.post(f"/api/snapshots/{over['id']}/ai-review").json()
            self.assertEqual(r1["status"], "done", r1.get("error"))
            self.assertNotIn("parallel", r1["final"]["layers"])
            self.assertEqual(seen[-1]["stage"]["readiness_model"]["stage"], "S1")    # LLM дождалась оценки
            self.assertNotIn("readiness_overview", seen[-1]["stage"])

            n = FakeReadiness.calls
            r2 = self.c.post(f"/api/snapshots/{zone['id']}/ai-review").json()
            self.assertEqual(r2["status"], "done", r2.get("error"))
            f = r2["final"]
            self.assertEqual(f["layers"]["parallel"], ["readiness", "llm"])
            self.assertEqual(FakeReadiness.calls, n + 1)
            self.assertIsNone(seen[-1]["stage"]["readiness_model"])                  # LLM не ждала оценку кадра
            self.assertEqual(seen[-1]["stage"]["readiness_overview"]["camera"], "CAM-06")   # опора — общий план
            self.assertEqual((f["stage"]["model"], f["stage"]["llm"]), ("S1", "S1"))     # оценка модели — в сведении
            self.assertEqual(set(f["layers"]["timings"]), {"readiness", "llm"})
            self.assertEqual(self.c.get(f"/api/snapshots/{zone['id']}").json()["assessment"]["stage"], "S1")
            full = self.c.get(f"/api/ai/reviews/{r2['id']}", params={"full": 1}).json()
            self.assertIsNone(full["context"]["stage"]["readiness_model"])            # в аудите — то, что видела LLM

            r3 = self.c.post(f"/api/snapshots/{zone['id']}/ai-review").json()         # оценка уже есть — по очереди
            self.assertNotIn("parallel", r3["final"]["layers"])
            self.assertEqual(seen[-1]["stage"]["readiness_model"]["stage"], "S1")
            self.assertEqual(FakeReadiness.calls, n + 1)

            import dataclasses                                                        # OKO_AI_PARALLEL=0
            from app.config import settings
            self._clear_readiness_cache()
            zone2 = self.c.get("/api/projects/1/snapshots", params={"day": DAY, "camera": "CAM-04"}).json()[0]
            with mock.patch.object(jobs, "settings", dataclasses.replace(settings, ai_parallel=False)):
                r4 = self.c.post(f"/api/snapshots/{zone2['id']}/ai-review").json()
            self.assertNotIn("parallel", r4["final"]["layers"])
            self.assertEqual(seen[-1]["stage"]["readiness_model"]["stage"], "S1")
        finally:
            self.enable()

    def test_parallel_quick_check(self):
        """Быстрая проверка с LLM, которая видит кадр: модель готовности и LLM одновременно — и в фоновых потоках."""
        import time as _t
        from pathlib import Path
        shot = Path(__file__).resolve().parents[2] / "data" / "demo" / "housing" / "snapshots" / "CAM-01_2026-09-24_10-30.jpg"
        seen = []
        self._vision_llm(seen)
        self._clear_readiness_cache()
        try:
            r = self.c.post("/api/analyze", data={"work_types": "12.3.1", "ai": "1"},
                            files={"file": (shot.name, shot.read_bytes(), "image/jpeg")}).json()
            j = r["ai_job"]
            self.assertEqual(j["status"], "done", j.get("error"))
            f = j["final"]
            self.assertEqual(f["layers"]["parallel"], ["readiness", "llm"])
            self.assertIsNone(seen[-1]["stage"]["readiness_model"])
            self.assertEqual((f["assessment"]["stage"], f["stage"]["model"]), ("S1", "S1"))

            self._clear_readiness_cache()                                             # то же в рабочих потоках
            jobs.SYNC = False
            try:
                j = self.c.post("/api/analyze/ai", data={"work_types": "12.3.1"},
                                files={"file": (shot.name, shot.read_bytes(), "image/jpeg")}).json()
                for _ in range(200):
                    j = self.c.get(f"/api/analyze/ai/{j['id']}").json()
                    if j["status"] in ("done", "error"):
                        break
                    _t.sleep(0.02)
            finally:
                jobs.SYNC = True
            self.assertEqual(j["status"], "done", j.get("error"))
            self.assertEqual(j["final"]["layers"]["parallel"], ["readiness", "llm"])
            self.assertEqual(j["final"]["assessment"]["stage"], "S1")
        finally:
            self.enable()

    def test_join_finishes_once(self):
        """Сведение вызывается ровно один раз, даже если части приходят одновременно и повторно."""
        import threading
        got = []
        j = jobs._Join(("readiness", "llm"), got.append)
        ts = [threading.Thread(target=j.put, args=(n, i)) for i, n in enumerate(["readiness", "llm"] * 10)]
        for t in ts:
            t.start()
        for t in ts:
            t.join()
        self.assertEqual(len(got), 1)
        self.assertEqual(set(got[0]), {"readiness", "llm"})

    def test_quick_check_detects_once(self):
        """Повторная проверка того же файла и «Повторить анализ» не запускают детектор заново."""
        from pathlib import Path
        from app.api import analyze as A
        from app.detection import get_detector
        shot = Path(__file__).resolve().parents[2] / "data" / "demo" / "housing" / "snapshots" / "CAM-01_2026-09-24_10-30.jpg"
        A._det_cache.clear()
        det = get_detector()
        with mock.patch.object(det, "detect", wraps=det.detect) as spy:
            a = self.c.post("/api/analyze", data={"work_types": "12.3.1"},
                            files={"file": (shot.name, shot.read_bytes(), "image/jpeg")}).json()
            b = self.c.post("/api/analyze/ai", data={"work_types": "12.3.1"},
                            files={"file": (shot.name, shot.read_bytes(), "image/jpeg")}).json()
            c = self.c.post("/api/analyze", data={"work_types": "12.3.1", "tiles": "3"},
                            files={"file": (shot.name, shot.read_bytes(), "image/jpeg")}).json()
        self.assertEqual(spy.call_count, 2)                                       # тот же файл; 3×3 — отдельно
        self.assertEqual(b["status"], "done", b.get("error"))
        self.assertEqual(len(a["boxes"]), len(c["boxes"]))

    def test_repeat_analysis_reuses_model_layers(self):
        """Повторный анализ того же кадра: модели не запускаются, из кеша, заново — только LLM."""
        sid = self._tz_deviation()["evidence"][1]["snapshot_id"]
        first = self.c.post(f"/api/snapshots/{sid}/ai-review").json()
        self.assertEqual(first["status"], "done", first.get("error"))
        calls = (FakeVLM.calls, FakeReadiness.calls)
        n_llm = len(self.requests)
        again = self.c.post(f"/api/snapshots/{sid}/ai-review").json()
        self.assertEqual(again["status"], "done", again.get("error"))
        self.assertEqual((FakeVLM.calls, FakeReadiness.calls), calls)
        self.assertEqual(len(self.requests), n_llm + 1)
        L_ = again["final"]["layers"]
        self.assertTrue({"vlm", "readiness"} <= set(L_["cached"]))
        self.assertIn("llm", L_["timings"])
        self.assertEqual(L_["vlm_reason"], "тест")           # почему выбрана эта VLM — в карточке
        self.assertEqual(again["final"]["equipment"], first["final"]["equipment"])

    def test_background_pipeline(self):
        """Без SYNC: модели — в рабочем потоке, LLM — в своём пуле; интерфейс опрашивает статус до «готово»."""
        import time as _t
        snaps = self.c.get("/api/projects/1/snapshots", params={"day": DAY, "camera": "CAM-02"}).json()
        sid = snaps[0]["id"]
        jobs.SYNC = False
        try:
            r = self.c.post(f"/api/snapshots/{sid}/ai-review").json()
            for _ in range(100):
                r = self.c.get(f"/api/ai/reviews/{r['id']}").json()
                if r["status"] in ("done", "error"):
                    break
                _t.sleep(0.05)
        finally:
            jobs.SYNC = True
        self.assertEqual(r["status"], "done", r.get("error"))
        self.assertIn("llm", r["final"]["layers"]["timings"])

    def test_disabled_layers(self):
        L.set_llm(L.LLMClient("off"))
        V.set_vlm(V.LocalVLM("off"))
        R.set_readiness(R.ReadinessModel("off"))
        try:
            self.assertEqual(self.c.post("/api/snapshots/1/ai-review").status_code, 409)
            self.assertEqual(self.c.post("/api/projects/1/ai-summary", params={"day": DAY}).status_code, 409)
        finally:
            self.enable()


class FrameZoneTest(unittest.TestCase):
    """Свой снимок с другого ракурса: весь кадр — одна зона вместо полигонов камеры."""

    @classmethod
    def setUpClass(cls):
        cls.ctx = TestClient(app)
        cls.c = cls.ctx.__enter__()

    @classmethod
    def tearDownClass(cls):
        cls.ctx.__exit__(None, None, None)

    def test_filename_dates(self):
        from app.imaging import filename_datetime as f
        self.assertEqual(str(f("089_предл_2025-03-01_2025-04-30_Screenshot_76.png")), "2025-03-01 12:00:00")
        self.assertEqual(str(f("027_дата_2024-01-21_Screenshot_90.png")), "2024-01-21 12:00:00")
        self.assertEqual(str(f("CAM-01_2026-09-24_10-30.jpg")), "2026-09-24 10:30:00")
        self.assertEqual(str(f("IMG_20260924_201530.jpg")), "2026-09-24 20:15:30")
        self.assertEqual(str(f("CAM-02_24.09.2026_10.30.jpg")), "2026-09-24 10:30:00")

    def test_whole_frame_zone_and_back(self):
        s = self.c.get("/api/projects/1/summary", params={"day": DAY}).json()
        tz = next(d for d in s["deviations"] if d["rule"] == "INCOMPLETE_SET")
        sid = tz["evidence"][0]["snapshot_id"]
        before = self.c.get(f"/api/snapshots/{sid}").json()
        self.assertIsNone(before["frame_zone"])
        self.assertEqual([c["zone"] for c in before["checks"]], ["Z1"])
        other = next(z["key"] for z in before["project_zones"] if z["key"] != "Z1")
        after = self.c.patch(f"/api/snapshots/{sid}", json={"frame_zone": other}).json()
        self.assertEqual(after["frame_zone"]["key"], other)
        self.assertEqual(after["zones"], [])                        # полигоны камеры не рисуются
        self.assertTrue(all(b["zone"] == other for b in after["boxes"] if not b["rejected"]))
        self.assertEqual([c["zone"] for c in after["checks"]], [other])
        self.assertEqual(self.c.patch(f"/api/snapshots/{sid}", json={"frame_zone": "нет-такой"}).status_code, 404)
        back = self.c.patch(f"/api/snapshots/{sid}", json={"frame_zone": None}).json()
        self.assertIsNone(back["frame_zone"])
        self.assertEqual([b["zone"] for b in back["boxes"]], [b["zone"] for b in before["boxes"]])
        s2 = self.c.get("/api/projects/1/summary", params={"day": DAY}).json()
        self.assertEqual(len(next(d for d in s2["deviations"] if d["rule"] == "INCOMPLETE_SET")["evidence"]), 2)


if __name__ == "__main__":
    unittest.main()
