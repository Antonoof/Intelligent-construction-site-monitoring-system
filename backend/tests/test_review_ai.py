"""ИИ-анализ: модель готовности (без torch — чистые функции), выбор VLM под память, клиенты YandexGPT / Claude /
ChatGPT, сверка слоёв, применение и отмена исправлений.

Внешние вызовы подменяются: LLM — httpx.MockTransport, VLM и модель готовности — заглушки с тем же интерфейсом.
"""
import json
import os
import tempfile
import unittest
from types import SimpleNamespace

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

    def test_low_confidence_blocks_corrections(self):
        llm = {"summary": "плохо видно", "confidence": 0.3, "detections": [{"id": 1, "verdict": "false_positive"}],
               "missed": [], "stage": {"value": "unknown"}, "deviations": [], "recommendations": []}
        f = fuse_snapshot(_ctx([self.D1]), None, llm, min_conf=0.6)
        self.assertEqual([c["apply"] for c in f["corrections"]], [False])
        self.assertIsNone(f["stage"]["final"])

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

    def plan(self):
        return {"enabled": True, "model": "fake-vlm", "device": "cpu", "loaded": True, "reason": "тест"}

    def describe(self, img, hint=""):
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
        self.assertEqual(dv["final"]["layers"]["snapshots_reviewed"], 1)
        self.assertGreaterEqual(dv["final"]["layers"]["readiness_frames"], 1)
        self.assertEqual(self.c.get("/api/projects/1/ai-summary", params={"day": DAY}).json()["id"], dv["id"])

        # отменить: рамка удаляется, отклонение из ТЗ снова с двумя снимками-доказательствами
        rv2 = self.c.post(f"/api/ai/reviews/{rv['id']}/revert").json()
        self.assertTrue(rv2["applied"]["reverted"])
        self.assertEqual(len(self.c.get(f"/api/snapshots/{sid}").json()["boxes"]), len(before["boxes"]))
        self.assertEqual(len(self._tz_deviation()["evidence"]), 2)

    def test_disabled_layers(self):
        L.set_llm(L.LLMClient("off"))
        V.set_vlm(V.LocalVLM("off"))
        R.set_readiness(R.ReadinessModel("off"))
        try:
            self.assertEqual(self.c.post("/api/snapshots/1/ai-review").status_code, 409)
            self.assertEqual(self.c.post("/api/projects/1/ai-summary", params={"day": DAY}).status_code, 409)
        finally:
            self.enable()


if __name__ == "__main__":
    unittest.main()
