"""Проверка REST API: демо-проект, загрузка графика и снимков, отклонения, вердикт инженера, быстрая проверка."""
import os
import tempfile
import unittest
from pathlib import Path

os.environ["OKO_DATA_DIR"] = tempfile.mkdtemp(prefix="oko-api-")
os.environ["OKO_DETECTOR"] = "demo"
os.environ["OKO_SEED_DEMO"] = "1"

from fastapi.testclient import TestClient  # noqa: E402

from app.main import app  # noqa: E402

DEMO = Path(__file__).resolve().parents[2] / "data" / "demo"


class ApiTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ctx = TestClient(app)
        cls.c = cls.ctx.__enter__()          # запуск lifespan: схема, методика, демо-проекты

    @classmethod
    def tearDownClass(cls):
        cls.ctx.__exit__(None, None, None)

    def test_health_and_reference(self):
        h = self.c.get("/api/health").json()
        self.assertEqual(h["status"], "ok")
        self.assertEqual(h["projects"], 2)
        eq = self.c.get("/api/reference/equipment").json()
        self.assertEqual(sum(1 for e in eq if e["in_tz"]), 8)
        wts = self.c.get("/api/reference/work-types", params={"limit": 1000}).json()
        self.assertEqual(len(wts), 377)
        self.assertEqual(self.c.get("/api/reference/methodology.xlsx").status_code, 200)

    def test_summary_and_deviations(self):
        s = self.c.get("/api/projects/1/summary", params={"day": "2026-09-24"}).json()
        self.assertEqual(s["kpi"]["snapshots"], 17)
        rules = sorted(d["rule"] for d in s["deviations"])
        self.assertIn("INCOMPLETE_SET", rules)
        tz = next(d for d in s["deviations"] if d["rule"] == "INCOMPLETE_SET")
        self.assertEqual(tz["zone"]["name"], "Котлован, секция 1")
        self.assertEqual(len(tz["evidence"]), 2)
        img = self.c.get(tz["evidence"][0]["thumb"])
        self.assertEqual(img.headers["content-type"], "image/jpeg")

    def test_new_project_flow(self):
        p = self.c.post("/api/projects", json={"name": "Проверка API", "object_type": "housing"}).json()
        pid = p["id"]
        site = {"zones": [{"key": "Z1", "name": "Котлован, секция 1"}],
                "cameras": [{"key": "CAM-01", "name": "Котлован", "default_zone": "Z1"}]}
        self.assertEqual(self.c.put(f"/api/projects/{pid}/site", json=site).status_code, 200)
        with open(DEMO / "housing" / "schedule_housing.xlsx", "rb") as fh:
            r = self.c.post(f"/api/projects/{pid}/schedule", files={"file": ("график.xlsx", fh.read())}).json()
        self.assertEqual(r["rows"], 17)
        self.assertEqual(r["matched"], 16)
        shots = sorted((DEMO / "housing" / "snapshots").glob("CAM-01_2026-09-24_1*.jpg"))[:2]   # 10:30 и 11:20
        files = [("files", (f.name, f.read_bytes(), "image/jpeg")) for f in shots]
        res = self.c.post(f"/api/projects/{pid}/snapshots", data={"camera": "CAM-01"}, files=files).json()
        self.assertEqual([x["time_source"] for x in res], ["filename", "filename"])
        devs = self.c.get(f"/api/projects/{pid}/deviations").json()
        self.assertEqual([d["rule"] for d in devs], ["INCOMPLETE_SET"])
        self.assertEqual(devs[0]["status"], "confirmed")
        # повторная загрузка — без дублей
        again = self.c.post(f"/api/projects/{pid}/snapshots", data={"camera": "CAM-01"}, files=files).json()
        self.assertTrue(all(x["duplicate"] for x in again))
        # инженер подтверждает — появляется нарушение
        d = self.c.post(f"/api/deviations/{devs[0]['id']}/review", json={"verdict": "accepted"}).json()
        self.assertEqual(d["review_status"], "accepted")
        vio = self.c.get(f"/api/projects/{pid}/violations").json()
        self.assertEqual(len(vio), 1)
        self.assertEqual(vio[0]["contractor"], "ООО «ЗемРесурс»")
        csv = self.c.get(f"/api/projects/{pid}/report", params={"format": "csv"})
        self.assertIn("Неполный комплект техники", csv.content.decode("utf-8-sig"))

    def test_analyze_endpoint(self):
        img = DEMO / "housing" / "snapshots" / "CAM-01_2026-09-24_10-30.jpg"
        r = self.c.post("/api/analyze", data={"work_types": "12.3.1"},
                        files={"file": (img.name, img.read_bytes(), "image/jpeg")}).json()
        self.assertEqual([f["rule"] for f in r["findings"]], ["INCOMPLETE_SET"])
        self.assertEqual(r["boxes"][0]["cls"], "excavator")
        bad = self.c.post("/api/analyze", data={"work_types": "99.9"},
                          files={"file": (img.name, img.read_bytes(), "image/jpeg")})
        self.assertEqual(bad.status_code, 422)

    def test_unknown_photo_without_weights_is_no_data(self):
        # чужой снимок при демо-детекторе — «нет данных», а не ложное «работы не ведутся»
        import io
        from PIL import Image, ImageOps
        img = Image.open(DEMO / "housing" / "snapshots" / "CAM-01_2026-09-24_10-30.jpg")
        buf = io.BytesIO()
        ImageOps.mirror(img).save(buf, "JPEG", quality=90)
        r = self.c.post("/api/analyze", data={"work_types": "12.3.1"},
                        files={"file": ("my_photo.jpg", buf.getvalue(), "image/jpeg")}).json()
        self.assertEqual([f["rule"] for f in r["findings"]], ["NO_DATA"])
        self.assertFalse(r["quality"]["ok"])

    def test_internal_detect_contract(self):
        img = DEMO / "housing" / "snapshots" / "CAM-01_2026-09-24_08-05.jpg"
        r = self.c.post("/internal/detect", files={"file": (img.name, img.read_bytes(), "image/jpeg")}).json()
        self.assertEqual(sorted(b["cls"] for b in r["boxes"]), ["dump_truck", "dump_truck", "excavator"])
        self.assertTrue(r["recognized"])


if __name__ == "__main__":
    unittest.main()
