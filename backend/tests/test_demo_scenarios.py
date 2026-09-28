"""Сквозная проверка на демо-данных: загрузка графика, снимков и ожидаемые отклонения по сценариям.

Каждый сценарий из data/demo/README.md должен давать ровно ожидаемое отклонение.
"""
import datetime as dt
import os
import tempfile
import unittest
from pathlib import Path

TMP = tempfile.mkdtemp(prefix="oko-test-")
os.environ["OKO_DATA_DIR"] = TMP
os.environ["OKO_DETECTOR"] = "demo"
os.environ["OKO_SEED_DEMO"] = "0"

from sqlalchemy import select  # noqa: E402

from app import models as M  # noqa: E402
from app import services as S  # noqa: E402
from app.db import SessionLocal, init_db  # noqa: E402
from app.demo import seed_demo  # noqa: E402
from app.methodology import get_methodology, sync_to_db  # noqa: E402


def devs(s, pid, day):
    out = {}
    for d in s.scalars(select(M.Deviation).where(M.Deviation.project_id == pid, M.Deviation.day == day)):
        z = s.get(M.Zone, d.zone_id).key if d.zone_id else "SITE"
        out.setdefault(z, []).append(d)
    return out


class DemoScenariosTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        init_db()
        cls.s = SessionLocal()
        sync_to_db(cls.s, get_methodology())
        cls.p1, cls.p2 = seed_demo(cls.s)
        cls.s.commit()

    @classmethod
    def tearDownClass(cls):
        cls.s.close()

    def rules(self, pid, day, zone):
        return sorted((d.rule_code, d.equipment_class or "", d.status) for d in devs(self.s, pid, day).get(zone, []))

    def test_schedule_import_and_matching(self):
        tasks = S.schedule_tasks(self.s, self.p1)
        by_name = {t.name: t for t in tasks}
        self.assertEqual(by_name["Устройство котлована"].work_type_id, "12.3.1")
        self.assertEqual(by_name["Вынос сетей: ливневая канализация"].work_type_id, "10.2-6")
        self.assertEqual(by_name["Устройство временных дорог"].match_method, "name")
        self.assertTrue(by_name["Устройство фундамента"].is_summary)
        self.assertIsNone(by_name["Монтаж башенного крана"].work_type_id)      # нет в справочнике
        self.assertEqual(by_name["Устройство фундаментной плиты"].planned_equipment,
                         {"concrete_mixer": 2, "concrete_pump": 1})

    def test_tz_example_pit(self):
        d = devs(self.s, self.p1, dt.date(2026, 9, 24))["Z1"]
        inc = [x for x in d if x.rule_code == "INCOMPLETE_SET"]
        self.assertEqual(len(inc), 1)
        self.assertEqual(inc[0].status, "confirmed")
        self.assertEqual(len(inc[0].evidence), 2)                              # 10:30 и 11:20
        self.assertIn("снижение темпа", inc[0].message)
        self.assertIn(("UNEXPECTED_EQUIPMENT", "roller", "confirmed"), self.rules(self.p1, dt.date(2026, 9, 24), "Z1"))

    def test_slab_mixers(self):
        r = self.rules(self.p1, dt.date(2026, 9, 24), "Z2")
        self.assertIn(("INSUFFICIENT_COUNT", "concrete_mixer", "confirmed"), r)
        self.assertEqual(self.rules(self.p1, dt.date(2026, 9, 25), "Z2"), [])

    def test_road_overrun_and_early_start(self):
        self.assertEqual(self.rules(self.p1, dt.date(2026, 9, 24), "Z3"),
                         [("STAGE_OVERRUN", "grader", "confirmed"), ("STAGE_OVERRUN", "roller", "confirmed")])
        self.assertEqual(self.rules(self.p1, dt.date(2026, 9, 24), "Z6"), [("EARLY_START", "drilling_rig", "confirmed")])

    def test_empty_trench_and_night(self):
        self.assertEqual(self.rules(self.p1, dt.date(2026, 9, 24), "Z4"), [("NO_EQUIPMENT", "", "confirmed")])

    def test_trench_missing_required(self):
        # 25.09: экскаватор без самосвала (к сведению) и самосвал без экскаватора — нет необходимой техники
        self.assertEqual(self.rules(self.p1, dt.date(2026, 9, 25), "Z4"),
                         [("INCOMPLETE_SET", "", "preliminary"), ("MISSING_REQUIRED", "", "preliminary")])

    def test_camp(self):
        self.assertEqual(self.rules(self.p1, dt.date(2026, 9, 24), "Z5"), [("NO_ACTIVE_STAGE", "excavator", "confirmed")])
        self.assertEqual(self.rules(self.p1, dt.date(2026, 9, 25), "Z5"), [("NO_DATA", "", "preliminary")])

    def test_day2_pit_ok(self):
        self.assertEqual(self.rules(self.p1, dt.date(2026, 9, 25), "Z1"), [])

    def test_winter_school(self):
        day = dt.date(2026, 2, 11)
        self.assertEqual(self.rules(self.p2, day, "Z1"), [("STAGE_OVERRUN", "bulldozer", "confirmed")])
        mism = [d for d in self.s.scalars(select(M.Deviation).where(M.Deviation.project_id == self.p2,
                                                                     M.Deviation.rule_code == "STAGE_MISMATCH"))]
        self.assertEqual(len(mism), 1)
        self.assertIn("S2", mism[0].message)

    def test_review_creates_violation_and_rejects(self):
        d = next(x for x in devs(self.s, self.p1, dt.date(2026, 9, 24))["Z4"] if x.rule_code == "NO_EQUIPMENT")
        S.review_deviation(self.s, d, "accepted", "подрядчик не вышел")
        v = self.s.scalar(select(M.Violation).where(M.Violation.deviation_id == d.id))
        self.assertEqual(v.number, "НР-2026-0001")
        self.assertEqual(v.contractor, "ООО «ИнжСети»")
        b = next(x for x in devs(self.s, self.p2, dt.date(2026, 2, 11))["Z1"] if x.rule_code == "STAGE_OVERRUN")
        S.review_deviation(self.s, b, "rejected", "уборка снега, не засыпка")
        # пересчёт не возвращает отклонённое инженером и не удаляет его историю
        S.recompute_day(self.s, self.s.get(M.Project, self.p2), dt.date(2026, 2, 11))
        self.assertEqual(self.s.get(M.Deviation, b.id).review_status, "rejected")
        self.s.rollback()

    def test_snapshot_checks_explain_tz_example(self):
        snap = self.s.scalar(select(M.Snapshot).where(M.Snapshot.project_id == self.p1,
                                                      M.Snapshot.original_name == "CAM-01_2026-09-24_10-30.jpg"))
        ch = self.s.scalar(select(M.SnapshotCheck).where(M.SnapshotCheck.snapshot_id == snap.id,
                                                          M.SnapshotCheck.zone_key == "Z1"))
        task = ch.payload["tasks"][0]
        self.assertEqual(task["work_type_id"], "12.3.1")
        self.assertEqual([c["state"] for c in task["companions"]], ["violated"])
        self.assertEqual(ch.payload["findings"][0]["rule"], "INCOMPLETE_SET")


if __name__ == "__main__":
    unittest.main()
