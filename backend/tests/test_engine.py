"""Проверки движка правил на коротких сценариях — по одному на каждое правило методики.

Запуск из backend/:  python -m unittest discover -s tests -v   (или python -m pytest tests)
"""
import datetime as dt
import unittest

from app.engine import (DetInfo, EngineContext, Obs, SITE_ZONE, TaskInfo, ZoneWindow, evaluate_window,
                        merge_slots, tasks_for_zone)
from app.methodology import get_methodology

M = get_methodology()
ALL = frozenset(M.classes) - {"concrete_pump", "asphalt_paver"}      # модель v2: 12 классов
DAY = dt.date(2026, 9, 24)
_id = iter(range(1, 10_000))


def obs(hhmm: str, *dets, valid=True, reason=""):
    h, mi = map(int, hhmm.split(":"))
    return Obs(snapshot_ids=[next(_id)], taken_at=dt.datetime.combine(DAY, dt.time(h, mi)), valid=valid,
               reason=reason, dets=[DetInfo(next(_id), c, conf) for c, conf in dets])


def task(wt_id, start="2026-09-10", end="2026-10-10", zone="Z1", tid=1, planned=None, summary=False):
    wt = M.work_types[wt_id]
    return TaskInfo(id=tid, wbs=wt_id, name=wt.name, work_type_id=wt_id, profile=M.profiles[wt.profile],
                    observability=wt.observability, zone_key=zone, start=dt.date.fromisoformat(start),
                    end=dt.date.fromisoformat(end), is_summary=summary, planned=planned or {})


def run(observations, active, upcoming=(), recent=(), site_equipment=(), detectable=ALL):
    zw = ZoneWindow(zone_key="Z1", zone_name="Котлован", day=DAY, obs=list(observations), active=list(active),
                    upcoming=list(upcoming), recent=list(recent), site_equipment=set(site_equipment))
    return evaluate_window(zw, EngineContext(M, frozenset(detectable)))


def rules(findings):
    return sorted(f.rule for f in findings)


class TzExampleTest(unittest.TestCase):
    """Пример из ТЗ: на этапе «Устройство котлована» экскаватор есть, самосвалов нет."""

    def test_single_snapshot_gives_preliminary_warning(self):
        f, check = run([obs("10:30", ("excavator", 0.91))], [task("12.3.1")])
        self.assertEqual(rules(f), ["INCOMPLETE_SET"])
        self.assertEqual(f[0].status, "preliminary")
        self.assertEqual(f[0].severity, "warning")
        self.assertIn("снижение темпа", f[0].message)
        self.assertIn("Котлован", f[0].message)            # затронутая зона
        self.assertEqual(len(f[0].evidence), 1)             # снимок-доказательство
        self.assertTrue(f[0].evidence[0][1])                # с рамкой экскаватора

    def test_series_without_trucks_is_confirmed_and_critical(self):
        f, _ = run([obs("09:00", ("excavator", 0.9)), obs("10:30", ("excavator", 0.88)),
                    obs("11:30", ("excavator", 0.93))], [task("12.3.1")])
        self.assertEqual(rules(f), ["INCOMPLETE_SET"])     # MISSING_REQUIRED(вывоз) поглощён парным правилом
        self.assertEqual(f[0].status, "confirmed")
        self.assertEqual(f[0].severity, "critical")
        self.assertEqual(f[0].snapshots_count, 3)

    def test_trucks_present_is_ok(self):
        f, check = run([obs("08:00", ("excavator", 0.9), ("dump_truck", 0.86), ("dump_truck", 0.8))],
                       [task("12.3.1")])
        self.assertEqual(f, [])
        self.assertEqual(check["tasks"][0]["status"], "ok")

    def test_weak_truck_box_prevents_false_alarm(self):
        # сомнительная рамка самосвала (0.35) — для правил отсутствия техника «есть»
        f, _ = run([obs("08:00", ("excavator", 0.9), ("dump_truck", 0.35))], [task("12.3.1")])
        self.assertEqual(f, [])

    def test_short_gap_between_trucks_is_not_a_deviation(self):
        # один снимок из четырёх без самосвала — доля 0.25 < 0.3 (rules.yaml), отклонения нет
        f, _ = run([obs("09:00", ("excavator", 0.9), ("dump_truck", 0.8)), obs("10:00", ("excavator", 0.9)),
                    obs("11:00", ("excavator", 0.9), ("dump_truck", 0.9)),
                    obs("12:00", ("excavator", 0.9), ("dump_truck", 0.9))], [task("12.3.1")])
        self.assertEqual(f, [])

    def test_third_of_the_day_without_trucks_is_reported(self):
        # треть смены без самосвалов — часы потерянного темпа, предупреждение (одного снимка мало — предварительно)
        f, _ = run([obs("09:00", ("excavator", 0.9), ("dump_truck", 0.8)), obs("10:00", ("excavator", 0.9)),
                    obs("11:00", ("excavator", 0.9), ("dump_truck", 0.9))], [task("12.3.1")])
        self.assertEqual(rules(f), ["INCOMPLETE_SET"])
        self.assertEqual(f[0].status, "preliminary")


class AbsenceRulesTest(unittest.TestCase):
    def test_no_equipment_confirmed_after_an_hour(self):
        f, _ = run([obs("09:00"), obs("13:00")], [task("10.2-6")])
        self.assertEqual(rules(f), ["NO_EQUIPMENT"])
        self.assertEqual(f[0].status, "confirmed")
        self.assertEqual(f[0].severity, "critical")

    def test_tower_crane_alone_does_not_make_zone_busy(self):
        f, _ = run([obs("09:00", ("tower_crane", 0.9)), obs("12:00", ("tower_crane", 0.9))], [task("12.3.7-1")])
        self.assertEqual(rules(f), ["NO_EQUIPMENT"])

    def test_missing_excavator_when_only_trucks(self):
        f, _ = run([obs("09:00", ("dump_truck", 0.9)), obs("10:00", ("dump_truck", 0.9))], [task("12.3.7-1")])
        self.assertEqual(rules(f), ["MISSING_REQUIRED"])
        self.assertEqual(f[0].role, "разработка грунта")

    def test_shift_group_needs_series(self):
        # щебень подвозится «за смену»: один снимок без самосвала — ещё не отклонение
        f, check = run([obs("09:00", ("grader", 0.9), ("roller", 0.9))], [task("12.4.12-1", zone="Z1")])
        self.assertEqual(f, [])
        self.assertEqual(check["tasks"][0]["status"], "pending")

    def test_not_detectable_class_is_not_required(self):
        # модель v1 не знает каток: уплотнение не проверяется на отсутствие катка
        v1 = ALL - {"roller", "truck"}
        f, check = run([obs("09:00", ("bulldozer", 0.9)), obs("11:00", ("bulldozer", 0.9))],
                       [task("12.3.7-4")], detectable=v1)
        self.assertNotIn("MISSING_REQUIRED", rules(f))
        self.assertEqual(check["tasks"][0]["groups"][0]["state"], "not_detectable")

    def test_site_equipment_satisfies_crane(self):
        # башенный кран заявлен на площадке — монолитные работы не требуют его в кадре
        f, _ = run([obs("09:00", ("truck", 0.9)), obs("12:00", ("truck", 0.9))], [task("12.3.9", zone="Z1")],
                   site_equipment={"tower_crane"})
        self.assertEqual(f, [])

    def test_insufficient_count_from_schedule(self):
        f, _ = run([obs("09:00", ("concrete_mixer", 0.9), ("tower_crane", 0.9)),
                    obs("10:00", ("concrete_mixer", 0.9), ("tower_crane", 0.9))],
                   [task("12.3.4-6", planned={"concrete_mixer": 2})])
        self.assertEqual(rules(f), ["INSUFFICIENT_COUNT"])

    def test_mixer_without_pump_is_info_until_pump_is_detectable(self):
        f, _ = run([obs("09:00", ("concrete_mixer", 0.9))], [task("12.4.9")])
        self.assertEqual(rules(f), ["INCOMPLETE_SET"])
        self.assertEqual(f[0].severity, "info")
        self.assertIn("не распознаёт", f[0].message)

    def test_low_observability_is_not_checked(self):
        f, check = run([obs("09:00"), obs("12:00")], [task("12.4.11-1")])
        self.assertEqual(f, [])
        self.assertEqual(check["tasks"][0]["status"], "not_checked")


class UnexpectedRulesTest(unittest.TestCase):
    def test_roller_in_pit_is_unexpected(self):
        f, _ = run([obs("14:00", ("excavator", 0.9), ("dump_truck", 0.9), ("roller", 0.84))], [task("12.3.7-1")])
        self.assertEqual(rules(f), ["UNEXPECTED_EQUIPMENT"])
        self.assertEqual(f[0].cls, "roller")

    def test_future_stage_equipment_is_early_start(self):
        f, _ = run([obs("14:00", ("excavator", 0.9), ("dump_truck", 0.9), ("drilling_rig", 0.9))],
                   [task("12.3.7-1")], upcoming=[task("12.3.4-1", start="2026-10-05", end="2026-10-20", tid=2)])
        self.assertEqual(rules(f), ["EARLY_START"])
        self.assertEqual(f[0].task_id, 2)
        self.assertEqual(f[0].severity, "info")

    def test_finished_stage_equipment_is_overrun(self):
        f, _ = run([obs("10:00", ("grader", 0.9)), obs("12:00", ("grader", 0.9), ("roller", 0.8))],
                   [], recent=[task("10.11.3", start="2026-09-01", end="2026-09-20", tid=3,
                                    planned={"grader": 1, "roller": 1})])
        self.assertEqual(rules(f), ["STAGE_OVERRUN", "STAGE_OVERRUN"])
        self.assertEqual({x.cls for x in f}, {"grader", "roller"})

    def test_zone_without_tasks(self):
        f, _ = run([obs("10:00", ("excavator", 0.9))], [])
        self.assertEqual(rules(f), ["NO_ACTIVE_STAGE"])

    def test_delivery_truck_is_allowed_anywhere_with_work(self):
        f, _ = run([obs("10:00", ("excavator", 0.9), ("dump_truck", 0.9), ("truck", 0.9))], [task("12.3.7-1")])
        self.assertEqual(f, [])

    def test_low_confidence_box_is_not_unexpected(self):
        f, _ = run([obs("10:00", ("excavator", 0.9), ("dump_truck", 0.9), ("roller", 0.42))], [task("12.3.7-1")])
        self.assertEqual(f, [])


class DataQualityTest(unittest.TestCase):
    def test_dark_frames_give_no_data_not_no_equipment(self):
        f, check = run([obs("21:30", valid=False, reason="темно"), obs("22:00", valid=False, reason="темно")],
                       [task("12.3.7-1")])
        self.assertEqual(rules(f), ["NO_DATA"])
        self.assertEqual(check["status"], "no_data")

    def test_two_cameras_same_slot_take_max_not_sum(self):
        a = obs("10:00", ("excavator", 0.9))
        b = obs("10:02", ("excavator", 0.9), ("dump_truck", 0.9))
        merged = merge_slots([a, b])
        self.assertEqual(len(merged), 1)
        self.assertEqual(sorted(d.cls for d in merged[0].dets), ["dump_truck", "excavator"])


class TaskSelectionTest(unittest.TestCase):
    def test_active_upcoming_recent(self):
        ts = [task("12.3.7-1", "2026-09-10", "2026-10-10", tid=1), task("12.3.4-1", "2026-10-05", "2026-10-20", tid=2),
              task("10.11.3", "2026-09-01", "2026-09-20", tid=3), task("12.3.1", "2026-09-01", "2026-12-01", zone=None, tid=4)]
        a, u, r = tasks_for_zone(ts, "Z1", DAY, 21, 30)
        self.assertEqual([t.id for t in a], [1])
        self.assertEqual([t.id for t in u], [2])
        self.assertEqual([t.id for t in r], [3])
        a, _, _ = tasks_for_zone(ts, SITE_ZONE, DAY, 21, 30)
        self.assertEqual([t.id for t in a], [4])


class MethodologyTest(unittest.TestCase):
    def test_all_work_types_have_profiles_and_stages(self):
        self.assertEqual(len(M.work_types), 377)
        for wt in M.work_types.values():
            self.assertIn(wt.profile, M.profiles)
            if wt.level > 1:
                self.assertIn(wt.stage, {"S1", "S2", "S3", "S4", "S5"}, wt.id)

    def test_tz_equipment_classes_are_covered(self):
        tz = {k for k, v in M.classes.items() if v.get("tz")}
        self.assertEqual(tz, {"dump_truck", "excavator", "roller", "crane_manipulator", "concrete_mixer",
                              "bulldozer", "truck", "mobile_crane"})
        used = set().union(*(p.classes_used() for p in M.profiles.values()))
        self.assertTrue(tz <= used)

    def test_class_aliases(self):
        self.assertEqual(M.normalize_class("Dump truck"), "dump_truck")
        self.assertEqual(M.normalize_class("Autocran"), "mobile_crane")
        self.assertEqual(M.normalize_class("Bucket loader Big"), "loader")
        self.assertEqual(M.normalize_class("Каток"), "roller")
        self.assertIsNone(M.normalize_class("Cleaning equipment"))

    def test_schedule_name_matching(self):
        wt, how, _ = M.match_work_type(None, "Разработка грунта котлована")
        self.assertEqual((M.work_types[wt].profile, how), ("EXCAVATION", "name"))
        self.assertEqual(M.match_work_type("12.3.1.", "что угодно")[:2], ("12.3.1", "code"))
        wt, how, _ = M.match_work_type(None, "Устройство фундаментной плиты", "housing")
        self.assertEqual((wt, how), ("12.3.4-6", "name"))


if __name__ == "__main__":
    unittest.main()
