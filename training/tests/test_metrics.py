"""AP50 на маленьких примерах с известным ответом."""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from metrics import average_precision, evaluate  # noqa: E402


class MetricsTest(unittest.TestCase):
    def test_perfect_detector(self):
        gt = {1: [("roller", 0, 0, 10, 10)], 2: [("roller", 5, 5, 20, 20), ("truck", 0, 0, 5, 5)]}
        pred = {1: [("roller", 0, 0, 10, 10, 0.9)], 2: [("roller", 5, 5, 20, 20, 0.8), ("truck", 0, 0, 5, 5, 0.7)]}
        r = evaluate(gt, pred, ["roller", "truck"])
        self.assertEqual(r["roller"]["ap50"], 1.0)
        self.assertEqual(r["truck"]["recall"], 1.0)
        self.assertEqual(r["__mean__"]["ap50"], 1.0)

    def test_false_positive_ranked_first_halves_precision(self):
        gt = {1: [("roller", 0, 0, 10, 10)]}
        pred = {1: [("roller", 50, 50, 60, 60, 0.95), ("roller", 0, 0, 10, 10, 0.9)]}
        r = evaluate(gt, pred, ["roller"])
        self.assertAlmostEqual(r["roller"]["ap50"], 0.5)
        self.assertAlmostEqual(r["roller"]["precision"], 0.5)

    def test_duplicate_detection_is_false_positive(self):
        gt = {1: [("truck", 0, 0, 10, 10)]}
        pred = {1: [("truck", 0, 0, 10, 10, 0.9), ("truck", 0, 0, 10, 10, 0.8)]}
        r = evaluate(gt, pred, ["truck"])
        self.assertEqual(r["truck"]["ap50"], 1.0)          # дубль ниже по рангу не портит AP
        self.assertEqual(r["truck"]["precision"], 0.5)

    def test_missed_objects(self):
        gt = {1: [("truck", 0, 0, 10, 10)], 2: [("truck", 0, 0, 10, 10)]}
        pred = {1: [("truck", 0, 0, 10, 10, 0.9)]}
        self.assertEqual(evaluate(gt, pred, ["truck"])["truck"]["ap50"], 0.5)

    def test_ap_envelope(self):
        import numpy as np
        self.assertAlmostEqual(average_precision(np.array([0.5, 0.5, 1.0]), np.array([1.0, 0.5, 0.67])), 0.835, places=3)


if __name__ == "__main__":
    unittest.main()
