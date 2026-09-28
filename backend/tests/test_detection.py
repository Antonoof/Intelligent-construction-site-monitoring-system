"""Детекторы: декодирование ONNX-выходов RF-DETR, детекция по фрагментам, привязка к зонам."""
import unittest

import numpy as np
from PIL import Image

from app.detection.base import RawDet
from app.detection.onnx_backend import decode, preprocess, resize_half_pixel
from app.detection.tiling import detect_tiled, nms, tiles_grid
from app.zones import assign_zone, point_in_polygon


class OnnxDecodeTest(unittest.TestCase):
    def test_decode_topk_and_background_slot(self):
        q, c = 5, 4                                   # 3 класса + фон (последний слот)
        logits = np.full((1, q, c), -8.0, dtype=np.float32)
        logits[0, 1, 2] = 3.0                         # запрос 1 → класс 2, уверенность ~0.95
        logits[0, 3, 0] = 0.5                         # запрос 3 → класс 0, ~0.62
        logits[0, 4, 3] = 9.0                         # фон — не должен попасть в выдачу
        dets = np.zeros((1, q, 4), dtype=np.float32)
        dets[0, 1] = [0.5, 0.5, 0.2, 0.4]
        dets[0, 3] = [0.25, 0.75, 0.1, 0.1]
        out = decode(dets, logits, 1000, 500, threshold=0.3)
        self.assertEqual([o[0] for o in out], [2, 0])
        self.assertAlmostEqual(out[0][1], 1 / (1 + np.exp(-3.0)), places=5)
        np.testing.assert_allclose(out[0][2], (400, 150, 600, 350), atol=1e-3)
        np.testing.assert_allclose(out[1][2], (200, 350, 300, 400), atol=1e-3)

    def test_preprocess_shape_and_normalization(self):
        img = Image.new("RGB", (64, 32), (124, 116, 104))     # ≈ среднее ImageNet → около нуля
        x = preprocess(img, 28, 28)
        self.assertEqual(x.shape, (1, 3, 28, 28))
        self.assertLess(float(np.abs(x).max()), 0.02)

    def test_resize_half_pixel_identity_and_mean(self):
        a = np.random.default_rng(0).random((20, 30, 3)).astype(np.float32)
        np.testing.assert_allclose(resize_half_pixel(a, 20, 30), a, atol=1e-6)
        self.assertAlmostEqual(float(resize_half_pixel(a, 10, 15).mean()), float(a.mean()), places=2)


class TilingTest(unittest.TestCase):
    def test_grid_covers_frame(self):
        g = tiles_grid(1280, 720, 3)
        self.assertEqual(len(g), 9)
        self.assertEqual((g[0][0], g[0][1]), (0, 0))
        self.assertEqual((g[-1][2], g[-1][3]), (1280, 720))

    def test_small_boxes_from_tiles_merged(self):
        img = Image.new("RGB", (1200, 600))

        def fake(im):
            if im.size == (1200, 600):          # целый кадр: одна крупная машина
                return [RawDet("excavator", 0.9, (100, 100, 500, 400))]
            return [RawDet("dump_truck", 0.7, (10, 10, 40, 30)), RawDet("excavator", 0.6, (0, 0, im.size[0], im.size[1]))]

        out = detect_tiled(img, fake, n=3)
        self.assertEqual(sum(1 for d in out if d.cls == "excavator"), 1)   # крупные рамки с фрагментов отброшены
        self.assertEqual(sum(1 for d in out if d.cls == "dump_truck"), 9)

    def test_nms(self):
        d = [RawDet("excavator", 0.9, (0, 0, 10, 10)), RawDet("excavator", 0.8, (1, 1, 10, 10)),
             RawDet("dump_truck", 0.8, (1, 1, 10, 10))]
        self.assertEqual(len(nms(d)), 2)


class ZonesTest(unittest.TestCase):
    def test_ground_point_decides_zone(self):
        left = [[0, 0.5], [0.5, 0.5], [0.5, 1], [0, 1]]
        right = [[0.5, 0.5], [1, 0.5], [1, 1], [0.5, 1]]
        polys = [("Z1", left), ("Z2", right)]
        # высокая стрела заходит в правую зону, но точка опоры — в левой
        self.assertEqual(assign_zone((100, 100, 700, 900), 1000, 1000, polys, None), "Z1")
        self.assertEqual(assign_zone((600, 600, 900, 900), 1000, 1000, polys, None), "Z2")
        self.assertEqual(assign_zone((10, 10, 50, 50), 1000, 1000, polys, "Z9"), "Z9")
        self.assertTrue(point_in_polygon(0.25, 0.75, left))
        self.assertFalse(point_in_polygon(0.75, 0.75, left))


if __name__ == "__main__":
    unittest.main()
