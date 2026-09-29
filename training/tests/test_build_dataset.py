"""Проверка сборки датасета на игрушечных источниках всех трёх форматов (YOLO, COCO, VOC)."""
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

from PIL import Image, ImageDraw

HERE = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("bd", HERE / "05_build_detector_dataset.py")
bd = importlib.util.module_from_spec(spec)
sys.modules["bd"] = bd
spec.loader.exec_module(bd)


def img(path: Path, color=(120, 140, 160), size=(320, 240), seed=0):
    import random
    rnd = random.Random(seed)
    path.parent.mkdir(parents=True, exist_ok=True)
    im = Image.new("RGB", size, color)
    d = ImageDraw.Draw(im)
    for _ in range(12):                          # случайные фигуры — снимки заведомо разные
        x, y = rnd.randint(0, 300), rnd.randint(0, 220)
        d.rectangle([x, y, x + rnd.randint(10, 90), y + rnd.randint(10, 60)],
                    fill=tuple(rnd.randint(0, 255) for _ in range(3)))
    im.save(path)


class BuildDatasetTest(unittest.TestCase):
    def test_three_formats_merge(self):
        tmp = Path(tempfile.mkdtemp())
        raw = tmp / "raw"
        # YOLO: images/ + labels/, классы из datasets.yaml (Roller → каток, Gazelle → грузовик, Cleaning — пропуск)
        for i in range(12):
            img(raw / "ds1" / "images" / f"a{i}.jpg", seed=i)
            (raw / "ds1" / "labels").mkdir(parents=True, exist_ok=True)
            (raw / "ds1" / "labels" / f"a{i}.txt").write_text(
                f"{i % 3} 0.3 0.4 0.2 0.3\n5 0.7 0.7 0.2 0.2\n11 0.5 0.5 0.1 0.1\n")
        # COCO
        imgs, anns = [], []
        for i in range(6):
            img(raw / "ds5" / "train" / f"c{i}.jpg", color=(90, 90, 100), seed=100 + i)
            imgs.append({"id": i, "file_name": f"c{i}.jpg", "width": 320, "height": 240})
            anns.append({"id": i, "image_id": i, "category_id": 2, "bbox": [20, 30, 80, 60]})
            anns.append({"id": 100 + i, "image_id": i, "category_id": 3, "bbox": [150, 100, 60, 50]})
        (raw / "ds5" / "train" / "_annotations.coco.json").write_text(json.dumps(
            {"images": imgs, "annotations": anns, "categories": [{"id": 1, "name": "crane"}, {"id": 2, "name": "pump-truck"},
                                                                 {"id": 3, "name": "roller"}]}))
        # VOC
        for i in range(5):
            img(raw / "ds4" / f"v{i}.jpg", color=(230, 232, 235), seed=200 + i)   # «зима»
            (raw / "ds4" / f"v{i}.xml").write_text(
                f"<annotation><filename>v{i}.jpg</filename><size><width>320</width><height>240</height></size>"
                "<object><name>Truck</name><bndbox><xmin>10</xmin><ymin>20</ymin><xmax>120</xmax><ymax>90</ymax></bndbox></object>"
                "<object><name>Excavator</name><bndbox><xmin>150</xmin><ymin>100</ymin><xmax>250</xmax><ymax>200</ymax></bndbox></object>"
                "</annotation>")
        # тот же снимок, пересохранённый в другом датасете, — перцептивный дубликат
        Image.open(raw / "ds1" / "images" / "a0.jpg").save(raw / "ds4" / "dup.jpg", quality=70)
        (raw / "ds4" / "dup.xml").write_text(
            "<annotation><filename>dup.jpg</filename><object><name>Truck</name><bndbox><xmin>10</xmin><ymin>20</ymin>"
            "<xmax>120</xmax><ymax>90</ymax></bndbox></object></annotation>")
        cfg = {"target_classes": ["excavator", "dump_truck", "roller", "truck", "concrete_pump"],
               "sources": [
                   {"id": "ds1", "format": "yolo",
                    "classes": ["Dump truck", "Excavator", "Motor grader", "Roller", "Crane manipulator", "Gazelle",
                                "Forklift Standart", "Bucket loader Big", "Mixer", "Tanker", "Bulldozer", "Cleaning equipment"],
                    "map": {"Gazelle": "truck", "Cleaning equipment": "skip"}},
                   {"id": "ds5", "format": "coco", "map": {"crane": "skip", "pump-truck": "concrete_pump"}},
                   {"id": "ds4", "format": "voc"}],
               "split": {"seed": 1, "valid": 0.1, "test": 0.2}}
        res = bd.build(raw, tmp / "out", cfg)
        self.assertEqual(res["sources"]["ds1"]["images"], 12)
        self.assertEqual(res["sources"]["ds5"]["boxes"], 12)
        self.assertEqual(res["sources"]["ds4"]["boxes"], 11)
        self.assertEqual(res["images"], 23)                                   # 24 снимка, 1 дубликат
        self.assertIn("Cleaning equipment", res["sources"]["ds1"]["unmapped"])
        self.assertIn("Motor grader", res["sources"]["ds1"]["unmapped"])       # грейдера нет в target_classes
        total = 0
        names = set()
        for s in ("train", "valid", "test"):
            d = json.loads((tmp / "out" / s / "_annotations.coco.json").read_text())
            total += len(d["images"])
            cid = {c["id"]: c["name"] for c in d["categories"]}
            names |= {cid[a["category_id"]] for a in d["annotations"]}
            for im in d["images"]:
                self.assertTrue((tmp / "out" / s / im["file_name"]).exists())
        self.assertEqual(total, res["images"])
        self.assertEqual(names, {"excavator", "dump_truck", "roller", "truck", "concrete_pump"})
        manifest = (tmp / "out" / "manifest.csv").read_text()
        self.assertIn("winter", manifest)
        self.assertTrue((tmp / "out" / "stats.md").exists())


if __name__ == "__main__":
    unittest.main()
