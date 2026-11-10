#!/usr/bin/env python3
from __future__ import annotations

import unittest

from can_fusion import CanFusion, CanFusionConfig


NAMES = [
    "1-Cheez-it",
    "2-Starkist_Tuna",
    "3-Scissors",
    "4-Frenchs_Mustard",
    "5-Tomato_Soup",
    "6-Foam_Brick",
    "7-Clamp",
    "8-Plastic_Banana",
    "9-Mug",
    "10-meat_can",
    "11-Padlock",
    "12-Baseball",
    "13-Bowl",
    "14-Sleeve",
    "15-Coca_cola_half",
    "16-ICRA_Duckie",
    "17-Eggs_Plush",
    "18-Dice",
    "19-Elmers_Glue",
    "20-Scotch_Tape",
    "21-Haribo_Golden_Bear",
    "22-E27_Adapter",
    "23-GU24_Adapter",
    "24-Spoon",
    "25-Flat_Screwdriver",
    "26-Plastic_Strawberry",
    "27-Coca_cola_full",
    "28-Kong_Duck_Dog_Toy",
    "29-Paper_Towels",
    "30-Stabilo_OHPen",
    "31-Plastic_White_Cup",
    "32-Wine_Glass",
    "33-Key",
    "34-Nail",
    "35-Book",
    "36-Adjustable_Wrench",
    "37-T-shirt",
    "38-Jumbo_Pencil_Cup",
    "39-Glove",
    "40-Pringles",
]
FRAME = (480, 640)
CONFIG = CanFusionConfig(
    secondary_conf=0.15,
    strong_conf=0.30,
    base_fallback_conf=0.75,
    temporal_min_hits=2,
    temporal_hold_inferences=3,
)


def det(class_id: int, conf: float, box: list[float], source: str = "model") -> dict:
    return {
        "id": class_id,
        "officialId": class_id + 1,
        "name": NAMES[class_id],
        "confidence": conf,
        "bbox": box,
        "source": source,
    }


def sec(class_id: int, conf: float, box: list[float]) -> dict:
    return {"class_id": class_id, "confidence": conf, "bbox": box}


class CanFusionTests(unittest.TestCase):
    def test_secondary_overrides_wrong_primary_can_and_confuser(self) -> None:
        fusion = CanFusion(NAMES, CONFIG)
        primary = [
            det(4, 0.62, [200, 150, 340, 330]),
            det(11, 0.60, [205, 155, 338, 328]),
            det(0, 0.90, [20, 20, 80, 80]),
        ]
        secondary = [sec(0, 0.80, [198, 148, 342, 332])]
        fused, debug = fusion.fuse(
            FRAME, primary, secondary, final_conf=0.15, use_temporal=False
        )
        ids = [item["id"] for item in fused]
        self.assertIn(1, ids)
        self.assertNotIn(4, ids)
        self.assertNotIn(11, ids)
        self.assertIn(0, ids)
        self.assertEqual(len(debug["suppressedConfusers"]), 1)

    def test_unrelated_objects_are_untouched(self) -> None:
        fusion = CanFusion(NAMES, CONFIG)
        primary = [det(0, 0.90, [20, 20, 100, 100]), det(2, 0.87, [400, 200, 470, 300])]
        secondary = [sec(1, 0.90, [150, 120, 280, 300])]
        fused, _ = fusion.fuse(
            FRAME, primary, secondary, final_conf=0.15, use_temporal=False
        )
        by_id = {item["id"]: item for item in fused}
        self.assertEqual(by_id[0]["bbox"], [20.0, 20.0, 100.0, 100.0])
        self.assertEqual(by_id[2]["bbox"], [400.0, 200.0, 470.0, 300.0])
        self.assertIn(4, by_id)

    def test_non_confuser_overlap_is_not_suppressed(self) -> None:
        fusion = CanFusion(NAMES, CONFIG)
        primary = [det(24, 0.91, [150, 120, 280, 300])]
        secondary = [sec(1, 0.90, [150, 120, 280, 300])]
        fused, _ = fusion.fuse(
            FRAME, primary, secondary, final_conf=0.15, use_temporal=False
        )
        self.assertIn(24, [item["id"] for item in fused])

    def test_no_secondary_keeps_only_strong_primary_can_fallback(self) -> None:
        fusion = CanFusion(NAMES, CONFIG)
        primary = [
            det(1, 0.45, [100, 100, 200, 240]),
            det(4, 0.89, [300, 100, 410, 260]),
        ]
        fused, _ = fusion.fuse(FRAME, primary, [], final_conf=0.15, use_temporal=False)
        ids = [item["id"] for item in fused]
        self.assertNotIn(1, ids)
        self.assertIn(4, ids)

    def test_duplicate_secondary_same_can_is_emitted_once(self) -> None:
        fusion = CanFusion(NAMES, CanFusionConfig(temporal_min_hits=1))
        secondary = [
            sec(0, 0.90, [100, 100, 220, 280]),
            sec(2, 0.55, [102, 102, 218, 278]),
        ]
        fused, _ = fusion.fuse(
            FRAME, [], secondary, final_conf=0.15, use_temporal=False
        )
        self.assertEqual(len([item for item in fused if item["id"] in {1, 4, 9}]), 1)

    def test_tiny_secondary_box_is_rejected(self) -> None:
        fusion = CanFusion(NAMES, CONFIG)
        secondary = [sec(1, 0.99, [100, 100, 112, 120])]
        fused, debug = fusion.fuse(
            FRAME, [], secondary, final_conf=0.15, use_temporal=False
        )
        self.assertFalse([item for item in fused if item["id"] in {1, 4, 9}])
        self.assertTrue(debug["rejectedSecondary"])

    def test_temporal_requires_two_hits(self) -> None:
        fusion = CanFusion(NAMES, CONFIG)
        secondary = [sec(1, 0.85, [100, 100, 240, 300])]
        first, _ = fusion.fuse(FRAME, [], secondary, final_conf=0.15, use_temporal=True)
        second, _ = fusion.fuse(FRAME, [], secondary, final_conf=0.15, use_temporal=True)
        self.assertFalse([item for item in first if item["id"] in {1, 4, 9}])
        self.assertTrue([item for item in second if item["id"] == 4])


if __name__ == "__main__":
    unittest.main()
