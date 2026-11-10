#!/usr/bin/env python3
from __future__ import annotations

import unittest

import cv2
import numpy as np

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


def blank_frame() -> np.ndarray:
    return np.full((480, 640, 3), 130, dtype=np.uint8)


def paint_silver_can(frame: np.ndarray, box: list[float]) -> None:
    x1, y1, x2, y2 = (int(round(value)) for value in box)
    cv2.rectangle(frame, (x1, y1), (x2, y2), (190, 190, 190), -1)


def paint_tomato_can(frame: np.ndarray, box: list[float]) -> None:
    x1, y1, x2, y2 = (int(round(value)) for value in box)
    cv2.rectangle(frame, (x1, y1), (x2, y2), (235, 235, 235), -1)
    cv2.rectangle(
        frame,
        (x1 + 8, y1 + 8),
        (x2 - 8, int(y1 + (y2 - y1) * 0.48)),
        (20, 20, 220),
        -1,
    )


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

    def test_raw_primary_class_id_is_accepted(self) -> None:
        fusion = CanFusion(NAMES, CONFIG)
        primary = [
            {
                "class_id": 0,
                "confidence": 0.90,
                "bbox": [20.0, 20.0, 100.0, 100.0],
            }
        ]
        fused, _ = fusion.fuse(
            FRAME, primary, [], final_conf=0.15, use_temporal=False
        )
        self.assertEqual(len(fused), 1)
        self.assertEqual(fused[0]["id"], 0)
        self.assertEqual(fused[0]["name"], NAMES[0])

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

    def test_temporal_class_switch_replaces_ghost_track(self) -> None:
        fusion = CanFusion(NAMES, CONFIG)
        tuna = [sec(0, 0.70, [100, 100, 240, 300])]
        tomato = [sec(1, 0.82, [102, 101, 238, 299])]
        fusion.fuse(FRAME, [], tuna, final_conf=0.15, use_temporal=True)
        fusion.fuse(FRAME, [], tomato, final_conf=0.15, use_temporal=True)
        switched, _ = fusion.fuse(
            FRAME, [], tomato, final_conf=0.15, use_temporal=True
        )
        cans = [item for item in switched if item["id"] in {1, 4, 9}]
        self.assertEqual([item["id"] for item in cans], [4])
        self.assertEqual(cans[0]["name"], NAMES[4])
        self.assertEqual(cans[0]["source"], "secondary_model")

    def test_temporal_anchor_keeps_ui_metadata(self) -> None:
        fusion = CanFusion(NAMES, CONFIG)
        secondary = [sec(2, 0.88, [300, 90, 420, 290])]
        fusion.fuse(FRAME, [], secondary, final_conf=0.15, use_temporal=True)
        fused, _ = fusion.fuse(
            FRAME, [], secondary, final_conf=0.15, use_temporal=True
        )
        can = next(item for item in fused if item["id"] == 9)
        self.assertEqual(can["name"], NAMES[9])
        self.assertEqual(can["officialId"], 10)

    def test_temporal_requires_two_hits(self) -> None:
        fusion = CanFusion(NAMES, CONFIG)
        secondary = [sec(1, 0.85, [100, 100, 240, 300])]
        first, _ = fusion.fuse(FRAME, [], secondary, final_conf=0.15, use_temporal=True)
        second, _ = fusion.fuse(FRAME, [], secondary, final_conf=0.15, use_temporal=True)
        self.assertFalse([item for item in first if item["id"] in {1, 4, 9}])
        self.assertTrue([item for item in second if item["id"] == 4])

    def test_three_separate_cans_are_not_merged(self) -> None:
        fusion = CanFusion(NAMES, CONFIG)
        secondary = [
            sec(0, 0.82, [345, 190, 504, 340]),
            sec(1, 0.78, [208, 11, 403, 227]),
            sec(2, 0.70, [134, 188, 325, 373]),
        ]
        fused, _ = fusion.fuse(
            FRAME, [], secondary, final_conf=0.15, use_temporal=False
        )
        cans = sorted(item["id"] for item in fused if item["id"] in {1, 4, 9})
        self.assertEqual(cans, [1, 4, 9])

    def test_tracker_does_not_reuse_track_across_can_classes(self) -> None:
        fusion = CanFusion(NAMES, CONFIG)
        tuna = [sec(0, 0.70, [100, 100, 240, 300])]
        tomato = [sec(1, 0.82, [102, 101, 238, 299])]
        fusion.fuse(FRAME, [], tuna, final_conf=0.15, use_temporal=True)
        fusion.fuse(FRAME, [], tomato, final_conf=0.15, use_temporal=True)
        self.assertEqual(
            {track.class_id for track in fusion.stabilizer.tracks},
            {1, 4},
        )

    def test_strong_primary_tuna_arbitrates_bad_secondary_tomato(self) -> None:
        fusion = CanFusion(NAMES, CONFIG)
        silver = [345, 190, 504, 340]
        gold = [134, 188, 325, 373]
        campbell = [208, 11, 403, 227]
        primary = [
            det(1, 0.81, silver),
            det(4, 0.758, campbell),
            det(11, 0.441, [214, 28, 419, 218]),
            det(18, 0.67, gold),
            det(11, 0.93, [510, 20, 590, 100]),
            det(18, 0.88, [20, 330, 90, 400]),
        ]
        # Reproduce the bad live tracker state: secondary calls the silver can
        # Tomato while omitting the real Campbell Tomato.
        secondary = [sec(1, 0.657, silver), sec(2, 0.699, gold)]
        fused = []
        for _ in range(2):
            fused, debug = fusion.fuse(
                FRAME, primary, secondary, final_conf=0.15, use_temporal=True
            )

        cans = {item["id"]: item for item in fused if item["id"] in {1, 4, 9}}
        self.assertEqual(set(cans), {1, 4, 9})
        self.assertEqual(cans[1]["source"], "primary_can_arbitrated")
        self.assertEqual(cans[1]["bbox"], [345.0, 190.0, 504.0, 340.0])
        self.assertEqual(cans[4]["bbox"], [208.0, 11.0, 403.0, 227.0])
        self.assertEqual(cans[9]["bbox"], [134.0, 188.0, 325.0, 373.0])
        self.assertEqual(debug["primaryCanArbitrations"][0]["primary"]["id"], 1)

        noncan_boxes = {
            item["id"]: item["bbox"] for item in fused if item["id"] not in {1, 4, 9}
        }
        self.assertEqual(noncan_boxes[11], [510.0, 20.0, 590.0, 100.0])
        self.assertEqual(noncan_boxes[18], [20.0, 330.0, 90.0, 400.0])

    def test_primary_fallback_tomato_suppresses_only_overlapping_baseball(self) -> None:
        fusion = CanFusion(NAMES, CONFIG)
        primary = [
            det(4, 0.90, [200, 20, 390, 220]),
            det(11, 0.80, [202, 22, 388, 218]),
            det(11, 0.91, [470, 20, 560, 110]),
        ]
        fused, debug = fusion.fuse(
            FRAME, primary, [], final_conf=0.15, use_temporal=False
        )
        baseballs = [item["bbox"] for item in fused if item["id"] == 11]
        self.assertEqual(baseballs, [[470.0, 20.0, 560.0, 110.0]])
        self.assertEqual(len(debug["suppressedConfusers"]), 1)

    def test_secondary_meat_suppresses_only_overlapping_glue(self) -> None:
        fusion = CanFusion(NAMES, CONFIG)
        primary = [
            det(18, 0.82, [130, 185, 326, 374]),
            det(18, 0.90, [470, 300, 570, 420]),
        ]
        secondary = [sec(2, 0.78, [134, 188, 325, 373])]
        fused, debug = fusion.fuse(
            FRAME, primary, secondary, final_conf=0.15, use_temporal=False
        )
        glue_boxes = [item["bbox"] for item in fused if item["id"] == 18]
        self.assertEqual(glue_boxes, [[470.0, 300.0, 570.0, 420.0]])
        self.assertEqual(len(debug["suppressedConfusers"]), 1)

    def test_silver_can_tuna_tomato_conflict_uses_color_and_primary(self) -> None:
        fusion = CanFusion(
            NAMES,
            CanFusionConfig(secondary_conf=0.005, temporal_min_hits=1),
        )
        silver = [345, 190, 501, 340]
        frame = blank_frame()
        paint_silver_can(frame, silver)
        primary = [det(1, 0.203, silver)]
        secondary = [sec(1, 0.527, silver), sec(0, 0.504, silver)]

        fused, debug = fusion.fuse(
            FRAME,
            primary,
            secondary,
            final_conf=0.15,
            use_temporal=False,
            frame=frame,
        )

        cans = [item for item in fused if item["id"] in {1, 4, 9}]
        self.assertEqual([item["id"] for item in cans], [1])
        self.assertEqual(cans[0]["name"], NAMES[1])
        self.assertTrue(debug["secondaryEvidenceArbitrations"])

    def test_silver_can_missing_tuna_candidate_is_relabelled(self) -> None:
        fusion = CanFusion(
            NAMES,
            CanFusionConfig(secondary_conf=0.005, temporal_min_hits=1),
        )
        silver = [345, 190, 501, 340]
        frame = blank_frame()
        paint_silver_can(frame, silver)
        primary = [det(1, 0.203, silver)]
        secondary = [sec(1, 0.527, silver)]

        fused, _ = fusion.fuse(
            FRAME,
            primary,
            secondary,
            final_conf=0.15,
            use_temporal=False,
            frame=frame,
        )

        can = next(item for item in fused if item["id"] in {1, 4, 9})
        self.assertEqual(can["id"], 1)
        self.assertEqual(can["source"], "secondary_color_arbitrated")
        self.assertFalse(can["secondaryClassMatched"])

    def test_low_confidence_tomato_is_recovered_and_hides_only_local_baseball(self) -> None:
        fusion = CanFusion(
            NAMES,
            CanFusionConfig(secondary_conf=0.005, temporal_min_hits=1),
        )
        campbell = [214, 27, 421, 215]
        frame = blank_frame()
        paint_tomato_can(frame, campbell)
        primary = [
            det(4, 0.129, campbell),
            det(11, 0.814, campbell),
            det(11, 0.910, [510, 20, 590, 100]),
            det(18, 0.900, [20, 330, 90, 400]),
        ]
        secondary = [sec(1, 0.0096, campbell)]

        fused, debug = fusion.fuse(
            FRAME,
            primary,
            secondary,
            final_conf=0.15,
            use_temporal=False,
            frame=frame,
        )

        cans = [item for item in fused if item["id"] in {1, 4, 9}]
        self.assertEqual([item["id"] for item in cans], [4])
        self.assertGreaterEqual(cans[0]["confidence"], 0.15)
        noncan_boxes = {
            item["id"]: item["bbox"] for item in fused if item["id"] not in {1, 4, 9}
        }
        self.assertEqual(noncan_boxes[11], [510.0, 20.0, 590.0, 100.0])
        self.assertEqual(noncan_boxes[18], [20.0, 330.0, 90.0, 400.0])
        self.assertEqual(len(debug["suppressedConfusers"]), 1)

    def test_primary_only_red_tomato_is_rescued_when_secondary_omits_it(self) -> None:
        fusion = CanFusion(
            NAMES,
            CanFusionConfig(secondary_conf=0.005, temporal_min_hits=1),
        )
        campbell = [214, 25, 405, 216]
        frame = blank_frame()
        paint_tomato_can(frame, campbell)
        primary = [
            det(4, 0.058, campbell),
            det(11, 0.816, campbell),
            det(11, 0.910, [510, 20, 590, 100]),
            det(18, 0.900, [20, 330, 90, 400]),
        ]

        fused, debug = fusion.fuse(
            FRAME,
            primary,
            [],
            final_conf=0.15,
            use_temporal=False,
            frame=frame,
        )

        cans = [item for item in fused if item["id"] in {1, 4, 9}]
        self.assertEqual([item["id"] for item in cans], [4])
        self.assertEqual(cans[0]["source"], "primary_color_fallback")
        self.assertGreaterEqual(cans[0]["confidence"], 0.15)
        noncan_boxes = {
            item["id"]: item["bbox"] for item in fused if item["id"] not in {1, 4, 9}
        }
        self.assertEqual(noncan_boxes[11], [510.0, 20.0, 590.0, 100.0])
        self.assertEqual(noncan_boxes[18], [20.0, 330.0, 90.0, 400.0])
        self.assertEqual(len(debug["primaryColorRescues"]), 1)
        self.assertEqual(len(debug["suppressedConfusers"]), 1)

    def test_primary_color_rescue_is_vetoed_by_colocated_non_confuser(self) -> None:
        fusion = CanFusion(
            NAMES,
            CanFusionConfig(secondary_conf=0.005, temporal_min_hits=1),
        )
        red_bottle = [278, 148, 405, 278]
        frame = blank_frame()
        paint_tomato_can(frame, red_bottle)
        primary = [
            det(4, 0.428, red_bottle),
            det(26, 0.722, red_bottle),
        ]

        fused, debug = fusion.fuse(
            FRAME,
            primary,
            [],
            final_conf=0.15,
            use_temporal=False,
            frame=frame,
        )

        self.assertFalse([item for item in fused if item["id"] in {1, 4, 9}])
        self.assertIn(26, [item["id"] for item in fused])
        self.assertEqual(debug["primaryColorRescues"], [])
        self.assertEqual(
            debug["primaryColorAssessments"][0]["reason"],
            "non_confuser_overlap",
        )

    def test_primary_color_rescue_requires_a_white_can_body(self) -> None:
        fusion = CanFusion(
            NAMES,
            CanFusionConfig(secondary_conf=0.005, temporal_min_hits=1),
        )
        red_object = [200, 120, 360, 300]
        frame = blank_frame()
        x1, y1, x2, y2 = red_object
        cv2.rectangle(frame, (x1, y1), (x2, y2), (20, 20, 220), -1)
        primary = [det(4, 0.60, red_object)]

        fused, debug = fusion.fuse(
            FRAME,
            primary,
            [],
            final_conf=0.15,
            use_temporal=False,
            frame=frame,
        )

        self.assertFalse([item for item in fused if item["id"] in {1, 4, 9}])
        self.assertEqual(debug["primaryColorRescues"], [])
        self.assertEqual(
            debug["primaryColorAssessments"][0]["reason"],
            "insufficient_white_body",
        )

    def test_low_confidence_background_can_candidates_are_rejected(self) -> None:
        fusion = CanFusion(
            NAMES,
            CanFusionConfig(secondary_conf=0.005, temporal_min_hits=1),
        )
        frame = blank_frame()
        secondary = [
            sec(0, 0.0190, [9, 0, 134, 40]),
            sec(2, 0.0053, [531, 208, 595, 308]),
        ]

        fused, debug = fusion.fuse(
            FRAME,
            [],
            secondary,
            final_conf=0.15,
            use_temporal=False,
            frame=frame,
        )

        self.assertFalse([item for item in fused if item["id"] in {1, 4, 9}])
        self.assertTrue(debug["rejectedSecondary"])


if __name__ == "__main__":
    unittest.main()
