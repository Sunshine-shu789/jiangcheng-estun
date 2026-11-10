#!/usr/bin/env python3
"""Regression tests for the three-can object-level arbitration."""

from __future__ import annotations

import unittest

import numpy as np

from can_postprocess import (
    MEAT_CAN_ID,
    TOMATO_ID,
    TUNA_ID,
    CanPostProcessor,
)


NAMES = [f"class-{index}" for index in range(40)]


def detection(class_id: int, confidence: float, box: list[float]) -> dict:
    return {
        "id": class_id,
        "officialId": class_id + 1,
        "name": NAMES[class_id],
        "confidence": confidence,
        "bbox": box,
    }


class CanObjectArbitrationTest(unittest.TestCase):
    def make_processor(self, *, hold_inferences: int = 0) -> CanPostProcessor:
        processor = CanPostProcessor(
            NAMES,
            {
                "temporal_min_hits": 1,
                "temporal_hold_inferences": hold_inferences,
            },
        )
        # Keep arbitration tests focused on detections; the gold-colour fallback
        # has its own image-based regression coverage in the full audit.
        processor._gold_candidates = lambda *_args, **_kwargs: ([], [])  # type: ignore[method-assign]
        return processor

    @staticmethod
    def frame() -> np.ndarray:
        return np.zeros((480, 640, 3), dtype=np.uint8)

    def test_distinct_top_view_tomato_does_not_require_red_label(self) -> None:
        processor = self.make_processor()
        raw = [
            detection(TUNA_ID, 0.92, [40, 260, 160, 390]),
            detection(TOMATO_ID, 0.90, [250, 140, 375, 260]),
            detection(MEAT_CAN_ID, 0.48, [470, 280, 590, 420]),
        ]

        output, debug = processor.process(self.frame(), raw, final_conf=0.15)

        self.assertEqual({item["id"] for item in output}, {TUNA_ID, TOMATO_ID, MEAT_CAN_ID})
        self.assertEqual(debug["tomatoDecision"], "accept_distinct_object")

    def test_tomato_duplicate_on_tuna_object_is_rejected_without_red(self) -> None:
        processor = self.make_processor()
        box = [150, 280, 275, 405]
        raw = [
            detection(TUNA_ID, 0.82, box),
            detection(TOMATO_ID, 0.83, [151, 281, 276, 406]),
        ]

        output, debug = processor.process(self.frame(), raw, final_conf=0.15)

        self.assertEqual([item["id"] for item in output], [TUNA_ID])
        self.assertEqual(
            debug["tomatoDecision"],
            "reject_tomato_on_tuna_object_without_red",
        )

    def test_red_label_replaces_tuna_on_shared_object(self) -> None:
        processor = self.make_processor()
        frame = self.frame()
        frame[300:375, 175:250] = (0, 0, 220)
        box = [150, 280, 275, 405]
        raw = [
            detection(TUNA_ID, 0.82, box),
            detection(TOMATO_ID, 0.83, [151, 281, 276, 406]),
        ]

        output, debug = processor.process(frame, raw, final_conf=0.15)

        self.assertEqual([item["id"] for item in output], [TOMATO_ID])
        self.assertEqual(debug["tomatoDecision"], "replace_tuna_on_shared_object")

    def test_gold_rule_only_suppresses_explicit_confusers(self) -> None:
        processor = self.make_processor()
        gold = processor._detection(
            MEAT_CAN_ID,
            0.88,
            [260, 150, 350, 280],
            "gold_shape_rule",
        )
        processor._gold_candidates = lambda *_args, **_kwargs: ([gold], [])  # type: ignore[method-assign]
        raw = [
            detection(2, 0.80, [260, 150, 350, 280]),  # Scissors, not a known proxy
            detection(21, 0.62, [261, 151, 349, 279]),  # E27 adapter, known proxy
            detection(22, 0.60, [262, 152, 348, 278]),  # GU24 adapter, known proxy
        ]

        output, _ = processor.process(self.frame(), raw, final_conf=0.15)

        self.assertEqual({item["id"] for item in output}, {2, MEAT_CAN_ID})

    def test_low_confidence_tuna_does_not_delete_true_padlock(self) -> None:
        processor = self.make_processor()
        box = [120, 155, 226, 248]
        raw = [
            detection(TUNA_ID, 0.33, box),
            detection(11, 0.96, [121, 156, 227, 249]),
        ]

        output, debug = processor.process(self.frame(), raw, final_conf=0.15)

        self.assertEqual([item["id"] for item in output], [11])
        self.assertEqual(
            debug["rejectedCansByStrongerConfuser"][0]["strongerClass"], 11
        )

    def test_low_confidence_meat_can_does_not_delete_true_dice(self) -> None:
        processor = self.make_processor()
        box = [140, 153, 255, 281]
        raw = [
            detection(MEAT_CAN_ID, 0.35, box),
            detection(18, 0.78, [141, 154, 256, 282]),
        ]

        output, debug = processor.process(self.frame(), raw, final_conf=0.15)

        self.assertEqual([item["id"] for item in output], [18])
        self.assertEqual(
            debug["rejectedCansByStrongerConfuser"][0]["strongerClass"], 18
        )

    def test_tomato_does_not_delete_true_coca_cola_can(self) -> None:
        processor = self.make_processor()
        raw = [
            detection(26, 0.72, [280, 146, 409, 282]),
            detection(TOMATO_ID, 0.43, [278, 148, 405, 278]),
        ]

        output, debug = processor.process(self.frame(), raw, final_conf=0.15)

        self.assertEqual([item["id"] for item in output], [26])
        self.assertEqual(
            debug["rejectedCansByStrongerConfuser"][0]["strongerClass"], 26
        )

    def test_can_box_touching_image_border_is_rejected(self) -> None:
        processor = self.make_processor()
        raw = [detection(TUNA_ID, 0.31, [540, 100, 640, 200])]

        output, debug = processor.process(self.frame(), raw, final_conf=0.15)

        self.assertEqual(output, [])
        self.assertEqual(
            debug["rejectedImplausibleCanCandidates"][0]["reason"],
            "touches_image_border",
        )

    def test_red_confirmed_low_confidence_tomato_beats_paper_towel_confusion(self) -> None:
        processor = self.make_processor()
        frame = self.frame()
        frame[180:300, 195:320] = (0, 0, 220)
        raw = [
            detection(TUNA_ID, 0.79, [340, 235, 493, 381]),
            detection(TOMATO_ID, 0.06, [183, 153, 340, 337]),
            detection(28, 0.20, [179, 147, 345, 345]),
        ]

        output, _ = processor.process(frame, raw, final_conf=0.15)

        self.assertEqual({item["id"] for item in output}, {TUNA_ID, TOMATO_ID})

    def test_lower_confidence_distinct_tomato_beats_higher_tomato_on_tuna(self) -> None:
        processor = self.make_processor()
        frame = self.frame()
        frame[180:300, 195:320] = (0, 0, 220)
        raw = [
            detection(TUNA_ID, 0.79, [340, 235, 493, 381]),
            # The model's highest-confidence tomato hypothesis is misplaced on
            # the tuna can. The true lower-confidence tomato still has red-label
            # and Paper-Towels context evidence.
            detection(TOMATO_ID, 0.09, [341, 236, 492, 380]),
            detection(TOMATO_ID, 0.06, [183, 153, 340, 337]),
            detection(28, 0.20, [179, 147, 345, 345]),
        ]

        output, debug = processor.process(frame, raw, final_conf=0.15)

        self.assertEqual({item["id"] for item in output}, {TUNA_ID, TOMATO_ID})
        self.assertEqual(debug["tomatoDecision"], "accept_low_conf_with_red_and_overlap_context")
        self.assertTrue(any(item["sameObjectAsTuna"] for item in debug["tomatoCandidates"]))

    def test_held_tomato_suppresses_its_paper_towel_duplicate(self) -> None:
        processor = self.make_processor(hold_inferences=4)
        first_frame = self.frame()
        first_frame[180:300, 195:320] = (0, 0, 220)
        first_raw = [
            detection(TUNA_ID, 0.79, [340, 235, 493, 381]),
            detection(TOMATO_ID, 0.06, [183, 153, 340, 337]),
            detection(28, 0.20, [179, 147, 345, 345]),
        ]
        first_output, _ = processor.process(first_frame, first_raw, final_conf=0.15)
        self.assertIn(TOMATO_ID, {item["id"] for item in first_output})

        # The current tomato hypothesis jumps onto the tuna, so no tomato is
        # selected this frame. The emitted tomato still comes from its held track,
        # and that same track must suppress the Paper-Towels duplicate.
        second_raw = [
            detection(TUNA_ID, 0.82, [340, 235, 493, 381]),
            detection(TOMATO_ID, 0.12, [341, 236, 492, 380]),
            detection(28, 0.21, [179, 147, 345, 345]),
        ]
        second_output, debug = processor.process(self.frame(), second_raw, final_conf=0.15)

        self.assertNotIn(TOMATO_ID, {item["id"] for item in debug["selectedCans"]})
        self.assertIn(TOMATO_ID, {item["id"] for item in debug["emittedCans"]})
        self.assertEqual({item["id"] for item in second_output}, {TUNA_ID, TOMATO_ID})

    def test_stable_tuna_survives_baseball_relabel_and_suppresses_it(self) -> None:
        processor = self.make_processor(hold_inferences=8)
        frame = self.frame()
        frame[180:300, 195:320] = (0, 0, 220)
        tuna_box = [340, 235, 493, 381]
        tomato_box = [183, 153, 340, 337]
        paper_box = [179, 147, 345, 345]
        first_raw = [
            detection(TUNA_ID, 0.79, tuna_box),
            detection(TOMATO_ID, 0.06, tomato_box),
            detection(28, 0.20, paper_box),
        ]
        processor.process(frame, first_raw, final_conf=0.15)

        # For several inferences the model calls the physical tuna can a
        # Baseball. The confirmed tuna track must retain the object identity and
        # suppress that same-object duplicate instead of showing ID11 in between.
        dropout_raw = [
            detection(11, 0.68, [342, 237, 492, 383]),
            detection(TOMATO_ID, 0.06, tomato_box),
            detection(28, 0.20, paper_box),
        ]
        for _ in range(5):
            output, debug = processor.process(frame, dropout_raw, final_conf=0.15)
            self.assertIn(TUNA_ID, {item["id"] for item in debug["emittedCans"]})
            self.assertEqual({item["id"] for item in output}, {TUNA_ID, TOMATO_ID})

    def test_current_selected_tuna_suppresses_baseball_without_emitted_track(self) -> None:
        processor = self.make_processor()
        tuna_box = [340, 235, 493, 381]
        raw = [
            detection(TUNA_ID, 0.72, tuna_box),
            detection(11, 0.68, [342, 237, 492, 383]),
        ]

        output, debug = processor.process(self.frame(), raw, final_conf=0.15)

        self.assertIn(TUNA_ID, {item["id"] for item in debug["selectedCans"]})
        self.assertNotIn(11, {item["id"] for item in output})

    def test_isolated_low_confidence_tomato_uses_red_and_white_geometry(self) -> None:
        processor = self.make_processor()
        frame = self.frame()
        frame[110:200, 260:380] = (0, 0, 220)
        frame[200:290, 260:380] = (245, 245, 245)
        raw = [detection(TOMATO_ID, 0.007, [250, 100, 390, 300])]

        output, debug = processor.process(frame, raw, final_conf=0.15)

        self.assertEqual([item["id"] for item in output], [TOMATO_ID])
        self.assertEqual(debug["tomatoDecision"], "accept_low_conf_with_red_and_white_geometry")

    def test_red_label_on_coca_cola_cannot_create_low_confidence_tomato(self) -> None:
        processor = self.make_processor()
        frame = self.frame()
        frame[180:300, 110:290] = (0, 0, 220)
        frame[250:300, 110:290] = (245, 245, 245)
        raw = [
            detection(26, 0.86, [97, 145, 302, 321]),
            detection(TOMATO_ID, 0.017, [97, 145, 302, 321]),
        ]

        output, debug = processor.process(frame, raw, final_conf=0.15)

        self.assertEqual([item["id"] for item in output], [26])
        self.assertEqual(debug["tomatoDecision"], "reject_insufficient_evidence")

    @staticmethod
    def tomato_like_frame() -> np.ndarray:
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        frame[110:200, 260:380] = (0, 0, 220)
        frame[200:290, 260:380] = (245, 245, 245)
        return frame

    def test_tomato_colour_rule_recovers_missing_id4_from_haribo(self) -> None:
        processor = self.make_processor()
        raw = [detection(20, 0.014, [250, 100, 390, 300])]

        output, debug = processor.process(
            self.tomato_like_frame(), raw, final_conf=0.15
        )

        self.assertEqual([item["id"] for item in output], [TOMATO_ID])
        self.assertEqual(debug["tomatoDecision"], "recover_from_color_geometry")
        self.assertEqual(debug["emittedCans"][0]["source"], "tomato_color_rule")
        self.assertEqual(debug["tomatoColorCandidates"][0]["supportClasses"], [20])

    def test_tomato_colour_rule_recovers_missing_id4_from_paper_towels(self) -> None:
        processor = self.make_processor()
        raw = [detection(28, 0.0063, [250, 100, 390, 300])]

        output, debug = processor.process(
            self.tomato_like_frame(), raw, final_conf=0.15
        )

        self.assertEqual([item["id"] for item in output], [TOMATO_ID])
        self.assertEqual(debug["tomatoDecision"], "recover_from_color_geometry")
        self.assertEqual(debug["tomatoColorCandidates"][0]["supportClasses"], [28])

    def test_tomato_colour_rule_never_changes_confident_haribo(self) -> None:
        processor = self.make_processor()
        raw = [detection(20, 0.95, [250, 100, 390, 300])]

        output, debug = processor.process(
            self.tomato_like_frame(), raw, final_conf=0.15
        )

        self.assertEqual([item["id"] for item in output], [20])
        self.assertEqual(debug["tomatoColorCandidates"], [])
        self.assertEqual(
            debug["tomatoColorRejected"][0]["reason"], "support_not_low_confidence"
        )

    def test_tomato_colour_rule_never_overwrites_coca_cola(self) -> None:
        for confuser_id in (14, 26):
            with self.subTest(confuser_id=confuser_id):
                processor = self.make_processor()
                box = [250, 100, 390, 300]
                raw = [
                    detection(20, 0.014, box),
                    detection(confuser_id, 0.86, box),
                ]

                output, debug = processor.process(
                    self.tomato_like_frame(), raw, final_conf=0.15
                )

                self.assertEqual(
                    [item["id"] for item in output], [confuser_id]
                )
                self.assertEqual(
                    debug["tomatoColorRejected"][0]["reason"],
                    "stronger_noncontext_confuser",
                )
                self.assertEqual(
                    debug["tomatoColorRejected"][0]["strongerClass"], confuser_id
                )

    def test_tomato_colour_rule_rejects_tuna_same_object(self) -> None:
        processor = self.make_processor()
        box = [250, 100, 390, 300]
        raw = [
            detection(TUNA_ID, 0.82, box),
            detection(20, 0.014, box),
        ]

        output, debug = processor.process(
            self.tomato_like_frame(), raw, final_conf=0.15
        )

        self.assertEqual([item["id"] for item in output], [TUNA_ID])
        self.assertEqual(
            debug["tomatoColorRejected"][0]["reason"], "same_object_as_tuna"
        )

    def test_tomato_colour_rule_requires_both_red_and_white_evidence(self) -> None:
        processor = self.make_processor()
        box = [250, 100, 390, 300]
        raw = [detection(20, 0.014, box)]

        white_frame = self.frame()
        white_frame[100:300, 250:390] = (245, 245, 245)
        output, debug = processor.process(white_frame, raw, final_conf=0.15)
        self.assertEqual(output, [])
        self.assertEqual(
            debug["tomatoColorRejected"][0]["reason"], "red_label_too_small"
        )

        red_frame = self.frame()
        red_frame[100:300, 250:390] = (0, 0, 220)
        output, debug = processor.process(red_frame, raw, final_conf=0.15)
        self.assertEqual(output, [])
        self.assertEqual(
            debug["tomatoColorRejected"][0]["reason"], "white_body_too_small"
        )


if __name__ == "__main__":
    unittest.main()
