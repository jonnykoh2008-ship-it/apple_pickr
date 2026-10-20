import unittest
from types import SimpleNamespace

import cv2
import numpy as np

from opencv_color_live_node import (
    ColorObservation,
    ColorMeasurementConfig,
    ObservedColorAccumulator,
    ApproximateThreeViewSynchronizer,
    KNOWN_CAMERA_VIEWS,
    TemporalColorConfig,
    TemporalColorTracker,
    appearance_descriptor,
    classify_three_view_detection,
    classify_color_ratio,
    combine_three_view_measurements,
    combine_three_view_result_sets,
    make_color_result_payload,
    measure_target_red,
    measure_visible_damage,
    process_color_frame,
)


class ColorMeasurementTests(unittest.TestCase):
    @staticmethod
    def circular_mask(shape=(180, 240), center=(120, 90), radius=45):
        mask = np.zeros(shape, dtype=np.uint8)
        cv2.circle(mask, center, radius, 255, thickness=-1)
        return mask

    def test_all_red_surface_is_near_one(self) -> None:
        image = np.full((180, 240, 3), 180, dtype=np.uint8)
        mask = self.circular_mask()
        image[mask > 0] = (25, 35, 205)

        masks, ratio = measure_target_red(image, mask)

        self.assertGreater(ratio, 0.99)
        self.assertTrue(masks.target_red.any())
        self.assertTrue(masks.valid_surface.any())

    def test_yellow_surface_is_zero(self) -> None:
        image = np.full((180, 240, 3), 180, dtype=np.uint8)
        mask = self.circular_mask()
        image[mask > 0] = (25, 205, 230)

        _, ratio = measure_target_red(image, mask)

        self.assertEqual(ratio, 0.0)

    def test_half_red_half_yellow_is_near_half(self) -> None:
        image = np.full((180, 240, 3), 180, dtype=np.uint8)
        mask = self.circular_mask()
        image[mask > 0] = (25, 205, 230)
        left_half = mask.copy()
        left_half[:, 120:] = 0
        image[left_half > 0] = (25, 35, 205)

        _, ratio = measure_target_red(image, mask)

        self.assertAlmostEqual(ratio, 0.5, delta=0.03)

    def test_white_specular_patch_is_excluded_from_denominator(self) -> None:
        image = np.full((180, 240, 3), 180, dtype=np.uint8)
        mask = self.circular_mask()
        image[mask > 0] = (25, 35, 205)
        cv2.circle(image, (120, 90), 14, (255, 255, 255), thickness=-1)

        masks, ratio = measure_target_red(image, mask)

        self.assertGreater(ratio, 0.99)
        self.assertTrue(masks.ignored[90, 120])

    def test_visible_brown_patch_produces_damage_ratio(self) -> None:
        image = np.full((180, 240, 3), 180, dtype=np.uint8)
        mask = self.circular_mask()
        image[mask > 0] = (25, 35, 205)
        cv2.circle(image, (120, 90), 12, (35, 80, 140), thickness=-1)
        color_masks, _ = measure_target_red(image, mask)

        damage, ratio = measure_visible_damage(
            image,
            mask,
            color_masks.valid_surface,
        )

        self.assertGreater(ratio, 0.05)
        self.assertTrue(damage.combined.any())

    def test_neutral_dark_shadow_is_not_visible_color_damage(self) -> None:
        image = np.full((180, 240, 3), 180, dtype=np.uint8)
        mask = self.circular_mask()
        image[mask > 0] = (25, 35, 205)
        cv2.circle(image, (120, 90), 12, (70, 70, 70), thickness=-1)
        color_masks, _ = measure_target_red(image, mask)

        damage, ratio = measure_visible_damage(
            image,
            mask,
            color_masks.valid_surface,
        )

        self.assertEqual(ratio, 0.0)
        self.assertFalse(damage.combined.any())

    def test_process_frame_detects_yellow_and_mixed_apples(self) -> None:
        image = np.full((240, 360, 3), 180, dtype=np.uint8)
        cv2.circle(image, (100, 120), 52, (25, 205, 230), thickness=-1)
        cv2.circle(image, (260, 120), 52, (25, 35, 205), thickness=-1)
        cv2.ellipse(image, (260, 120), (52, 52), 0, -90, 90, (25, 205, 230), -1)

        overlay, results = process_color_frame(image)

        self.assertEqual(overlay.shape, image.shape)
        self.assertEqual(len(results), 2)
        self.assertLess(results[0].color_ratio, 0.01)
        self.assertAlmostEqual(results[1].color_ratio, 0.5, delta=0.08)

    def test_process_frame_reports_visible_damage_on_detected_apple(self) -> None:
        image = np.full((180, 240, 3), 180, dtype=np.uint8)
        cv2.circle(image, (120, 90), 45, (25, 35, 205), thickness=-1)
        cv2.circle(image, (120, 90), 12, (35, 80, 140), thickness=-1)

        overlay, results = process_color_frame(image)

        self.assertEqual(overlay.shape, image.shape)
        self.assertEqual(len(results), 1)
        self.assertGreater(results[0].damage_ratio, 0.05)

    def test_rgb_fallback_recovers_from_flat_depth(self) -> None:
        image = np.full((180, 240, 3), 180, dtype=np.uint8)
        cv2.circle(image, (120, 90), 40, (25, 35, 205), thickness=-1)
        flat_depth = np.full((180, 240), 1000, dtype=np.uint16)

        _, results = process_color_frame(image, flat_depth)

        self.assertEqual(len(results), 1)
        self.assertGreater(results[0].color_ratio, 0.99)

    def test_empty_mask_returns_zero(self) -> None:
        image = np.zeros((20, 20, 3), dtype=np.uint8)
        masks, ratio = measure_target_red(
            image,
            np.zeros((20, 20), dtype=np.uint8),
        )
        self.assertEqual(ratio, 0.0)
        self.assertFalse(masks.valid_surface.any())

    def test_invalid_mask_shape_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "apple_mask"):
            measure_target_red(
                np.zeros((20, 20, 3), dtype=np.uint8),
                np.zeros((10, 10), dtype=np.uint8),
            )

    def test_invalid_edge_exclusion_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "edge_exclusion_ratio"):
            ColorMeasurementConfig(edge_exclusion_ratio=1.0)

    def test_three_view_ratio_is_weighted_by_valid_surface_pixels(self) -> None:
        def result(valid_pixels: int, red_pixels: int, damage_pixels: int = 0):
            width = valid_pixels
            valid = np.ones((1, width), dtype=bool)
            red = np.zeros((1, width), dtype=bool)
            red[:, :red_pixels] = True
            damage = np.zeros((1, width), dtype=bool)
            damage[:, :damage_pixels] = True
            ignored = np.zeros((1, width), dtype=bool)
            apple = SimpleNamespace()
            return SimpleNamespace(
                apple=apple,
                masks=SimpleNamespace(
                    valid_surface=valid,
                    target_red=red,
                    ignored=ignored,
                ),
                damage=SimpleNamespace(combined=damage),
                color_ratio=red_pixels / valid_pixels,
                damage_ratio=damage_pixels / valid_pixels,
            )

        measurement = combine_three_view_measurements(
            (result(100, 100, 10), result(300, 0, 30), result(100, 50, 10)),
            views=KNOWN_CAMERA_VIEWS,
        )

        self.assertEqual(measurement.target_red_pixels, 150)
        self.assertEqual(measurement.damage_pixels, 50)
        self.assertEqual(measurement.valid_surface_pixels, 500)
        self.assertEqual(measurement.views_used, 3)
        self.assertAlmostEqual(measurement.color_ratio, 0.3)
        self.assertAlmostEqual(measurement.damage_ratio, 0.1)

    def test_three_view_result_sets_require_all_views_and_equal_counts(self) -> None:
        image = np.full((180, 240, 3), 180, dtype=np.uint8)
        cv2.circle(image, (120, 90), 40, (25, 35, 205), thickness=-1)
        _, results = process_color_frame(image)

        combined = combine_three_view_result_sets(
            {"top": results, "left": results, "right": results},
            views=KNOWN_CAMERA_VIEWS,
        )

        self.assertEqual(len(combined), 1)
        self.assertGreater(combined[0].color_ratio, 0.99)
        with self.assertRaisesRegex(ValueError, "exactly one apple"):
            combine_three_view_result_sets(
                {"top": results, "left": (), "right": results}
            )

    def test_partial_three_view_detection_waits_until_all_views_are_ready(self) -> None:
        placeholder = (SimpleNamespace(),)

        waiting, counts = classify_three_view_detection(
            {"top": (), "left": placeholder, "right": ()},
            views=KNOWN_CAMERA_VIEWS,
        )
        ready, _ = classify_three_view_detection(
            {"top": placeholder, "left": placeholder, "right": placeholder},
            views=KNOWN_CAMERA_VIEWS,
        )
        recheck, _ = classify_three_view_detection(
            {
                "top": (SimpleNamespace(), SimpleNamespace()),
                "left": placeholder,
                "right": placeholder,
            }
        )

        self.assertEqual(waiting, "WAITING")
        self.assertEqual(counts, {"top": 0, "left": 1, "right": 0})
        self.assertEqual(ready, "READY")
        self.assertEqual(recheck, "RECHECK")

    def test_approximate_three_view_synchronizer_accepts_20ms_offset(self) -> None:
        def message(stamp_ns: int, value: str):
            stamp = SimpleNamespace(
                sec=stamp_ns // 1_000_000_000,
                nanosec=stamp_ns % 1_000_000_000,
            )
            return SimpleNamespace(
                header=SimpleNamespace(stamp=stamp),
                value=value,
            )

        synchronizer = ApproximateThreeViewSynchronizer(
            queue_size=3,
            tolerance_ms=20.0,
            views=KNOWN_CAMERA_VIEWS,
        )
        top = (message(1_000_000_000, "top-rgb"), message(1_000_000_000, "top-depth"))
        right = (
            message(1_000_000_000, "right-rgb"),
            message(1_000_000_000, "right-depth"),
        )
        left = (
            message(1_016_700_000, "left-rgb"),
            message(1_016_700_000, "left-depth"),
        )

        self.assertIsNone(synchronizer.add("top", top))
        self.assertIsNone(synchronizer.add("right", right))
        synchronized = synchronizer.add("left", left)

        self.assertEqual(
            tuple(pair[0].value for pair in synchronized),
            ("top-rgb", "left-rgb", "right-rgb"),
        )

    def test_approximate_three_view_synchronizer_rejects_over_20ms_offset(self) -> None:
        def message(stamp_ns: int):
            stamp = SimpleNamespace(
                sec=stamp_ns // 1_000_000_000,
                nanosec=stamp_ns % 1_000_000_000,
            )
            return SimpleNamespace(header=SimpleNamespace(stamp=stamp))

        synchronizer = ApproximateThreeViewSynchronizer(
            queue_size=3,
            tolerance_ms=20.0,
            views=KNOWN_CAMERA_VIEWS,
        )
        top = (message(1_000_000_000), message(1_000_000_000))
        right = (message(1_000_000_000), message(1_000_000_000))
        late_left = (message(1_020_000_001), message(1_020_000_001))

        self.assertIsNone(synchronizer.add("top", top))
        self.assertIsNone(synchronizer.add("right", right))
        self.assertIsNone(synchronizer.add("left", late_left))

    def test_accumulator_accepts_every_valid_frame(self) -> None:
        observations = ObservedColorAccumulator()

        for frame_index in range(1, 13):
            self.assertTrue(observations.add(ColorObservation(frame_index, 0.5)))

        summary = observations.summary()

        self.assertTrue(summary.ready)
        self.assertEqual(summary.frame_count, 12)
        self.assertAlmostEqual(summary.observed_surface_ratio, 0.5)

    def test_observed_surface_ratio_uses_median_to_reject_outliers(self) -> None:
        observations = ObservedColorAccumulator()
        for frame_index, ratio in enumerate((0.50, 0.52, 0.51, 0.0, 1.0), start=1):
            observations.add(ColorObservation(frame_index, ratio))

        summary = observations.summary()

        self.assertAlmostEqual(summary.observed_surface_ratio, 0.51)
        self.assertEqual(summary.grade, "MEDIUM")

    def test_damage_ratio_uses_median_across_three_view_sets(self) -> None:
        observations = ObservedColorAccumulator()
        damage_ratios = (0.02, 0.03, 0.025, 0.80, 0.0)
        for frame_index, damage_ratio in enumerate(damage_ratios, start=1):
            observations.add(ColorObservation(frame_index, 0.5, damage_ratio))

        summary = observations.summary()

        self.assertEqual(summary.frame_count, 5)
        self.assertAlmostEqual(summary.damage_ratio, 0.025)

    def test_tracker_keeps_id_and_counts_stalled_view_frames(self) -> None:
        image = np.full((180, 240, 3), 180, dtype=np.uint8)
        cv2.circle(image, (120, 90), 40, (25, 35, 205), thickness=-1)
        _, results = process_color_frame(image)
        tracker = TemporalColorTracker(
            TemporalColorConfig()
        )

        first = tracker.update(image, results, 1)
        second = tracker.update(image, results, 2)

        self.assertEqual(first[0][0], second[0][0])
        self.assertTrue(first[0][2])
        self.assertTrue(second[0][2])
        self.assertEqual(second[0][1].frame_count, 2)

    def test_approved_color_grade_boundaries(self) -> None:
        self.assertEqual(classify_color_ratio(0.0), "LOW")
        self.assertEqual(classify_color_ratio(0.399999), "LOW")
        self.assertEqual(classify_color_ratio(0.40), "MEDIUM")
        self.assertEqual(classify_color_ratio(0.599999), "MEDIUM")
        self.assertEqual(classify_color_ratio(0.60), "HIGH")
        self.assertEqual(classify_color_ratio(1.0), "HIGH")

    def test_later_frames_continue_updating_result(self) -> None:
        observations = ObservedColorAccumulator()
        for index in range(6):
            observations.add(ColorObservation(index + 1, 0.3))

        self.assertEqual(observations.summary().grade, "LOW")
        for index in range(6, 16):
            observations.add(ColorObservation(index + 1, 0.8))

        final_summary = observations.summary()
        self.assertEqual(final_summary.frame_count, 16)
        self.assertAlmostEqual(final_summary.observed_surface_ratio, 0.8)
        self.assertEqual(final_summary.grade, "HIGH")

    def test_expired_track_emits_final_observed_surface_result(self) -> None:
        image = np.full((180, 240, 3), 180, dtype=np.uint8)
        cv2.circle(image, (120, 90), 40, (25, 35, 205), thickness=-1)
        _, results = process_color_frame(image)
        tracker = TemporalColorTracker(
            TemporalColorConfig(expire_after_missing_frames=1)
        )

        tracker.update(image, results, 1)
        tracker.update(image, (), 2)
        tracker.update(image, (), 3)
        completed = tracker.drain_completed()

        self.assertEqual(len(completed), 1)
        self.assertEqual(completed[0][1].frame_count, 1)
        self.assertEqual(completed[0][1].grade, "HIGH")

    def test_completed_summary_converts_to_quality_result_payload(self) -> None:
        observations = ObservedColorAccumulator()
        for frame_index, ratio in enumerate((0.58, 0.62, 0.61), start=1):
            observations.add(ColorObservation(frame_index, ratio))

        payload = make_color_result_payload(7, observations.summary())

        self.assertEqual(payload.inspection_id, "opencv-color-7")
        self.assertEqual(payload.apple_id, "apple-7")
        self.assertEqual(payload.grade, "HIGH")
        self.assertAlmostEqual(payload.color_ratio, 0.61)
        self.assertEqual(payload.frames_used, 3)
        self.assertEqual(payload.frame_indices, (0, 1, 2))
        self.assertEqual(payload.status, "VALID")

    def test_appearance_descriptor_is_position_normalized(self) -> None:
        first = np.full((180, 240, 3), 180, dtype=np.uint8)
        second = first.copy()
        cv2.circle(first, (80, 90), 35, (25, 35, 205), thickness=-1)
        cv2.circle(second, (160, 90), 35, (25, 35, 205), thickness=-1)
        _, first_results = process_color_frame(first)
        _, second_results = process_color_frame(second)

        first_descriptor = appearance_descriptor(first, first_results[0])
        second_descriptor = appearance_descriptor(second, second_results[0])

        self.assertLess(
            float(np.mean(np.abs(first_descriptor - second_descriptor))),
            0.02,
        )


if __name__ == "__main__":
    unittest.main()
