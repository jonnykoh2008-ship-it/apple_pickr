from __future__ import annotations

import unittest

from quality_rules import (
    AppleMeasurements,
    FrameMeasurements,
    Grade,
    ResultStatus,
    aggregate_frames,
    aggregate_measurement_frames,
    grade_color_ratio,
    grade_diameter_mm,
    grade_measurements,
)


class SizeRuleBoundaryTest(unittest.TestCase):
    def test_low_is_strictly_below_60_mm(self) -> None:
        self.assertEqual(grade_diameter_mm(59.999), Grade.LOW)
        self.assertEqual(grade_diameter_mm(60.0), Grade.MEDIUM)

    def test_high_includes_75_mm(self) -> None:
        self.assertEqual(grade_diameter_mm(74.999), Grade.MEDIUM)
        self.assertEqual(grade_diameter_mm(75.0), Grade.HIGH)

    def test_current_mvp_ignores_deferred_quality_fields(self) -> None:
        result = grade_measurements(
            AppleMeasurements(
                color_ratio=0.0,
                diameter_mm=80.0,
                damage_area_cm2=99.0,
                severe_defect=True,
            )
        )
        self.assertEqual(result, Grade.HIGH)

    def test_negative_diameter_is_invalid(self) -> None:
        with self.assertRaises(ValueError):
            grade_diameter_mm(-0.1)


class SizeAggregationTest(unittest.TestCase):
    @staticmethod
    def frame(diameter: float, confidence: float = 0.9) -> FrameMeasurements:
        return FrameMeasurements(
            diameter_mm=diameter,
            diameter_confidence=confidence,
        )

    def test_diameter_only_measurements_produce_valid_result(self) -> None:
        result = aggregate_measurement_frames(
            [self.frame(75.0)],
            [3],
        )
        self.assertEqual(result.status, ResultStatus.VALID)
        self.assertEqual(result.grade, Grade.HIGH)
        self.assertEqual(result.frames_used, (3,))
        self.assertEqual(result.confidence, 0.9)
        assert result.measurements is not None
        self.assertEqual(result.measurements.diameter_mm, 75.0)
        self.assertIsNone(result.measurements.color_ratio)
        self.assertIsNone(result.measurements.damage_area_cm2)

    def test_even_frame_count_uses_statistical_median(self) -> None:
        result = aggregate_measurement_frames(
            [self.frame(value) for value in (59.0, 59.0, 75.0, 75.0)],
            range(4),
            min_valid_views=4,
        )
        self.assertEqual(result.status, ResultStatus.VALID)
        assert result.measurements is not None
        self.assertEqual(result.measurements.diameter_mm, 67.0)
        self.assertEqual(result.grade, Grade.MEDIUM)

    def test_legacy_aggregate_also_uses_statistical_median(self) -> None:
        values = [
            AppleMeasurements(diameter_mm=value)
            for value in (59.0, 59.0, 75.0, 75.0)
        ]
        result = aggregate_frames(values, range(4), min_valid_views=4)
        self.assertEqual(result.status, ResultStatus.VALID)
        assert result.measurements is not None
        self.assertEqual(result.measurements.diameter_mm, 67.0)

    def test_low_confidence_reduces_valid_view_count(self) -> None:
        frames = [self.frame(80.0) for _ in range(4)]
        frames[0] = self.frame(80.0, confidence=0.49)
        result = aggregate_measurement_frames(
            frames,
            range(4),
            confidence_threshold=0.5,
            min_valid_views=4,
        )
        self.assertEqual(result.status, ResultStatus.INSUFFICIENT_VIEWS)

    def test_missing_diameter_is_unclassified(self) -> None:
        result = aggregate_measurement_frames(
            [FrameMeasurements(color_ratio=0.9, color_confidence=0.9)],
            [0],
        )
        self.assertEqual(result.status, ResultStatus.UNCLASSIFIED)
        self.assertIsNone(result.grade)


class ColorRuleBoundaryTest(unittest.TestCase):
    """Approved provisional boundaries: 60% and 80%.

    Measured on the conveyor: a well-coloured apple read 84-87% and a
    yellow-tinged one 54-56%, so both clear their boundary by 4-7 points.
    """

    def test_high_includes_its_lower_boundary(self) -> None:
        self.assertEqual(grade_color_ratio(1.0), Grade.HIGH)
        self.assertEqual(grade_color_ratio(0.80), Grade.HIGH)

    def test_medium_spans_sixty_to_eighty(self) -> None:
        self.assertEqual(grade_color_ratio(0.799), Grade.MEDIUM)
        self.assertEqual(grade_color_ratio(0.60), Grade.MEDIUM)

    def test_low_is_below_sixty(self) -> None:
        self.assertEqual(grade_color_ratio(0.599), Grade.LOW)
        self.assertEqual(grade_color_ratio(0.0), Grade.LOW)

    def test_measured_apples_land_on_the_expected_grades(self) -> None:
        for ratio in (0.84, 0.87):
            self.assertEqual(grade_color_ratio(ratio), Grade.HIGH)
        for ratio in (0.54, 0.56):
            self.assertEqual(grade_color_ratio(ratio), Grade.LOW)

    def test_ratio_outside_zero_to_one_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            grade_color_ratio(1.5)
        with self.assertRaises(ValueError):
            grade_color_ratio(-0.1)


class ColorGradingTest(unittest.TestCase):
    @staticmethod
    def _frames(colours):
        return [
            FrameMeasurements(
                diameter_mm=80.0,
                diameter_confidence=0.9,
                color_ratio=value,
                color_confidence=0.9,
            )
            for value in colours
        ]

    def test_colour_grading_uses_the_mean_of_the_views(self) -> None:
        result = aggregate_measurement_frames(
            self._frames([0.90, 0.84, 0.87]),
            (0, 1, 2),
            min_valid_views=3,
            grade_by="color",
        )
        self.assertEqual(result.status, ResultStatus.VALID)
        self.assertAlmostEqual(result.measurements.color_ratio, 0.87)
        self.assertEqual(result.grade, Grade.HIGH)

    def test_yellow_tinged_apple_grades_low(self) -> None:
        result = aggregate_measurement_frames(
            self._frames([0.54, 0.56, 0.55]),
            (0, 1, 2),
            min_valid_views=3,
            grade_by="color",
        )
        self.assertEqual(result.grade, Grade.LOW)

    def test_colour_grading_without_colour_is_unclassified(self) -> None:
        result = aggregate_measurement_frames(
            self._frames([None, None, None]),
            (0, 1, 2),
            min_valid_views=3,
            grade_by="color",
        )
        self.assertEqual(result.status, ResultStatus.UNCLASSIFIED)
        self.assertIsNone(result.grade)


class MeasurementCarryThroughTest(unittest.TestCase):
    """Every measured field must survive aggregation.

    Regression: colour ratio reached FrameMeasurements but the aggregate built
    AppleMeasurements without it, so /quality/results reported NaN even though
    the predictor and geometry were both working.
    """

    @staticmethod
    def _frames():
        return [
            FrameMeasurements(
                diameter_mm=diameter,
                diameter_confidence=0.9,
                color_ratio=colour,
                color_confidence=0.9,
            )
            for diameter, colour in (
                (79.0, 0.90),
                (81.0, 0.70),
                (80.0, 0.80),
            )
        ]

    def test_diameter_uses_the_median(self) -> None:
        result = aggregate_measurement_frames(
            self._frames(), (0, 1, 2), min_valid_views=3
        )
        self.assertAlmostEqual(result.measurements.diameter_mm, 80.0)

    def test_colour_uses_the_mean(self) -> None:
        result = aggregate_measurement_frames(
            self._frames(), (0, 1, 2), min_valid_views=3
        )
        self.assertAlmostEqual(result.measurements.color_ratio, 0.80)

    def test_colour_stays_none_when_no_frame_measured_it(self) -> None:
        frames = [
            FrameMeasurements(diameter_mm=80.0, diameter_confidence=0.9)
            for _ in range(3)
        ]
        result = aggregate_measurement_frames(
            frames, (0, 1, 2), min_valid_views=3
        )
        self.assertIsNone(result.measurements.color_ratio)


class ColourAreaWeightingTest(unittest.TestCase):
    """Colour must be pooled by surface area, not averaged across views.

    Views see different amounts of apple, so a plain mean lets a small distant
    view count as much as a large near one. Live apple1: views of 0.188 / 0.892
    / 0.708 average to 0.596 but pool to 0.527.
    """

    @staticmethod
    def _frames():
        # Same three ratios as the live apple1 capture, with its pixel counts.
        return [
            FrameMeasurements(
                diameter_mm=80.0,
                diameter_confidence=0.9,
                color_ratio=coloured / area,
                color_pixels=coloured,
                measurable_pixels=area,
                color_confidence=0.9,
            )
            for coloured, area in ((1002, 5323), (1972, 2212), (3892, 5501))
        ]

    def test_pooled_ratio_is_used_when_pixel_counts_exist(self) -> None:
        result = aggregate_measurement_frames(
            self._frames(), (0, 1, 2), min_valid_views=3
        )
        expected = (1002 + 1972 + 3892) / (5323 + 2212 + 5501)
        self.assertAlmostEqual(result.measurements.color_ratio, expected)

    def test_pooled_ratio_differs_from_the_plain_mean(self) -> None:
        frames = self._frames()
        mean = sum(f.color_ratio for f in frames) / len(frames)
        result = aggregate_measurement_frames(frames, (0, 1, 2), min_valid_views=3)
        self.assertAlmostEqual(mean, 0.5957, places=3)
        self.assertAlmostEqual(result.measurements.color_ratio, 0.5267, places=3)

    def test_falls_back_to_the_mean_without_pixel_counts(self) -> None:
        """Synthetic labels carry a ratio and no pixels; they must still work."""
        frames = [
            FrameMeasurements(
                diameter_mm=80.0, diameter_confidence=0.9,
                color_ratio=value, color_confidence=0.9,
            )
            for value in (0.90, 0.70, 0.80)
        ]
        result = aggregate_measurement_frames(frames, (0, 1, 2), min_valid_views=3)
        self.assertAlmostEqual(result.measurements.color_ratio, 0.80)

    def test_zero_measurable_area_does_not_divide_by_zero(self) -> None:
        frames = [
            FrameMeasurements(
                diameter_mm=80.0, diameter_confidence=0.9,
                color_ratio=0.5, color_pixels=0, measurable_pixels=0,
                color_confidence=0.9,
            )
            for _ in range(3)
        ]
        result = aggregate_measurement_frames(frames, (0, 1, 2), min_valid_views=3)
        self.assertAlmostEqual(result.measurements.color_ratio, 0.5)


if __name__ == "__main__":
    unittest.main()
