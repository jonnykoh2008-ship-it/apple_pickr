import unittest

import cv2
import numpy as np

from opencv_damage_grader import (
    DamageDetectionConfig,
    detect_damage,
    segmentation_metrics,
)


class OpenCVDamageGraderTests(unittest.TestCase):
    def test_detects_three_synthetic_damage_appearances(self):
        image = np.zeros((160, 160, 3), dtype=np.uint8)
        apple = np.zeros((160, 160), dtype=np.uint8)
        cv2.circle(apple, (80, 80), 62, 255, -1)
        image[apple > 0] = (190, 40, 35)
        image[45:62, 48:65] = (188, 134, 93)  # bright wound
        image[72:91, 67:88] = (130, 75, 43)  # browning
        image[102:120, 96:115] = (70, 32, 27)  # bruise

        result = detect_damage(image, apple)

        self.assertTrue(result.bright_wound[52, 55])
        self.assertTrue(result.browning[80, 75])
        self.assertTrue(result.bruise[110, 105])
        self.assertFalse(result.combined[80, 25])

    def test_removes_tiny_isolated_candidate(self):
        image = np.full((80, 80, 3), (190, 40, 35), dtype=np.uint8)
        apple = np.ones((80, 80), dtype=np.uint8) * 255
        image[40, 40] = (188, 134, 93)
        config = DamageDetectionConfig(min_component_area_px=3)

        result = detect_damage(image, apple, config)

        self.assertFalse(result.combined.any())

    def test_healthy_yellow_cheek_and_shaded_edge_are_not_damage(self):
        image = np.zeros((160, 160, 3), dtype=np.uint8)
        apple = np.zeros((160, 160), dtype=np.uint8)
        cv2.circle(apple, (80, 80), 62, 255, -1)
        image[apple > 0] = (190, 40, 35)
        cv2.ellipse(image, (80, 80), (54, 54), 0, -70, 70, (225, 180, 45), -1)
        cv2.ellipse(image, (80, 80), (60, 60), 0, 105, 255, (115, 28, 25), 4)

        result = detect_damage(image, apple)

        self.assertFalse(result.combined.any())

    def test_apple_boundary_is_excluded_from_damage_candidates(self):
        image = np.full((100, 100, 3), (190, 40, 35), dtype=np.uint8)
        apple = np.zeros((100, 100), dtype=np.uint8)
        cv2.circle(apple, (50, 50), 35, 255, -1)
        cv2.circle(image, (50, 50), 35, (70, 32, 27), 2)

        result = detect_damage(image, apple)

        self.assertFalse(result.combined.any())

    def test_segmentation_metrics_are_exact(self):
        truth = np.zeros((4, 4), dtype=np.uint8)
        truth[0, :2] = 1
        prediction = np.zeros((4, 4), dtype=np.uint8)
        prediction[0, 0] = 1
        prediction[1, 0] = 1

        metrics = segmentation_metrics(prediction, truth)

        self.assertEqual(metrics.true_positive, 1)
        self.assertEqual(metrics.false_positive, 1)
        self.assertEqual(metrics.false_negative, 1)
        self.assertAlmostEqual(metrics.precision, 0.5)
        self.assertAlmostEqual(metrics.recall, 0.5)
        self.assertAlmostEqual(metrics.iou, 1.0 / 3.0)
        self.assertAlmostEqual(metrics.dice, 0.5)

    def test_rejects_mismatched_apple_mask(self):
        image = np.zeros((20, 20, 3), dtype=np.uint8)
        with self.assertRaisesRegex(ValueError, "apple_mask shape"):
            detect_damage(image, np.zeros((10, 10), dtype=np.uint8))


if __name__ == "__main__":
    unittest.main()
