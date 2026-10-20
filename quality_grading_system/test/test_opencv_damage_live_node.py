import unittest

import cv2
import numpy as np

from opencv_damage_live_node import (
    AppleInstanceConfig,
    detect_apple_instances,
    process_frame,
    remove_rubber_reflections_by_depth,
)


class AppleInstanceDetectionTests(unittest.TestCase):
    @staticmethod
    def two_apple_image() -> np.ndarray:
        image = np.full((240, 360, 3), 180, dtype=np.uint8)
        cv2.circle(image, (100, 120), 58, (25, 35, 200), thickness=-1)
        cv2.circle(image, (260, 120), 52, (30, 45, 190), thickness=-1)
        return image

    def test_detects_two_separated_apples(self) -> None:
        apples = detect_apple_instances(self.two_apple_image())
        self.assertEqual(len(apples), 2)
        centers = sorted(apple.center[0] for apple in apples)
        self.assertAlmostEqual(centers[0], 100.0, delta=3.0)
        self.assertAlmostEqual(centers[1], 260.0, delta=3.0)
        self.assertTrue(all(apple.mask.dtype == np.uint8 for apple in apples))

    def test_max_apples_limits_results(self) -> None:
        config = AppleInstanceConfig(max_apples=1)
        apples = detect_apple_instances(self.two_apple_image(), config)
        self.assertEqual(len(apples), 1)

    def test_small_saturated_noise_is_rejected(self) -> None:
        image = self.two_apple_image()
        cv2.circle(image, (15, 15), 3, (0, 255, 0), thickness=-1)
        apples = detect_apple_instances(image)
        self.assertEqual(len(apples), 2)

    def test_long_red_conveyor_parts_are_not_apples(self) -> None:
        image = np.full((240, 360, 3), 180, dtype=np.uint8)
        cv2.rectangle(image, (20, 45), (340, 62), (25, 35, 200), thickness=-1)
        cv2.rectangle(image, (30, 175), (330, 190), (30, 40, 190), thickness=-1)
        cv2.circle(image, (170, 120), 18, (25, 35, 205), thickness=-1)

        apples = detect_apple_instances(image)

        self.assertEqual(len(apples), 1)
        self.assertAlmostEqual(apples[0].center[0], 170.0, delta=3.0)
        self.assertAlmostEqual(apples[0].center[1], 120.0, delta=3.0)

    def test_non_red_round_object_is_not_an_apple(self) -> None:
        image = np.full((180, 240, 3), 180, dtype=np.uint8)
        cv2.circle(image, (70, 90), 35, (20, 210, 20), thickness=-1)
        cv2.circle(image, (170, 90), 35, (25, 35, 200), thickness=-1)

        apples = detect_apple_instances(image)

        self.assertEqual(len(apples), 1)
        self.assertAlmostEqual(apples[0].center[0], 170.0, delta=3.0)

    def test_detects_yellow_apple(self) -> None:
        image = np.full((180, 240, 3), 180, dtype=np.uint8)
        cv2.circle(image, (120, 90), 35, (25, 205, 230), thickness=-1)

        apples = detect_apple_instances(image)

        self.assertEqual(len(apples), 1)
        self.assertAlmostEqual(apples[0].center[0], 120.0, delta=3.0)

    def test_red_and_yellow_skin_is_one_apple(self) -> None:
        image = np.full((180, 240, 3), 180, dtype=np.uint8)
        cv2.circle(image, (120, 90), 40, (25, 35, 205), thickness=-1)
        cv2.ellipse(image, (120, 90), (40, 40), 0, -90, 90, (25, 205, 230), -1)

        apples = detect_apple_instances(image)

        self.assertEqual(len(apples), 1)
        self.assertAlmostEqual(apples[0].center[0], 120.0, delta=3.0)

    def test_depth_removes_connected_rubber_reflection(self) -> None:
        image = np.full((240, 360, 3), 180, dtype=np.uint8)
        cv2.circle(image, (180, 90), 36, (25, 35, 205), thickness=-1)
        cv2.rectangle(image, (174, 124), (186, 215), (28, 38, 185), -1)
        depth_mm = np.full((240, 360), 1000, dtype=np.uint16)
        cv2.circle(depth_mm, (180, 90), 36, 900, thickness=-1)

        apples = detect_apple_instances(image, depth_mm=depth_mm)

        self.assertEqual(len(apples), 1)
        x, y, width, height = apples[0].bounding_box
        self.assertAlmostEqual(x + width * 0.5, 180.0, delta=3.0)
        self.assertLessEqual(height, 78)

    def test_depth_filter_rejects_shape_mismatch(self) -> None:
        candidate = np.zeros((20, 20), dtype=np.uint8)
        with self.assertRaisesRegex(ValueError, "depth_mm shape"):
            remove_rubber_reflections_by_depth(
                candidate,
                np.zeros((10, 10), dtype=np.uint16),
                8,
            )

    def test_process_frame_runs_damage_detection_per_apple(self) -> None:
        image = self.two_apple_image()
        image[105:118, 88:102] = (70, 120, 188)
        image[128:142, 250:264] = (43, 75, 130)
        overlay, results = process_frame(image)
        self.assertEqual(overlay.shape, image.shape)
        self.assertEqual(len(results), 2)
        self.assertTrue(any(result.damage.combined.any() for result in results))

    def test_rejects_non_bgr_input(self) -> None:
        with self.assertRaisesRegex(ValueError, "uint8 HxWx3"):
            detect_apple_instances(np.zeros((20, 20), dtype=np.uint8))


if __name__ == "__main__":
    unittest.main()
