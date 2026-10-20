import unittest

import numpy as np

from opencv_color_predictor import (
    ColorPredictionError,
    TargetColorConfig,
    target_color_mask,
)


def _pixels(*rgb_values):
    """One row of RGB pixels, shaped the way target_color_mask expects."""
    return np.asarray([list(rgb_values)], dtype=np.uint8)


class TargetColorMaskTests(unittest.TestCase):
    """The rule is HSV only: hue decides, saturation and value guard it."""

    def _verdict(self, rgb):
        return bool(target_color_mask(_pixels(rgb))[0, 0])

    def test_bright_red_counts(self):
        self.assertTrue(self._verdict((200, 45, 40)))

    def test_dark_red_counts(self):
        """Regression: an absolute R floor of 40 used to reject this.

        A distant view renders the apple dark. Live measurement on
        conveyor_camera_01 showed that condition alone dropping 35.19 percent
        of the surface even though hue, saturation and value all passed.
        """
        self.assertTrue(self._verdict((32, 7, 6)))

    def test_shading_does_not_change_the_verdict(self):
        """The same skin at three brightnesses must read the same colour."""
        skin = np.asarray((200, 45, 40), dtype=np.float32)
        for scale in (1.0, 0.5, 0.25):
            dimmed = tuple(int(round(channel * scale)) for channel in skin)
            with self.subTest(scale=scale):
                self.assertTrue(self._verdict(dimmed))

    def test_yellow_is_rejected(self):
        self.assertFalse(self._verdict((238, 205, 80)))

    def test_orange_is_rejected(self):
        """Hue 17 is outside the target band even though it reads warm."""
        self.assertFalse(self._verdict((235, 150, 45)))

    def test_washed_out_highlight_is_rejected_by_saturation(self):
        """Hue is meaningless as saturation falls, so the floor carries it."""
        self.assertFalse(self._verdict((250, 244, 245)))

    def test_near_black_is_rejected_by_value(self):
        self.assertFalse(self._verdict((6, 3, 3)))

    def test_hue_wraps_around_zero(self):
        """Red straddles hue 0, so both bands must accept."""
        config = TargetColorConfig()
        low_side = target_color_mask(_pixels((200, 60, 45)), config)[0, 0]
        high_side = target_color_mask(_pixels((200, 45, 60)), config)[0, 0]
        self.assertTrue(bool(low_side))
        self.assertTrue(bool(high_side))

    def test_mask_shape_matches_the_image(self):
        image = np.zeros((7, 5, 3), dtype=np.uint8)
        self.assertEqual(target_color_mask(image).shape, (7, 5))

    def test_non_three_channel_input_is_rejected(self):
        with self.assertRaises(ColorPredictionError):
            target_color_mask(np.zeros((4, 4), dtype=np.uint8))


class ConfigSurfaceTests(unittest.TestCase):
    def test_config_carries_no_rgb_ratio_knobs(self):
        """Guard against the RGB ratio rule creeping back in unmeasured."""
        config = TargetColorConfig()
        for removed in ("min_red", "min_red_over_green", "min_red_over_blue"):
            with self.subTest(field=removed):
                self.assertFalse(hasattr(config, removed))


if __name__ == "__main__":
    unittest.main()
