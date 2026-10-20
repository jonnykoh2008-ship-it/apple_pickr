from __future__ import annotations

import io
import unittest

import numpy as np
from PIL import Image

from depth_geometry import (
    GeometryMeasurementError,
    combine_prediction_with_geometry,
    decode_aligned_depth_m,
    measure_geometry,
)
from inspection_session import InspectionFrame
from predictor import FrameModelPrediction
from quality_rules import Grade, ResultStatus, aggregate_measurement_frames


def png_bytes(array, prefix: bytes = b"") -> bytes:
    buffer = io.BytesIO()
    Image.fromarray(array).save(buffer, format="PNG")
    return prefix + buffer.getvalue()


def make_frame(frame_index: int = 0) -> InspectionFrame:
    width = height = 100
    apple_mask = np.zeros((height, width), dtype=np.uint8)
    apple_mask[20:80, 10:90] = 255
    ignore_mask = np.zeros((height, width), dtype=np.uint8)
    depth_mm = np.zeros((height, width), dtype=np.uint16)
    depth_mm[apple_mask > 0] = 500
    camera_k = (500.0, 0.0, 50.0, 0.0, 500.0, 50.0, 0.0, 0.0, 1.0)
    camera_p = (
        500.0, 0.0, 50.0, 0.0,
        0.0, 500.0, 50.0, 0.0,
        0.0, 0.0, 1.0, 0.0,
    )
    return InspectionFrame(
        inspection_id="geometry-inspection",
        apple_id="geometry-apple",
        frame_index=frame_index,
        total_frames=4,
        image_data=png_bytes(np.zeros((height, width, 3), dtype=np.uint8)),
        image_format="rgb8; jpeg compressed bgr8",
        apple_mask_data=png_bytes(apple_mask),
        apple_mask_format="mono8; png",
        ignore_mask_data=png_bytes(ignore_mask),
        ignore_mask_format="mono8; png",
        depth_data=png_bytes(depth_mm, prefix=b"compressed!!"),
        depth_format="16UC1; compressedDepth png",
        camera_width=width,
        camera_height=height,
        camera_k=camera_k,
        camera_p=camera_p,
        stamp_ns=1_000_000_000 + frame_index,
        frame_id="quality_camera_top_optical_frame",
    )


def make_prediction() -> FrameModelPrediction:
    color_mask = np.zeros((100, 100), dtype=np.float32)
    color_mask[20:80, 10:78] = 1.0
    damage_mask = np.zeros((100, 100), dtype=np.float32)
    damage_mask[30:40, 30:40] = 1.0
    return FrameModelPrediction(
        color_mask=color_mask,
        damage_mask=damage_mask,
        severe_defect=False,
        color_confidence=0.9,
        damage_confidence=0.9,
        severe_confidence=0.9,
    )


class DepthGeometryTest(unittest.TestCase):
    def test_decodes_compressed_depth_payload_and_units(self) -> None:
        depth_m = decode_aligned_depth_m(make_frame())
        self.assertAlmostEqual(float(depth_m[30, 30]), 0.5)
        self.assertEqual(float(depth_m[0, 0]), 0.0)

    def test_measures_depth_diameter_and_confidence(self) -> None:
        result = measure_geometry(make_frame())
        self.assertGreater(result.diameter_mm, 75.0)
        self.assertLess(result.diameter_mm, 82.0)
        self.assertAlmostEqual(result.diameter_confidence, 1.0)

    def test_combines_masks_with_geometry(self) -> None:
        result = combine_prediction_with_geometry(make_frame(), make_prediction())
        self.assertAlmostEqual(result.color_ratio, 0.85, places=2)
        self.assertGreater(result.diameter_mm, 75.0)
        self.assertGreater(result.damage_area_cm2, 0.0)
        self.assertLess(result.damage_area_cm2, 1.0)
        self.assertEqual(result.severe_defect, False)
        self.assertGreaterEqual(result.diameter_confidence, 0.5)

    def test_ignore_mask_is_excluded_from_color_ratio_denominator(self) -> None:
        frame = make_frame()
        ignore_mask = np.zeros((100, 100), dtype=np.uint8)
        ignore_mask[20:80, 78:90] = 255
        frame = InspectionFrame(
            **{**frame.__dict__, "ignore_mask_data": png_bytes(ignore_mask)}
        )
        result = combine_prediction_with_geometry(frame, make_prediction())
        self.assertAlmostEqual(result.color_ratio, 1.0, places=2)


    def test_four_real_geometry_frames_can_reach_valid_high(self) -> None:
        values = [
            combine_prediction_with_geometry(make_frame(index), make_prediction())
            for index in range(4)
        ]
        result = aggregate_measurement_frames(values, range(4))
        self.assertEqual(result.status, ResultStatus.VALID)
        self.assertEqual(result.grade, Grade.HIGH)

    def test_rejects_wrong_depth_contract(self) -> None:
        frame = make_frame()
        with self.assertRaises(GeometryMeasurementError):
            decode_aligned_depth_m(
                InspectionFrame(**{**frame.__dict__, "depth_format": "32FC1"})
            )

    def test_rejects_dimension_mismatch(self) -> None:
        frame = make_frame()
        with self.assertRaises(GeometryMeasurementError):
            measure_geometry(
                InspectionFrame(**{**frame.__dict__, "camera_width": 101})
            )


class DepthContaminationTest(unittest.TestCase):
    """Belt shadow inside the apple mask must not inflate the diameter.

    Observed on the conveyor: identical apples measured 81 mm on clean frames
    and 94-113 mm whenever the mask bled onto the belt behind the apple.
    """

    WIDTH = HEIGHT = 100
    APPLE_DEPTH_MM = 500
    # 벨트는 사과 최근접면보다 지름만큼 뒤에 있다.
    BELT_DEPTH_MM = 580

    def _frame(self, *, with_shadow: bool) -> InspectionFrame:
        apple_mask = np.zeros((self.HEIGHT, self.WIDTH), dtype=np.uint8)
        apple_mask[40:60, 40:60] = 255
        depth_mm = np.zeros((self.HEIGHT, self.WIDTH), dtype=np.uint16)
        depth_mm[apple_mask > 0] = self.APPLE_DEPTH_MM
        if with_shadow:
            # 그림자가 마스크에 붙고, 그 픽셀은 벨트 깊이를 가진다.
            apple_mask[60:80, 40:60] = 255
            depth_mm[60:80, 40:60] = self.BELT_DEPTH_MM

        base = make_frame()
        return InspectionFrame(
            **{
                **base.__dict__,
                "camera_width": self.WIDTH,
                "camera_height": self.HEIGHT,
                "apple_mask_data": png_bytes(apple_mask),
                "ignore_mask_data": png_bytes(
                    np.zeros((self.HEIGHT, self.WIDTH), dtype=np.uint8)
                ),
                "depth_data": png_bytes(depth_mm, prefix=b"compressed!!"),
                "image_data": png_bytes(
                    np.zeros((self.HEIGHT, self.WIDTH, 3), dtype=np.uint8)
                ),
            }
        )

    def test_shadow_pixels_do_not_change_the_diameter(self) -> None:
        clean = measure_geometry(self._frame(with_shadow=False)).diameter_mm
        shadowed = measure_geometry(self._frame(with_shadow=True)).diameter_mm
        self.assertAlmostEqual(clean, shadowed, places=6)

    def test_clean_frame_still_measures(self) -> None:
        self.assertGreater(measure_geometry(self._frame(with_shadow=False)).diameter_mm, 0.0)


if __name__ == "__main__":
    unittest.main()
