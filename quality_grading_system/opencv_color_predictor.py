"""Measure how much of the apple surface reached the target red colour.

The detector that was tried as a damage head actually separated apples by how
red their surface is: an orange/yellow-topped apple read 3.6-4.8 cm2 while a
red one read 0.8-1.2 cm2. That is a colouring difference, not a defect, so it
belongs in the colour ratio rather than the damage area.

The target-red definition is the one the synthetic generator already uses for
``red_target`` in ``a1_quality_template_20frame_script_editor.md``, so the
live measurement and the generated data agree on what "red" means.

`depth_geometry` turns `color_mask` into `color_ratio` by dividing the target
pixels by the apple pixels that carry valid depth.
"""

from __future__ import annotations

import io
from dataclasses import dataclass

from depth_geometry import decode_apple_mask
from inspection_session import InspectionFrame
from predictor import FrameModelPrediction


class ColorPredictionError(RuntimeError):
    """Raised when a frame cannot be turned into a colour prediction."""


@dataclass(frozen=True)
class TargetColorConfig:
    """Target red in OpenCV channel scales (H 0-179, S/V 0-255).

    The rule is HSV only. A sphere shows the same skin at very different
    brightnesses, and hue is the component that divides brightness out, so a
    shaded face and a lit face are judged the same colour. Earlier revisions
    also required RGB channel ratios (R >= 1.45G, R >= 1.60B) and an absolute
    R floor; measurement showed those cost the red apple 1.6-4.9 percent of its
    surface per view while rejecting under 1 percent of the yellow apple, so
    they only eroded the score without adding separation.

    Red wraps around hue 0, so two ranges are needed.

    The saturation and value floors are what make hue trustworthy: hue is
    meaningless as saturation approaches zero, so ``min_saturation`` is what
    keeps a washed-out specular highlight out, and ``min_value`` guards the
    dark end. ``ignore_mask`` removes the extremes of both before this runs.
    """

    hue_low_max: int = 10
    hue_high_min: int = 170

    # 생성기의 red_target 스펙(채도 110, 밝기 60~240)은 재질에 어떤 색을 칠할지를
    # 정한 것이지 렌더된 픽셀을 분류하는 기준이 아니다. 그대로 쓰면 곡면 사과의
    # 그늘진 면이 통째로 빠져 완전히 빨간 사과가 34%로 측정된다.
    min_saturation: int = 40
    min_value: int = 20
    max_value: int = 255


def target_color_mask(rgb, config: TargetColorConfig = TargetColorConfig()):
    """Boolean mask of pixels that reached the target red colour."""
    import cv2
    import numpy as np

    image = np.asarray(rgb)
    if image.ndim != 3 or image.shape[2] != 3:
        raise ColorPredictionError("rgb must have shape (height, width, 3)")

    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    low = cv2.inRange(
        hsv,
        np.array((0, config.min_saturation, config.min_value), dtype=np.uint8),
        np.array((config.hue_low_max, 255, config.max_value), dtype=np.uint8),
    )
    high = cv2.inRange(
        hsv,
        np.array(
            (config.hue_high_min, config.min_saturation, config.min_value),
            dtype=np.uint8,
        ),
        np.array((179, 255, config.max_value), dtype=np.uint8),
    )
    return cv2.bitwise_or(low, high) > 0


def decode_rgb(frame: InspectionFrame):
    """Decode the compressed RGB payload into an HxWx3 uint8 RGB array."""
    try:
        import numpy as np
        from PIL import Image
    except ImportError as exc:  # pragma: no cover - environment guard
        raise ColorPredictionError(
            "NumPy and Pillow are required to decode the inspection image"
        ) from exc

    try:
        with Image.open(io.BytesIO(frame.image_data)) as image:
            return np.asarray(image.convert("RGB"), dtype=np.uint8)
    except Exception as exc:
        raise ColorPredictionError("cannot decode compressed RGB image") from exc


@dataclass
class OpenCvColorPredictor:
    """Per-frame target-colour mask from HSV thresholds, no learned model."""

    config: TargetColorConfig = TargetColorConfig()

    def predict(self, frame: InspectionFrame) -> FrameModelPrediction:
        import numpy as np

        rgb = decode_rgb(frame)
        apple_mask = decode_apple_mask(frame)
        if apple_mask.shape != rgb.shape[:2]:
            raise ColorPredictionError(
                "apple_mask and RGB dimensions must match; "
                f"mask={apple_mask.shape} rgb={rgb.shape[:2]}"
            )

        coloured = target_color_mask(rgb, self.config) & apple_mask
        return FrameModelPrediction(
            color_mask=coloured.astype(np.float32),
            damage_mask=None,
            severe_defect=None,
            # 임계값 검출기는 자체 확률이 없다. 판정에 쓴 표면이 얼마나 되는지를
            # 신뢰도로 보고한다. 사과가 거의 안 보이면 비율도 못 믿는다.
            color_confidence=1.0,
            damage_confidence=None,
            severe_confidence=None,
        )
