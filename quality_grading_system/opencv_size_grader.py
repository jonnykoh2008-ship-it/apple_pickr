"""OpenCV proof-of-concept for single-apple size grading.

The detector is intentionally limited to the first GPU PC 2 milestone: one
apple, a fixed camera, and a background whose saturation differs from the
apple. Apparent pixel diameter is converted to millimetres only through an
explicit calibration supplied by the operator.
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from statistics import median
from typing import Iterable

from quality_rules import Grade, grade_diameter_mm


SUPPORTED_IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff"}


class AppleNotDetected(RuntimeError):
    """Raised when no contour satisfies the single-apple detector."""


@dataclass(frozen=True)
class DetectionConfig:
    min_saturation: int = 35
    min_value: int = 30
    max_value: int = 250
    # 컨베이어 카메라 3대는 검사 지점까지 거리가 다르다. 가장 먼 view에서
    # 사과는 지름 40px대까지 작아지므로, 0.002(지름 48px 요구)는 그 view를
    # 구조적으로 배제해 3면 검사를 무력화한다. 색상·원형도 필터가 잡음을
    # 걸러주므로 면적 하한은 낮게 둔다.
    min_area_ratio: float = 0.0005
    morphology_kernel: int = 7
    # 사과 과피의 hue 대역. OpenCV hue는 0~179이고 빨강은 0에서 감기므로
    # [0, low_max]와 [high_min, 179] 두 구간을 함께 사용한다. 채도만 보면
    # 잔디·컨베이어 구조물처럼 화면에서 더 큰 물체가 사과를 이긴다.
    #
    # 상한을 주황(hue 25)까지 넓힌다. 사과는 순수 빨강이 아니라 노랑기를
    # 띠는 면이 있고, 위에서 내려다보는 view는 조명 반사로 더 주황에
    # 가까워진다. 빨강만 허용하면 그 view가 통째로 검출되지 않는다.
    # 초록(35~85)과 파랑(100~130)은 여전히 배제된다.
    hue_low_max: int = 25
    hue_high_min: int = 165
    # 사과는 최소 외접원을 어느 정도 채운다. 가늘고 긴 구조물을 배제한다.
    min_fill_ratio: float = 0.55
    # 화면을 뒤덮는 배경 구조물을 사과로 잡지 않도록 상한을 둔다.
    max_area_ratio: float = 0.60

    def __post_init__(self) -> None:
        for name, value in (
            ("min_saturation", self.min_saturation),
            ("min_value", self.min_value),
            ("max_value", self.max_value),
        ):
            if not 0 <= value <= 255:
                raise ValueError(f"{name} must be between 0 and 255")
        if self.min_value > self.max_value:
            raise ValueError("min_value cannot exceed max_value")
        if not 0.0 < self.min_area_ratio < 1.0:
            raise ValueError("min_area_ratio must be between 0 and 1")
        if self.morphology_kernel < 1 or self.morphology_kernel % 2 == 0:
            raise ValueError("morphology_kernel must be a positive odd integer")
        for name, value in (
            ("hue_low_max", self.hue_low_max),
            ("hue_high_min", self.hue_high_min),
        ):
            if not 0 <= value <= 179:
                raise ValueError(f"{name} must be between 0 and 179")
        if self.hue_low_max >= self.hue_high_min:
            raise ValueError("hue_low_max must be below hue_high_min")
        if not 0.0 <= self.min_fill_ratio <= 1.0:
            raise ValueError("min_fill_ratio must be between 0 and 1")
        if not self.min_area_ratio < self.max_area_ratio <= 1.0:
            raise ValueError("max_area_ratio must exceed min_area_ratio")


@dataclass(frozen=True)
class AppleDetection:
    contour: object
    bounding_box: tuple[int, int, int, int]
    center: tuple[float, float]
    diameter_px: float
    contour_area_px2: float
    confidence: float
    mask: object
    # 한 화면에 사과가 여러 개면 검사 대상이 모호하다. 붙어 있는 두 사과는
    # 하나의 윤곽으로 합쳐져 직경이 두 배로 측정되므로 호출부가 이 값을 보고
    # RECHECK 정책을 적용할 수 있어야 한다.
    candidate_count: int = 1


@dataclass(frozen=True)
class SizeGradeResult:
    detection: AppleDetection
    diameter_mm: float
    grade: Grade


@dataclass(frozen=True)
class SizeCalibration:
    mm_per_pixel: float
    intercept_mm: float = 0.0

    def __post_init__(self) -> None:
        if self.mm_per_pixel <= 0.0:
            raise ValueError("mm_per_pixel must be positive")

    def diameter_mm(self, diameter_px: float) -> float:
        return self.mm_per_pixel * diameter_px + self.intercept_mm

    @classmethod
    def from_pixels_per_mm(cls, pixels_per_mm: float) -> "SizeCalibration":
        if pixels_per_mm <= 0.0:
            raise ValueError("pixels_per_mm must be positive")
        return cls(mm_per_pixel=1.0 / pixels_per_mm)


def _cv2():
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError(
            "OpenCV is required; install the ROS/Ubuntu python3-opencv package"
        ) from exc
    return cv2


def detect_single_apple(
    image,
    config: DetectionConfig = DetectionConfig(),
) -> AppleDetection:
    """Return the largest saturated foreground contour as the apple."""

    cv2 = _cv2()
    import numpy as np

    if image is None or not isinstance(image, np.ndarray) or image.ndim != 3:
        raise ValueError("image must be a non-empty BGR array")
    if image.shape[2] != 3 or image.shape[0] < 2 or image.shape[1] < 2:
        raise ValueError("image must have shape HxWx3")

    blurred = cv2.GaussianBlur(image, (5, 5), 0)
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
    # 빨강은 hue 0에서 감기므로 두 구간을 합친다.
    mask = cv2.bitwise_or(
        cv2.inRange(
            hsv,
            np.array((0, config.min_saturation, config.min_value), dtype=np.uint8),
            np.array(
                (config.hue_low_max, 255, config.max_value), dtype=np.uint8
            ),
        ),
        cv2.inRange(
            hsv,
            np.array(
                (config.hue_high_min, config.min_saturation, config.min_value),
                dtype=np.uint8,
            ),
            np.array((179, 255, config.max_value), dtype=np.uint8),
        ),
    )
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (config.morphology_kernel, config.morphology_kernel),
    )
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )
    image_area = image.shape[0] * image.shape[1]
    minimum_area = image_area * config.min_area_ratio
    maximum_area = image_area * config.max_area_ratio

    def is_round_enough(candidate) -> bool:
        """Reject elongated structures such as conveyor frames and rails."""
        _centre, candidate_radius = cv2.minEnclosingCircle(
            cv2.convexHull(candidate)
        )
        enclosing_area = max(float(np.pi * candidate_radius * candidate_radius), 1.0)
        return float(cv2.contourArea(candidate)) / enclosing_area >= config.min_fill_ratio

    sized = [
        contour
        for contour in contours
        if minimum_area <= cv2.contourArea(contour) <= maximum_area
    ]
    candidates = [contour for contour in sized if is_round_enough(contour)]
    if not candidates:
        if sized:
            raise AppleNotDetected(
                f"{len(sized)} red contours matched the size range but none "
                f"filled {config.min_fill_ratio:.0%} of their enclosing circle"
            )
        raise AppleNotDetected(
            "no red foreground contour met the configured size range"
        )

    contour = max(candidates, key=cv2.contourArea)
    hull = cv2.convexHull(contour)
    (center_x, center_y), radius = cv2.minEnclosingCircle(hull)
    x, y, width, height = cv2.boundingRect(hull)
    contour_area = float(cv2.contourArea(contour))
    circle_area = max(float(np.pi * radius * radius), 1.0)
    fill_ratio = min(1.0, contour_area / circle_area)
    area_margin = min(1.0, contour_area / max(minimum_area * 4.0, 1.0))
    confidence = max(0.0, min(1.0, 0.7 * fill_ratio + 0.3 * area_margin))
    return AppleDetection(
        contour=contour,
        bounding_box=(int(x), int(y), int(width), int(height)),
        center=(float(center_x), float(center_y)),
        diameter_px=float(radius * 2.0),
        contour_area_px2=contour_area,
        confidence=confidence,
        mask=mask,
        candidate_count=len(candidates),
    )


def pixels_per_mm_from_reference(
    image,
    reference_diameter_mm: float,
    config: DetectionConfig = DetectionConfig(),
) -> float:
    if reference_diameter_mm <= 0.0:
        raise ValueError("reference_diameter_mm must be positive")
    detection = detect_single_apple(image, config)
    return detection.diameter_px / reference_diameter_mm


def fit_linear_calibration(
    diameter_pixels: Iterable[float],
    diameter_mm: Iterable[float],
) -> SizeCalibration:
    """Fit diameter_mm = slope * diameter_px + intercept."""

    pixels = [float(value) for value in diameter_pixels]
    millimetres = [float(value) for value in diameter_mm]
    if len(pixels) != len(millimetres):
        raise ValueError("pixel and millimetre calibration values must have equal length")
    if len(pixels) < 2:
        raise ValueError("at least two calibration points are required")
    mean_pixels = sum(pixels) / len(pixels)
    mean_mm = sum(millimetres) / len(millimetres)
    variance = sum((value - mean_pixels) ** 2 for value in pixels)
    if variance <= 0.0:
        raise ValueError("calibration pixel diameters must not all be equal")
    slope = sum(
        (pixel - mean_pixels) * (millimetre - mean_mm)
        for pixel, millimetre in zip(pixels, millimetres)
    ) / variance
    intercept = mean_mm - slope * mean_pixels
    return SizeCalibration(slope, intercept)


def calibration_from_boundary_metadata(
    metadata_path: str | Path,
    config: DetectionConfig = DetectionConfig(),
    *,
    case_index: int = 0,
) -> tuple[SizeCalibration, tuple[tuple[float, float], ...]]:
    """Fit size calibration from one case of rendered size scenarios."""

    cv2 = _cv2()
    path = Path(metadata_path).expanduser().resolve()
    metadata = json.loads(path.read_text(encoding="utf-8"))
    samples: list[tuple[float, float]] = []
    for scenario in metadata.get("scenarios", []):
        if scenario.get("study") != "size":
            continue
        if int(scenario.get("case_index", -1)) != case_index:
            continue
        measured_mm = scenario.get("aggregate_measured_diameter_mm")
        frame_indices = scenario.get("frame_indices", [])
        if measured_mm is None or not frame_indices:
            continue
        view_diameters = []
        for frame_index in frame_indices:
            image_path = path.parent / f"rgb_{int(frame_index):04d}.png"
            image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            if image is None:
                raise ValueError(f"cannot read calibration image: {image_path}")
            view_diameters.append(detect_single_apple(image, config).diameter_px)
        samples.append((float(median(view_diameters)), float(measured_mm)))
    if len(samples) < 2:
        raise ValueError(
            f"metadata must contain at least two size scenarios for case {case_index}"
        )
    calibration = fit_linear_calibration(
        (sample[0] for sample in samples),
        (sample[1] for sample in samples),
    )
    return calibration, tuple(samples)


def grade_image_by_size(
    image,
    calibration: float | SizeCalibration,
    config: DetectionConfig = DetectionConfig(),
) -> SizeGradeResult:
    if isinstance(calibration, (int, float)):
        calibration = SizeCalibration.from_pixels_per_mm(float(calibration))
    detection = detect_single_apple(image, config)
    diameter_mm = calibration.diameter_mm(detection.diameter_px)
    return SizeGradeResult(
        detection=detection,
        diameter_mm=diameter_mm,
        grade=grade_diameter_mm(diameter_mm),
    )


def draw_size_result(image, result: SizeGradeResult):
    """Draw contour, bounding box, calibrated diameter and grade."""

    cv2 = _cv2()
    output = image.copy()
    x, y, width, height = result.detection.bounding_box
    grade_colors = {
        Grade.HIGH: (0, 200, 0),
        Grade.MEDIUM: (0, 200, 255),
        Grade.LOW: (0, 0, 255),
    }
    color = grade_colors[result.grade]
    cv2.drawContours(output, [result.detection.contour], -1, color, 2)
    cv2.rectangle(output, (x, y), (x + width, y + height), color, 2)
    label = (
        f"{result.grade.value}  {result.diameter_mm:.1f} mm  "
        f"{result.detection.diameter_px:.1f} px"
    )
    cv2.putText(
        output,
        label,
        (x, max(25, y - 10)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        color,
        2,
        cv2.LINE_AA,
    )
    return output


def _input_images(path: Path) -> Iterable[Path]:
    if path.is_file():
        if path.suffix.lower() not in SUPPORTED_IMAGE_SUFFIXES:
            raise ValueError(f"unsupported image suffix: {path.suffix}")
        return (path,)
    if not path.is_dir():
        raise FileNotFoundError(path)
    images = tuple(
        candidate
        for candidate in sorted(path.iterdir())
        if candidate.is_file()
        and candidate.suffix.lower() in SUPPORTED_IMAGE_SUFFIXES
    )
    if not images:
        raise ValueError(f"no supported images found in {path}")
    rgb_images = tuple(
        candidate for candidate in images if candidate.name.startswith("rgb_")
    )
    return rgb_images or images


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Detect one apple with OpenCV and grade it using calibrated size only."
        )
    )
    parser.add_argument("input", type=Path, help="Input image or directory")
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Annotated output directory",
    )
    parser.add_argument("--pixels-per-mm", type=float)
    parser.add_argument("--reference-image", type=Path)
    parser.add_argument("--reference-diameter-mm", type=float)
    parser.add_argument("--boundary-metadata", type=Path)
    parser.add_argument("--calibration-case-index", type=int, default=0)
    parser.add_argument("--min-saturation", type=int, default=35)
    parser.add_argument("--min-value", type=int, default=30)
    parser.add_argument("--max-value", type=int, default=250)
    parser.add_argument("--min-area-ratio", type=float, default=0.002)
    parser.add_argument("--morphology-kernel", type=int, default=7)
    parser.add_argument("--display", action="store_true")
    return parser


def main(args: list[str] | None = None) -> int:
    cv2 = _cv2()
    options = _parser().parse_args(args)
    config = DetectionConfig(
        min_saturation=options.min_saturation,
        min_value=options.min_value,
        max_value=options.max_value,
        min_area_ratio=options.min_area_ratio,
        morphology_kernel=options.morphology_kernel,
    )

    calibration = None
    calibration_samples: tuple[tuple[float, float], ...] = ()
    selected_methods = sum(
        (
            options.pixels_per_mm is not None,
            options.reference_image is not None
            or options.reference_diameter_mm is not None,
            options.boundary_metadata is not None,
        )
    )
    if selected_methods != 1:
        raise ValueError(
            "select exactly one calibration method: --pixels-per-mm, "
            "--reference-image with --reference-diameter-mm, or --boundary-metadata"
        )
    if options.boundary_metadata is not None:
        calibration, calibration_samples = calibration_from_boundary_metadata(
            options.boundary_metadata,
            config,
            case_index=options.calibration_case_index,
        )
    elif options.pixels_per_mm is not None:
        calibration = SizeCalibration.from_pixels_per_mm(options.pixels_per_mm)
    else:
        if (
            options.reference_image is None
            or options.reference_diameter_mm is None
        ):
            raise ValueError(
                "provide --pixels-per-mm or both --reference-image and "
                "--reference-diameter-mm"
            )
        reference = cv2.imread(str(options.reference_image), cv2.IMREAD_COLOR)
        if reference is None:
            raise ValueError(
                f"cannot read reference image: {options.reference_image}"
            )
        pixels_per_mm = pixels_per_mm_from_reference(
            reference,
            float(options.reference_diameter_mm),
            config,
        )
        calibration = SizeCalibration.from_pixels_per_mm(pixels_per_mm)
    assert calibration is not None

    options.output.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    for image_path in _input_images(options.input):
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            rows.append({"image": str(image_path), "status": "READ_FAILED"})
            continue
        try:
            result = grade_image_by_size(image, calibration, config)
        except AppleNotDetected as exc:
            rows.append(
                {
                    "image": str(image_path),
                    "status": "NO_DETECTION",
                    "error": str(exc),
                }
            )
            continue
        annotated = draw_size_result(image, result)
        output_path = options.output / image_path.name
        if not cv2.imwrite(str(output_path), annotated):
            raise RuntimeError(f"cannot write annotated image: {output_path}")
        rows.append(
            {
                "image": str(image_path),
                "status": "VALID",
                "grade": result.grade.value,
                "diameter_mm": f"{result.diameter_mm:.6f}",
                "diameter_px": f"{result.detection.diameter_px:.6f}",
                "confidence": f"{result.detection.confidence:.6f}",
                "calibration_mm_per_pixel": f"{calibration.mm_per_pixel:.9f}",
                "calibration_intercept_mm": f"{calibration.intercept_mm:.9f}",
                "annotated_image": str(output_path),
            }
        )
        print(
            f"{image_path.name}: grade={result.grade.value} "
            f"diameter={result.diameter_mm:.2f}mm "
            f"confidence={result.detection.confidence:.3f}"
        )
        if options.display:
            cv2.imshow("OpenCV apple size grading", annotated)
            if cv2.waitKey(0) in (27, ord("q")):
                break

    if options.display:
        cv2.destroyAllWindows()
    csv_path = options.output / "size_grading_results.csv"
    fieldnames = sorted({key for row in rows for key in row})
    with csv_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(
        f"calibration: mm_per_pixel={calibration.mm_per_pixel:.9f} "
        f"intercept_mm={calibration.intercept_mm:.6f} "
        f"samples={len(calibration_samples)}; results={csv_path}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
