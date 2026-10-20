"""Pure-Python quality labels shared by the demo and synthetic dataset tools.

The thresholds intentionally mirror ``docs/features/quality_grading.md``.
Model confidence remains configurable around the decided initial threshold so
it can be calibrated after validation.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
from statistics import median
from typing import Iterable


class Grade(str, Enum):
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"


class ResultStatus(str, Enum):
    VALID = "VALID"
    RECHECK = "RECHECK"
    UNCLASSIFIED = "UNCLASSIFIED"
    TIMEOUT = "TIMEOUT"
    LATE_RESULT = "LATE_RESULT"
    ID_MISMATCH = "ID_MISMATCH"
    INSUFFICIENT_VIEWS = "INSUFFICIENT_VIEWS"


@dataclass(frozen=True)
class AppleMeasurements:
    color_ratio: float | None = None
    diameter_mm: float | None = None
    damage_area_cm2: float | None = None
    severe_defect: bool | None = None


@dataclass(frozen=True)
class QualityResult:
    grade: str | None
    status: str
    confidence: float | None
    measurements: AppleMeasurements | None
    frames_used: tuple[int, ...]

    def to_dict(self) -> dict:
        value = asdict(self)
        value["frames_used"] = list(self.frames_used)
        return value


@dataclass(frozen=True)
class FrameMeasurements:
    """One frame's model/geometric outputs and their validity confidence.

    ``None`` means the measurement was not produced.  This is important for
    the current synthetic dataset: diameter labels exist, while color and
    damage annotations do not.
    """

    color_ratio: float | None = None
    diameter_mm: float | None = None
    damage_area_cm2: float | None = None
    severe_defect: bool | None = None
    color_confidence: float | None = None
    diameter_confidence: float | None = None
    damage_confidence: float | None = None
    severe_confidence: float | None = None
    # 착색률을 면적으로 가중해 통합하려면 비율만으로는 부족하다. 뷰마다
    # 보이는 표면 크기가 다르기 때문이다. 실측에서 원거리 view 는 2,212px,
    # 근거리 view 는 5,501px 를 봤는데 비율 평균은 둘에 같은 가중치를 준다.
    # 픽셀 수를 함께 실어 보내면 aggregate 가 sum(C)/sum(A) 를 계산할 수 있다.
    # 합성 데이터처럼 비율만 있는 입력에서는 None 으로 남고 평균으로 되돌아간다.
    color_pixels: int | None = None
    measurable_pixels: int | None = None

    @property
    def confidence(self) -> float | None:
        values = [
            value
            for value in (
                self.color_confidence,
                self.diameter_confidence,
                self.damage_confidence,
                self.severe_confidence,
            )
            if value is not None
        ]
        return sum(values) / len(values) if values else None


def grade_diameter_mm(diameter_mm: float) -> Grade:
    """Apply the approved size-only MVP boundaries."""
    if diameter_mm < 0.0:
        raise ValueError("diameter_mm must be non-negative")
    if diameter_mm < 60.0:
        return Grade.LOW
    if diameter_mm >= 75.0:
        return Grade.HIGH
    return Grade.MEDIUM


# 착색률 등급 경계. 사용자 승인을 받은 시험값이며 영구 확정값이 아니다.
# docs/features/quality_grading.md 는 이 경계를 TBD 로 두고 있다.
# 근거: 컨베이어 실측에서 잘 착색된 사과가 84~87%, 노란빛이 도는 사과가
# 54~56% 로 나왔다. 두 무리가 각 경계에서 4~7%p 여유를 두고 떨어지며,
# 이는 관측된 산포(약 +-1.5%p)보다 충분히 크다.
COLOR_HIGH_MIN_RATIO = 0.80
COLOR_MEDIUM_MIN_RATIO = 0.60


def grade_color_ratio(color_ratio: float) -> Grade:
    """Apply the approved colour-ratio boundaries.

    경계값은 더 좋은 등급 쪽에 포함한다. 착색은 높을수록 좋으므로 정확히
    0.80 은 HIGH 이고 정확히 0.60 은 MEDIUM 이다. 크기 규칙에서 정확히
    75mm 가 HIGH 인 것과 같은 방향이다.
    """
    if not 0.0 <= color_ratio <= 1.0:
        raise ValueError("color_ratio must be between 0 and 1")
    if color_ratio >= COLOR_HIGH_MIN_RATIO:
        return Grade.HIGH
    if color_ratio >= COLOR_MEDIUM_MIN_RATIO:
        return Grade.MEDIUM
    return Grade.LOW


def grade_measurements(measurements: AppleMeasurements) -> Grade:
    """Grade only by diameter during the current OpenCV MVP."""
    if measurements.diameter_mm is None:
        raise ValueError("diameter_mm is required for size-only grading")
    return grade_diameter_mm(measurements.diameter_mm)


def aggregate_frames(
    frame_measurements: Iterable[AppleMeasurements],
    frame_indices: Iterable[int],
    *,
    min_valid_views: int = 1,
    confidence: float | None = None,
) -> QualityResult:
    """Aggregate size measurements for one apple using the diameter median."""
    values = list(frame_measurements)
    indices = tuple(frame_indices)
    if len(values) != len(indices):
        raise ValueError("frame_measurements and frame_indices must have equal length")
    if len(values) < min_valid_views:
        fallback = values[0] if values else None
        return QualityResult(None, ResultStatus.INSUFFICIENT_VIEWS, confidence, fallback, indices)

    diameters = [item.diameter_mm for item in values]
    if any(value is None for value in diameters):
        return QualityResult(
            None,
            ResultStatus.UNCLASSIFIED,
            confidence,
            None,
            indices,
        )
    measurements = AppleMeasurements(
        diameter_mm=float(
            median(float(value) for value in diameters if value is not None)
        ),
    )
    return QualityResult(grade_measurements(measurements), ResultStatus.VALID, confidence, measurements, indices)


def _is_valid(value: object | None, confidence: float | None, threshold: float) -> bool:
    return value is not None and confidence is not None and confidence >= threshold


def aggregate_measurement_frames(
    frame_measurements: Iterable[FrameMeasurements],
    frame_indices: Iterable[int],
    *,
    confidence_threshold: float = 0.5,
    min_valid_views: int = 1,
    min_damage_views: int = 0,
    grade_by: str = "size",
) -> QualityResult:
    """Aggregate model measurements using the decided GPU PC 2 policy."""

    values = list(frame_measurements)
    indices = tuple(frame_indices)
    if len(values) != len(indices):
        raise ValueError("frame_measurements and frame_indices must have equal length")
    if not 0.0 <= confidence_threshold <= 1.0:
        raise ValueError("confidence_threshold must be between 0 and 1")
    if grade_by not in ("size", "color"):
        raise ValueError("grade_by must be 'size' or 'color'")

    del min_damage_views
    if values and all(item.diameter_mm is None for item in values):
        return QualityResult(None, ResultStatus.UNCLASSIFIED, None, None, indices)

    valid_pairs = [
        (index, item)
        for index, item in zip(indices, values)
        if _is_valid(
            item.diameter_mm,
            item.diameter_confidence,
            confidence_threshold,
        )
    ]
    used_indices = tuple(index for index, _ in valid_pairs)
    valid_confidences = [
        item.diameter_confidence
        for _, item in valid_pairs
        if item.diameter_confidence is not None
    ]
    final_confidence = (
        sum(valid_confidences) / len(valid_confidences)
        if valid_confidences
        else None
    )

    if len(valid_pairs) < min_valid_views:
        return QualityResult(
            None,
            ResultStatus.INSUFFICIENT_VIEWS,
            final_confidence,
            None,
            used_indices,
        )

    diameters = [item.diameter_mm for _, item in valid_pairs]
    assert all(value is not None for value in diameters)

    # G2-03 확정 통합 규칙 중 현재 쓰는 두 가지: 직경은 중앙값, 착색률은
    # 여러 방향을 합쳐 특정 시점의 편향을 줄인다.
    #
    # 합치는 방법은 뷰별 비율의 단순 평균이 아니라 면적 가중이다. 착색률이
    # 답하려는 질문은 "표면 중 얼마가 착색되었는가"인데, 뷰마다 보이는 표면
    # 크기가 다르므로 비율을 그냥 평균하면 작은 뷰가 과대 대표된다. 실측
    # apple1 은 뷰별로 0.188 / 0.892 / 0.708 이었고 단순 평균은 0.596,
    # 면적 가중은 0.527 이다. 고르게 착색된 사과에서는 둘이 거의 같고
    # (apple1_01 은 0.960 대 0.957) 착색이 치우친 사과에서만 갈린다.
    color_pairs = [
        (int(item.color_pixels), int(item.measurable_pixels))
        for _, item in valid_pairs
        if item.color_pixels is not None and item.measurable_pixels is not None
    ]
    color_values = [
        float(item.color_ratio)
        for _, item in valid_pairs
        if item.color_ratio is not None
    ]
    total_measurable = sum(area for _, area in color_pairs)
    if color_pairs and total_measurable > 0:
        color_ratio = sum(coloured for coloured, _ in color_pairs) / total_measurable
    elif color_values:
        # 픽셀 수가 없는 입력(합성 데이터 라벨 등)은 비율 평균으로 되돌아간다.
        color_ratio = sum(color_values) / len(color_values)
    else:
        color_ratio = None
    measurements = AppleMeasurements(
        diameter_mm=float(
            median(float(value) for value in diameters if value is not None)
        ),
        color_ratio=color_ratio,
    )

    if grade_by == "color":
        if measurements.color_ratio is None:
            return QualityResult(
                None, ResultStatus.UNCLASSIFIED, final_confidence, measurements, used_indices
            )
        grade = grade_color_ratio(measurements.color_ratio)
    else:
        grade = grade_measurements(measurements)

    return QualityResult(
        grade,
        ResultStatus.VALID,
        final_confidence,
        measurements,
        used_indices,
    )
