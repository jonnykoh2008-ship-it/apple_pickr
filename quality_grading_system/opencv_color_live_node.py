"""Display three-view OpenCV coloration and visible-damage ratios.

This GPU PC 2 diagnostic node measures the fraction of valid visible apple
surface classified as target red or visible damage from synchronized top,
left, and right views.  Each synchronized triplet is combined by matching
pixel count divided by valid-surface pixel count, so a small or partially
ignored view is not weighted the same as a fully visible view.  The result
describes the three observed surfaces, not the complete physical apple surface.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from time import perf_counter
from typing import Any

import cv2
import numpy as np

from conveyor_camera_adapter_node import decode_depth_mm, decode_rgb_bgr, stamp_to_ns
from opencv_damage_grader import DamageDetectionConfig, DamageMasks, detect_damage
from opencv_damage_live_node import (
    DEFAULT_MAX_APPLES,
    DEFAULT_SYNC_QUEUE_SIZE,
    AppleInstance,
    AppleInstanceConfig,
    ExactRgbDepthSynchronizer,
    detect_apple_instances,
)


KNOWN_CAMERA_VIEWS = ("top", "left", "right")

# 현재 스테이지에는 conv_rsd455(탑뷰) 한 대만 있다. 뷰를 늘리려면 슬라이스를
# 넓히면 되고, 아래 로직은 전부 CAMERA_VIEWS 길이를 따라간다.
CAMERA_VIEWS = KNOWN_CAMERA_VIEWS[:1]
THREE_VIEW_READY = "READY"
THREE_VIEW_WAITING = "WAITING"
THREE_VIEW_RECHECK = "RECHECK"
DEFAULT_VIEW_TOPICS = {
    "top": (
        "/conveyor_camera/color/image_raw",
        "/conveyor_camera/depth/image_raw",
    ),
    "left": (
        "/conveyor_camera_01/color/image_raw",
        "/conveyor_camera_01/depth/image_raw",
    ),
    "right": (
        "/conveyor_camera_02/color/image_raw",
        "/conveyor_camera_02/depth/image_raw",
    ),
}
VISIBLE_DAMAGE_CONFIG = DamageDetectionConfig(max_component_area_ratio=1.0)


try:
    import rclpy
    from appleproj_interfaces.msg import QualityResult as QualityResultMessage
    from rclpy.node import Node
    from rclpy.parameter import Parameter
    from rclpy.qos import (
        DurabilityPolicy,
        HistoryPolicy,
        QoSProfile,
        ReliabilityPolicy,
    )
    from sensor_msgs.msg import Image

    _ROS_IMPORT_ERROR: ImportError | None = None
except ImportError as exc:
    rclpy = None  # type: ignore[assignment]
    Node = object  # type: ignore[assignment,misc]
    Parameter = None  # type: ignore[assignment,misc]
    QoSProfile = None  # type: ignore[assignment,misc]
    ReliabilityPolicy = None  # type: ignore[assignment,misc]
    DurabilityPolicy = None  # type: ignore[assignment,misc]
    HistoryPolicy = None  # type: ignore[assignment,misc]
    Image = None  # type: ignore[assignment,misc]
    QualityResultMessage = None  # type: ignore[assignment,misc]
    _ROS_IMPORT_ERROR = exc


@dataclass(frozen=True)
class ColorMeasurementConfig:
    """Provisional camera-space thresholds for target-red measurement."""

    red_hue_low_max: int = 12
    red_hue_high_min: int = 168
    red_min_saturation: int = 80
    red_min_value: int = 35
    red_max_value: int = 244
    specular_min_value: int = 245
    specular_max_saturation: int = 60
    shadow_max_value: int = 25
    edge_exclusion_ratio: float = 0.10
    morphology_kernel: int = 3

    def __post_init__(self) -> None:
        if not 0 <= self.red_hue_low_max < self.red_hue_high_min <= 179:
            raise ValueError("red hue limits must satisfy 0 <= low < high <= 179")
        for name, value in (
            ("red_min_saturation", self.red_min_saturation),
            ("red_min_value", self.red_min_value),
            ("red_max_value", self.red_max_value),
            ("specular_min_value", self.specular_min_value),
            ("specular_max_saturation", self.specular_max_saturation),
            ("shadow_max_value", self.shadow_max_value),
        ):
            if not 0 <= value <= 255:
                raise ValueError(f"{name} must be between 0 and 255")
        if self.red_min_value > self.red_max_value:
            raise ValueError("red_min_value must not exceed red_max_value")
        if not 0.0 <= self.edge_exclusion_ratio < 1.0:
            raise ValueError("edge_exclusion_ratio must be in [0, 1)")
        if self.morphology_kernel < 1 or self.morphology_kernel % 2 == 0:
            raise ValueError("morphology_kernel must be a positive odd integer")


@dataclass(frozen=True)
class ColorMasks:
    valid_surface: np.ndarray
    target_red: np.ndarray
    ignored: np.ndarray


@dataclass(frozen=True)
class LiveColorResult:
    apple: AppleInstance
    masks: ColorMasks
    damage: DamageMasks
    color_ratio: float
    damage_ratio: float


@dataclass(frozen=True)
class ThreeViewColorMeasurement:
    color_ratio: float
    damage_ratio: float
    target_red_pixels: int
    damage_pixels: int
    valid_surface_pixels: int
    views_used: int


@dataclass
class ApproximateThreeViewSynchronizer:
    """Bounded nearest-stamp synchronizer for top, left, and right RGB-D pairs."""

    queue_size: int = DEFAULT_SYNC_QUEUE_SIZE
    tolerance_ms: float = 20.0
    # 뷰 구성은 스테이지마다 다르다. 기본은 현재 활성 카메라를 따른다.
    views: tuple[str, ...] = CAMERA_VIEWS

    def __post_init__(self) -> None:
        if self.queue_size < 1:
            raise ValueError("queue_size must be positive")
        if not np.isfinite(self.tolerance_ms) or self.tolerance_ms < 0.0:
            raise ValueError("tolerance_ms must be finite and non-negative")
        self._tolerance_ns = int(round(self.tolerance_ms * 1_000_000.0))
        self._pending: dict[
            str, OrderedDict[int, tuple[Any, Any]]
        ] = {view: OrderedDict() for view in self.views}

    def add(
        self,
        view_name: str,
        rgb_depth: tuple[Any, Any],
    ) -> tuple[tuple[Any, Any], ...] | None:
        if view_name not in self.views:
            raise ValueError(f"unsupported camera view: {view_name}")
        rgb_message, depth_message = rgb_depth
        rgb_stamp = stamp_to_ns(rgb_message.header.stamp)
        depth_stamp = stamp_to_ns(depth_message.header.stamp)
        if rgb_stamp != depth_stamp:
            raise ValueError("RGB and depth timestamps must match within one view")
        pending = self._pending[view_name]
        pending[rgb_stamp] = rgb_depth
        pending.move_to_end(rgb_stamp)
        while len(pending) > self.queue_size:
            pending.popitem(last=False)
        if any(not self._pending[view] for view in self.views):
            return None

        selected_stamps = {view_name: rgb_stamp}
        for other_view in self.views:
            if other_view == view_name:
                continue
            selected_stamps[other_view] = min(
                self._pending[other_view],
                key=lambda stamp: abs(stamp - rgb_stamp),
            )
        interval_ns = max(selected_stamps.values()) - min(selected_stamps.values())
        if interval_ns > self._tolerance_ns:
            return None

        synchronized = tuple(
            self._pending[view][selected_stamps[view]] for view in self.views
        )
        for view in self.views:
            selected_stamp = selected_stamps[view]
            view_pending = self._pending[view]
            stale_stamps = [stamp for stamp in view_pending if stamp <= selected_stamp]
            for stamp in stale_stamps:
                view_pending.pop(stamp, None)
        return synchronized


def combine_three_view_measurements(
    results: tuple[LiveColorResult, ...],
    *,
    views: tuple[str, ...] = CAMERA_VIEWS,
) -> ThreeViewColorMeasurement:
    """Combine exactly three color/damage observations by visible pixel area."""

    if len(results) != len(views):
        raise ValueError("top, left, and right color observations are required")
    target_red_pixels = sum(int(result.masks.target_red.sum()) for result in results)
    damage_pixels = sum(int(result.damage.combined.sum()) for result in results)
    valid_surface_pixels = sum(
        int(result.masks.valid_surface.sum()) for result in results
    )
    if valid_surface_pixels < 1:
        raise ValueError("three-view measurement has no valid apple surface pixels")
    return ThreeViewColorMeasurement(
        color_ratio=float(target_red_pixels / valid_surface_pixels),
        damage_ratio=float(damage_pixels / valid_surface_pixels),
        target_red_pixels=target_red_pixels,
        damage_pixels=damage_pixels,
        valid_surface_pixels=valid_surface_pixels,
        views_used=len(results),
    )


def combine_three_view_result_sets(
    view_results: dict[str, tuple[LiveColorResult, ...]],
    *,
    views: tuple[str, ...] = CAMERA_VIEWS,
) -> tuple[LiveColorResult, ...]:
    """Require one apple in every view and combine the three observations."""

    missing = [view for view in views if view not in view_results]
    if missing:
        raise ValueError(f"missing camera views: {missing}")
    counts = {view: len(view_results[view]) for view in views}
    if len(set(counts.values())) != 1 or counts["top"] != 1:
        raise ValueError(
            "all three camera views must detect exactly one apple; "
            f"got {counts}"
        )

    combined = []
    for apple_index in range(counts["top"]):
        observations = tuple(
            view_results[view][apple_index] for view in views
        )
        measurement = combine_three_view_measurements(observations)
        top_result = observations[0]
        combined.append(
            LiveColorResult(
                apple=top_result.apple,
                masks=top_result.masks,
                damage=top_result.damage,
                color_ratio=measurement.color_ratio,
                damage_ratio=measurement.damage_ratio,
            )
        )
    return tuple(combined)


def classify_three_view_detection(
    view_results: dict[str, tuple[LiveColorResult, ...]],
    *,
    views: tuple[str, ...] = CAMERA_VIEWS,
) -> tuple[str, dict[str, int]]:
    """Classify one synchronized set without treating partial entry as an error."""

    missing = [view for view in views if view not in view_results]
    if missing:
        raise ValueError(f"missing camera views: {missing}")
    counts = {view: len(view_results[view]) for view in views}
    if all(count == 1 for count in counts.values()):
        return THREE_VIEW_READY, counts
    if any(count > 1 for count in counts.values()):
        return THREE_VIEW_RECHECK, counts
    return THREE_VIEW_WAITING, counts


@dataclass(frozen=True)
class TemporalColorConfig:
    max_track_jump_ratio: float = 2.5
    expire_after_missing_frames: int = 15

    def __post_init__(self) -> None:
        if self.max_track_jump_ratio <= 0.0:
            raise ValueError("max_track_jump_ratio must be positive")
        if self.expire_after_missing_frames < 1:
            raise ValueError("expire_after_missing_frames must be positive")


@dataclass(frozen=True)
class ColorObservation:
    frame_index: int
    color_ratio: float
    damage_ratio: float = 0.0


@dataclass(frozen=True)
class ColorSummary:
    frame_count: int
    observed_surface_ratio: float
    standard_deviation: float
    minimum_ratio: float
    maximum_ratio: float
    ready: bool
    grade: str | None
    damage_ratio: float = 0.0
    damage_standard_deviation: float = 0.0


@dataclass(frozen=True)
class ColorResultPayload:
    inspection_id: str
    apple_id: str
    grade: str
    color_ratio: float
    frames_used: int
    frame_indices: tuple[int, ...]
    status: str = "VALID"


COLOR_HIGH_THRESHOLD = 0.60
COLOR_MEDIUM_THRESHOLD = 0.40
QUALITY_RESULT_TOPIC = "/quality/results"


def classify_color_ratio(color_ratio: float) -> str:
    """Return the approved Fuji-family coloration grade."""

    ratio = float(color_ratio)
    if not np.isfinite(ratio) or not 0.0 <= ratio <= 1.0:
        raise ValueError("color_ratio must be finite and between 0 and 1")
    if ratio >= COLOR_HIGH_THRESHOLD:
        return "HIGH"
    if ratio >= COLOR_MEDIUM_THRESHOLD:
        return "MEDIUM"
    return "LOW"


def make_color_result_payload(
    track_id: int,
    summary: ColorSummary,
) -> ColorResultPayload:
    """Build the transport-neutral final result for one completed track."""

    if track_id < 1:
        raise ValueError("track_id must be positive")
    if not summary.ready or summary.grade is None or summary.frame_count < 1:
        raise ValueError("a completed color track must contain valid observations")
    if summary.frame_count > 65_535:
        raise ValueError("frames_used exceeds the QualityResult uint16 contract")
    suffix = str(track_id)
    return ColorResultPayload(
        inspection_id=f"opencv-color-{suffix}",
        apple_id=f"apple-{suffix}",
        grade=summary.grade,
        color_ratio=summary.observed_surface_ratio,
        frames_used=summary.frame_count,
        frame_indices=tuple(range(summary.frame_count)),
    )


def appearance_descriptor(
    image_bgr: np.ndarray,
    result: LiveColorResult,
    size: int = 16,
) -> np.ndarray:
    """Return a position-normalized colour thumbnail for view deduplication."""

    if size < 4:
        raise ValueError("appearance descriptor size must be at least 4")
    x, y, width, height = result.apple.bounding_box
    crop = image_bgr[y : y + height, x : x + width]
    valid = result.masks.valid_surface[y : y + height, x : x + width]
    if crop.size == 0 or not valid.any():
        return np.zeros(size * size * 4, dtype=np.float32)
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:, :, 0] /= 179.0
    hsv[:, :, 1:] /= 255.0
    resized_hsv = cv2.resize(hsv, (size, size), interpolation=cv2.INTER_AREA)
    resized_valid = cv2.resize(
        valid.astype(np.uint8),
        (size, size),
        interpolation=cv2.INTER_NEAREST,
    ).astype(np.float32)
    descriptor = np.concatenate(
        (
            (resized_hsv * resized_valid[:, :, None]).reshape(-1),
            resized_valid.reshape(-1),
        )
    )
    return descriptor.astype(np.float32)


class ObservedColorAccumulator:
    """Accumulate every valid frame observed during one apple track."""

    def __init__(self) -> None:
        self.observations: list[ColorObservation] = []

    def add(self, observation: ColorObservation) -> bool:
        if self.observations and observation.frame_index <= self.observations[-1].frame_index:
            raise ValueError("frame_index must increase within an apple track")
        classify_color_ratio(observation.color_ratio)
        if (
            not np.isfinite(observation.damage_ratio)
            or not 0.0 <= observation.damage_ratio <= 1.0
        ):
            raise ValueError("damage_ratio must be finite and between 0 and 1")
        self.observations.append(observation)
        return True

    def summary(self) -> ColorSummary:
        if not self.observations:
            return ColorSummary(0, 0.0, 0.0, 0.0, 0.0, False, None)
        ratios = np.asarray(
            [observation.color_ratio for observation in self.observations],
            dtype=np.float64,
        )
        damage_ratios = np.asarray(
            [observation.damage_ratio for observation in self.observations],
            dtype=np.float64,
        )
        observed_surface_ratio = float(np.median(ratios))
        return ColorSummary(
            frame_count=len(self.observations),
            observed_surface_ratio=observed_surface_ratio,
            standard_deviation=float(ratios.std()),
            minimum_ratio=float(ratios.min()),
            maximum_ratio=float(ratios.max()),
            ready=True,
            grade=classify_color_ratio(observed_surface_ratio),
            damage_ratio=float(np.median(damage_ratios)),
            damage_standard_deviation=float(damage_ratios.std()),
        )


@dataclass
class _AppleTrack:
    track_id: int
    center: tuple[float, float]
    size_px: float
    missing_frames: int
    observations: ObservedColorAccumulator


class TemporalColorTracker:
    """Associate separated apples and aggregate every valid visible frame."""

    def __init__(self, config: TemporalColorConfig) -> None:
        self.config = config
        self._tracks: dict[int, _AppleTrack] = {}
        self._next_track_id = 1
        self._completed: list[tuple[int, ColorSummary]] = []

    def drain_completed(self) -> tuple[tuple[int, ColorSummary], ...]:
        completed = tuple(self._completed)
        self._completed.clear()
        return completed

    def update(
        self,
        image_bgr: np.ndarray,
        results: tuple[LiveColorResult, ...],
        frame_index: int,
    ) -> tuple[tuple[int, ColorSummary, bool], ...]:
        for track in self._tracks.values():
            track.missing_frames += 1

        unmatched_tracks = set(self._tracks)
        assignments: list[tuple[int, ColorSummary, bool]] = []
        for result in results:
            center = result.apple.center
            x, y, width, height = result.apple.bounding_box
            size = float(max(width, height))
            matching = []
            for track_id in unmatched_tracks:
                track = self._tracks[track_id]
                distance = float(np.hypot(
                    center[0] - track.center[0],
                    center[1] - track.center[1],
                ))
                limit = max(size, track.size_px) * self.config.max_track_jump_ratio
                if distance <= limit:
                    matching.append((distance, track_id))
            if matching:
                _distance, track_id = min(matching)
                unmatched_tracks.remove(track_id)
                track = self._tracks[track_id]
                track.center = center
                track.size_px = size
                track.missing_frames = 0
            else:
                track_id = self._next_track_id
                self._next_track_id += 1
                track = _AppleTrack(
                    track_id=track_id,
                    center=center,
                    size_px=size,
                    missing_frames=0,
                    observations=ObservedColorAccumulator(),
                )
                self._tracks[track_id] = track

            observation = ColorObservation(
                frame_index=frame_index,
                color_ratio=result.color_ratio,
                damage_ratio=result.damage_ratio,
            )
            accepted = track.observations.add(observation)
            assignments.append((track_id, track.observations.summary(), accepted))

        expired = [
            track_id
            for track_id, track in self._tracks.items()
            if track.missing_frames > self.config.expire_after_missing_frames
        ]
        for track_id in expired:
            track = self._tracks.pop(track_id)
            self._completed.append((track_id, track.observations.summary()))
        return tuple(assignments)


def measure_target_red(
    image_bgr: np.ndarray,
    apple_mask: np.ndarray,
    config: ColorMeasurementConfig = ColorMeasurementConfig(),
) -> tuple[ColorMasks, float]:
    """Measure target-red pixels over the valid visible apple surface."""

    image = np.asarray(image_bgr)
    mask = np.asarray(apple_mask)
    if image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8:
        raise ValueError("image_bgr must be a uint8 HxWx3 array")
    if mask.shape != image.shape[:2] or mask.dtype != np.uint8:
        raise ValueError("apple_mask must be a uint8 mask matching the image")

    apple = mask > 0
    if not apple.any():
        empty = np.zeros(mask.shape, dtype=bool)
        return ColorMasks(empty, empty, empty), 0.0

    distance = cv2.distanceTransform(apple.astype(np.uint8), cv2.DIST_L2, 5)
    maximum_distance = float(distance.max())
    edge_distance = maximum_distance * config.edge_exclusion_ratio
    interior = apple & (distance > edge_distance)

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hue, saturation, value = cv2.split(hsv)
    specular = (
        (value >= config.specular_min_value)
        & (saturation <= config.specular_max_saturation)
    )
    shadow = value <= config.shadow_max_value
    valid_surface = interior & ~specular & ~shadow

    red_hue = (hue <= config.red_hue_low_max) | (
        hue >= config.red_hue_high_min
    )
    red = (
        valid_surface
        & red_hue
        & (saturation >= config.red_min_saturation)
        & (value >= config.red_min_value)
        & (value <= config.red_max_value)
    )
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (config.morphology_kernel, config.morphology_kernel),
    )
    red = cv2.morphologyEx(red.astype(np.uint8), cv2.MORPH_OPEN, kernel) > 0
    red &= valid_surface

    valid_pixels = int(valid_surface.sum())
    ratio = float(red.sum() / valid_pixels) if valid_pixels else 0.0
    ignored = apple & ~valid_surface
    return ColorMasks(valid_surface, red, ignored), ratio


def measure_visible_damage(
    image_bgr: np.ndarray,
    apple_mask: np.ndarray,
    valid_surface: np.ndarray,
    config: DamageDetectionConfig = VISIBLE_DAMAGE_CONFIG,
) -> tuple[DamageMasks, float]:
    """Measure visible color-based damage only on the valid apple surface."""

    image = np.asarray(image_bgr)
    mask = np.asarray(apple_mask)
    valid = np.asarray(valid_surface, dtype=bool)
    if image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8:
        raise ValueError("image_bgr must be a uint8 HxWx3 array")
    if mask.shape != image.shape[:2] or mask.dtype != np.uint8:
        raise ValueError("apple_mask must be a uint8 mask matching the image")
    if valid.shape != image.shape[:2]:
        raise ValueError("valid_surface must match the image")

    raw = detect_damage(
        cv2.cvtColor(image, cv2.COLOR_BGR2RGB),
        mask,
        config,
    )
    # A dark area alone is not considered damage because it may be a shadow.
    # Keep the visibly brown/bright wound color candidates and ignore the
    # brightness-only bruise candidate in this simple controlled-light path.
    combined = (raw.bright_wound | raw.browning) & valid
    empty = np.zeros_like(combined)
    damage = DamageMasks(
        bright_wound=raw.bright_wound & combined,
        browning=raw.browning & combined,
        bruise=empty,
        combined=combined,
    )
    valid_pixels = int(valid.sum())
    ratio = float(combined.sum() / valid_pixels) if valid_pixels else 0.0
    return damage, ratio


def process_color_frame(
    image_bgr: np.ndarray,
    depth_mm: np.ndarray | None = None,
    *,
    instance_config: AppleInstanceConfig = AppleInstanceConfig(),
    color_config: ColorMeasurementConfig = ColorMeasurementConfig(),
    damage_config: DamageDetectionConfig = VISIBLE_DAMAGE_CONFIG,
) -> tuple[np.ndarray, tuple[LiveColorResult, ...]]:
    """Detect apples and draw target-red and visible-damage pixel ratios."""

    apples = detect_apple_instances(
        image_bgr,
        instance_config,
        depth_mm=depth_mm,
    )
    # Depth suppresses rubber reflections in the live conveyor view, but a
    # missing or locally flat synthetic depth surface can also erase the true
    # apple candidate. Recover only when the depth frame genuinely cannot
    # separate foreground.
    #
    # Retrying whenever depth merely found nothing turns an empty conveyor into
    # a detection: the scenery beside the belt is grass, whose hue reads about
    # 39 and so slips under yellow_hue_max (40). Measured on the live top view,
    # ten consecutive frames of empty belt gave 0 apples with depth and 1-2
    # without, and every one of those was grass.
    if (
        depth_mm is not None
        and not apples
        and not depth_can_separate_foreground(
            depth_mm, instance_config.depth_foreground_margin_mm
        )
    ):
        apples = detect_apple_instances(image_bgr, instance_config)
    apples = tuple(sorted(apples, key=lambda apple: apple.center[0]))
    overlay = image_bgr.copy()
    results = []
    contour_colors = ((0, 220, 0), (255, 180, 0))
    for index, apple in enumerate(apples):
        masks, ratio = measure_target_red(image_bgr, apple.mask, color_config)
        damage, damage_ratio = measure_visible_damage(
            image_bgr,
            apple.mask,
            masks.valid_surface,
            damage_config,
        )
        results.append(LiveColorResult(apple, masks, damage, ratio, damage_ratio))

        red_overlay = np.zeros_like(overlay)
        red_overlay[masks.target_red] = (255, 0, 255)
        overlay = cv2.addWeighted(overlay, 1.0, red_overlay, 0.55, 0.0)

        damage_overlay = np.zeros_like(overlay)
        damage_overlay[damage.combined] = (0, 0, 255)
        overlay = cv2.addWeighted(overlay, 1.0, damage_overlay, 0.65, 0.0)

        ignored_overlay = np.zeros_like(overlay)
        ignored_overlay[masks.ignored] = (120, 120, 120)
        overlay = cv2.addWeighted(overlay, 1.0, ignored_overlay, 0.20, 0.0)

        color = contour_colors[index % len(contour_colors)]
        x, y, width, height = apple.bounding_box
        cv2.drawContours(overlay, [apple.contour], -1, color, 2)
        cv2.rectangle(overlay, (x, y), (x + width, y + height), color, 2)
        cv2.putText(
            overlay,
            (
                f"apple {index + 1} color={ratio * 100.0:.2f}% "
                f"damage={damage_ratio * 100.0:.2f}%"
            ),
            (x, max(24, y - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2,
            cv2.LINE_AA,
        )
    return overlay, tuple(results)


def depth_can_separate_foreground(depth_mm, foreground_margin_mm: int) -> bool:
    """True when the depth frame has enough range spread to filter on.

    A flat synthetic depth surface, or a frame with almost no valid range,
    cannot tell the apple from what lies behind it; a real view can, because
    the belt and the scenery behind it sit far apart. Measured on the live top
    view the spread is about 800 mm, while the flat synthetic case is 0.
    """
    depth = np.asarray(depth_mm, dtype=np.float64)
    valid = depth[np.isfinite(depth) & (depth > 0.0)]
    if valid.size < 16:
        return False
    spread = float(np.percentile(valid, 95) - np.percentile(valid, 5))
    return spread >= max(2 * int(foreground_margin_mm), 1)


def _reliable_qos(depth: int = 10) -> Any:
    if QoSProfile is None:
        raise RuntimeError("rclpy is required to construct QoS")
    return QoSProfile(
        reliability=ReliabilityPolicy.RELIABLE,
        durability=DurabilityPolicy.VOLATILE,
        history=HistoryPolicy.KEEP_LAST,
        depth=depth,
    )


class OpenCVColorLiveNode(Node):  # type: ignore[misc]
    def __init__(self) -> None:
        if _ROS_IMPORT_ERROR is not None:
            raise RuntimeError(
                "ROS 2 Python packages are unavailable; source ROS 2 Jazzy first"
            ) from _ROS_IMPORT_ERROR
        super().__init__(
            "opencv_color_live_node",
            parameter_overrides=[Parameter("use_sim_time", value=True)],
            automatically_declare_parameters_from_overrides=True,
        )
        instance_defaults = AppleInstanceConfig()
        color_defaults = ColorMeasurementConfig()
        temporal_defaults = TemporalColorConfig()
        defaults = (
            ("display", True),
            ("sync_queue_size", DEFAULT_SYNC_QUEUE_SIZE),
            ("cross_view_sync_tolerance_ms", 20.0),
            ("max_apples", DEFAULT_MAX_APPLES),
            ("apple_red_hue_low_max", instance_defaults.red_hue_low_max),
            ("apple_yellow_hue_max", instance_defaults.yellow_hue_max),
            ("apple_red_hue_high_min", instance_defaults.red_hue_high_min),
            ("apple_min_saturation", instance_defaults.min_saturation),
            ("apple_min_value", instance_defaults.min_value),
            ("apple_max_value", instance_defaults.max_value),
            ("apple_min_area_ratio", instance_defaults.min_area_ratio),
            ("apple_max_area_ratio", instance_defaults.max_area_ratio),
            ("apple_min_short_to_long_ratio", instance_defaults.min_short_to_long_ratio),
            ("apple_min_circularity", instance_defaults.min_circularity),
            ("apple_min_solidity", instance_defaults.min_solidity),
            ("apple_border_margin_px", instance_defaults.border_margin_px),
            ("apple_morphology_kernel", instance_defaults.morphology_kernel),
            ("depth_foreground_margin_mm", instance_defaults.depth_foreground_margin_mm),
            ("red_hue_low_max", color_defaults.red_hue_low_max),
            ("red_hue_high_min", color_defaults.red_hue_high_min),
            ("red_min_saturation", color_defaults.red_min_saturation),
            ("red_min_value", color_defaults.red_min_value),
            ("red_max_value", color_defaults.red_max_value),
            ("specular_min_value", color_defaults.specular_min_value),
            ("specular_max_saturation", color_defaults.specular_max_saturation),
            ("shadow_max_value", color_defaults.shadow_max_value),
            ("edge_exclusion_ratio", color_defaults.edge_exclusion_ratio),
            ("color_morphology_kernel", color_defaults.morphology_kernel),
            ("max_track_jump_ratio", temporal_defaults.max_track_jump_ratio),
            (
                "track_expire_after_missing_frames",
                temporal_defaults.expire_after_missing_frames,
            ),
        )
        for view_name in CAMERA_VIEWS:
            rgb_topic, depth_topic = DEFAULT_VIEW_TOPICS[view_name]
            defaults += (
                (f"{view_name}_rgb_topic", rgb_topic),
                (f"{view_name}_depth_topic", depth_topic),
            )
        for name, default in defaults:
            if not self.has_parameter(name):
                self.declare_parameter(name, default)

        self._display = bool(self.get_parameter("display").value)
        self._instance_config = AppleInstanceConfig(
            red_hue_low_max=int(self.get_parameter("apple_red_hue_low_max").value),
            yellow_hue_max=int(self.get_parameter("apple_yellow_hue_max").value),
            red_hue_high_min=int(self.get_parameter("apple_red_hue_high_min").value),
            min_saturation=int(self.get_parameter("apple_min_saturation").value),
            min_value=int(self.get_parameter("apple_min_value").value),
            max_value=int(self.get_parameter("apple_max_value").value),
            min_area_ratio=float(self.get_parameter("apple_min_area_ratio").value),
            max_area_ratio=float(self.get_parameter("apple_max_area_ratio").value),
            min_short_to_long_ratio=float(
                self.get_parameter("apple_min_short_to_long_ratio").value
            ),
            min_circularity=float(self.get_parameter("apple_min_circularity").value),
            min_solidity=float(self.get_parameter("apple_min_solidity").value),
            border_margin_px=int(self.get_parameter("apple_border_margin_px").value),
            morphology_kernel=int(self.get_parameter("apple_morphology_kernel").value),
            depth_foreground_margin_mm=int(
                self.get_parameter("depth_foreground_margin_mm").value
            ),
            max_apples=int(self.get_parameter("max_apples").value),
        )
        self._color_config = ColorMeasurementConfig(
            red_hue_low_max=int(self.get_parameter("red_hue_low_max").value),
            red_hue_high_min=int(self.get_parameter("red_hue_high_min").value),
            red_min_saturation=int(self.get_parameter("red_min_saturation").value),
            red_min_value=int(self.get_parameter("red_min_value").value),
            red_max_value=int(self.get_parameter("red_max_value").value),
            specular_min_value=int(self.get_parameter("specular_min_value").value),
            specular_max_saturation=int(
                self.get_parameter("specular_max_saturation").value
            ),
            shadow_max_value=int(self.get_parameter("shadow_max_value").value),
            edge_exclusion_ratio=float(self.get_parameter("edge_exclusion_ratio").value),
            morphology_kernel=int(
                self.get_parameter("color_morphology_kernel").value
            ),
        )
        self._damage_config = VISIBLE_DAMAGE_CONFIG
        sync_queue_size = int(self.get_parameter("sync_queue_size").value)
        self._view_synchronizers = {
            view_name: ExactRgbDepthSynchronizer(queue_size=sync_queue_size)
            for view_name in CAMERA_VIEWS
        }
        self._cross_view_sync_tolerance_ms = float(
            self.get_parameter("cross_view_sync_tolerance_ms").value
        )
        self._three_view_synchronizer = ApproximateThreeViewSynchronizer(
            queue_size=sync_queue_size,
            tolerance_ms=self._cross_view_sync_tolerance_ms,
        )
        self._temporal_config = TemporalColorConfig(
            max_track_jump_ratio=float(
                self.get_parameter("max_track_jump_ratio").value
            ),
            expire_after_missing_frames=int(
                self.get_parameter("track_expire_after_missing_frames").value
            ),
        )
        self._tracker = TemporalColorTracker(self._temporal_config)
        self._frames = 0
        self._last_detection_state = None
        topic_descriptions = []
        for view_name in CAMERA_VIEWS:
            rgb_topic = str(self.get_parameter(f"{view_name}_rgb_topic").value)
            depth_topic = str(self.get_parameter(f"{view_name}_depth_topic").value)
            self.create_subscription(
                Image,
                rgb_topic,
                lambda message, view=view_name: self._on_component(
                    view, "rgb", message
                ),
                _reliable_qos(),
            )
            self.create_subscription(
                Image,
                depth_topic,
                lambda message, view=view_name: self._on_component(
                    view, "depth", message
                ),
                _reliable_qos(),
            )
            topic_descriptions.append(f"{view_name}={rgb_topic} + {depth_topic}")
        self._result_publisher = self.create_publisher(
            QualityResultMessage,
            QUALITY_RESULT_TOPIC,
            _reliable_qos(),
        )
        self.get_logger().info(
            "OpenCV three-view coloration subscribed to "
            + "; ".join(topic_descriptions)
            + f"; cross-view tolerance={self._cross_view_sync_tolerance_ms:.1f}ms; "
            f"final results -> {QUALITY_RESULT_TOPIC}"
        )

    def _publish_final_result(self, track_id: int, summary: ColorSummary) -> None:
        payload = make_color_result_payload(track_id, summary)
        now = self.get_clock().now().to_msg()
        message = QualityResultMessage()
        message.header.stamp = now
        message.header.frame_id = "quality_grading"
        message.inspection_id = payload.inspection_id
        message.apple_id = payload.apple_id
        message.grade = int(getattr(QualityResultMessage, payload.grade))
        message.confidence = float("nan")
        message.color_ratio = payload.color_ratio
        message.diameter_mm = float("nan")
        message.damage_area_cm2 = float("nan")
        message.frames_used = payload.frames_used
        message.frame_indices = list(payload.frame_indices)
        message.result_timestamp = now
        message.status = int(getattr(QualityResultMessage, payload.status))
        self._result_publisher.publish(message)

    def _on_component(self, view_name: str, component: str, message: Any) -> None:
        rgb_depth = self._view_synchronizers[view_name].add(component, message)
        if rgb_depth is None:
            return
        synchronized = self._three_view_synchronizer.add(view_name, rgb_depth)
        if synchronized is not None:
            self._process_three_views(synchronized)

    def _process_three_views(
        self,
        synchronized: tuple[tuple[Any, Any], ...],
    ) -> None:
        started = perf_counter()
        images = {}
        overlays = {}
        view_results = {}
        try:
            for view_name, (rgb_message, depth_message) in zip(
                CAMERA_VIEWS, synchronized, strict=True
            ):
                image = decode_rgb_bgr(rgb_message)
                depth = decode_depth_mm(depth_message)
                overlay, results = process_color_frame(
                    image,
                    depth,
                    instance_config=self._instance_config,
                    color_config=self._color_config,
                    damage_config=self._damage_config,
                )
                images[view_name] = image
                overlays[view_name] = overlay
                view_results[view_name] = results
        except Exception as exc:
            self.get_logger().error(
                f"Rejected three-view RGB-D set: {type(exc).__name__}: {exc}"
            )
            return

        self._frames += 1
        top_image = images["top"]
        top_overlay = overlays["top"]
        detection_state, detection_counts = classify_three_view_detection(view_results)
        if detection_state == THREE_VIEW_READY:
            results = combine_three_view_result_sets(view_results)
            tracked = self._tracker.update(top_image, results, self._frames)
        else:
            results = ()
            tracked = self._tracker.update(top_image, (), self._frames)

        if detection_state != self._last_detection_state:
            message = (
                f"three-view state={detection_state} counts={detection_counts}"
            )
            if detection_state == THREE_VIEW_RECHECK:
                self.get_logger().warning(message)
            else:
                self.get_logger().info(message)
            self._last_detection_state = detection_state

        for overlay in overlays.values():
            cv2.putText(
                overlay,
                f"three-view {detection_state} {detection_counts}",
                (16, 28),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, 255, 255),
                2,
                cv2.LINE_AA,
            )

        for result, (track_id, summary, accepted) in zip(
            results,
            tracked,
            strict=True,
        ):
            x, y, _width, height = result.apple.bounding_box
            state = "TRACK"
            marker = "+" if accepted else "="
            grade = summary.grade if summary.grade is not None else "--"
            cv2.putText(
                top_overlay,
                (
                    f"id={track_id} color={summary.observed_surface_ratio * 100.0:.2f}% "
                    f"damage={summary.damage_ratio * 100.0:.2f}% "
                    f"sets={summary.frame_count} images={summary.frame_count * len(CAMERA_VIEWS)} "
                    f"grade={grade} {state}{marker}"
                ),
                (x, min(top_overlay.shape[0] - 8, y + height + 22)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.52,
                (0, 255, 255),
                2,
                cv2.LINE_AA,
            )

        for track_id, summary in self._tracker.drain_completed():
            try:
                self._publish_final_result(track_id, summary)
            except Exception as exc:
                self.get_logger().error(
                    f"Failed to publish final coloration result for id={track_id}: "
                    f"{type(exc).__name__}: {exc}"
                )
                continue
            self.get_logger().info(
                f"FINAL id={track_id} "
                f"observed_surface_color_ratio={summary.observed_surface_ratio * 100.0:.2f}% "
                f"visible_damage_ratio={summary.damage_ratio * 100.0:.2f}% "
                f"sets={summary.frame_count} "
                f"images={summary.frame_count * len(CAMERA_VIEWS)} grade={summary.grade} "
                f"published={QUALITY_RESULT_TOPIC}"
            )

        elapsed_ms = (perf_counter() - started) * 1000.0
        if self._frames % 30 == 0:
            if detection_state == THREE_VIEW_READY:
                detail = ", ".join(
                    (
                        f"id={track_id} color={summary.observed_surface_ratio * 100.0:.2f}% "
                        f"damage={summary.damage_ratio * 100.0:.2f}% "
                        f"sets={summary.frame_count} "
                        f"images={summary.frame_count * len(CAMERA_VIEWS)} "
                        f"grade={summary.grade if summary.grade else '--'}"
                    )
                    for result, (track_id, summary, _accepted) in zip(
                        results,
                        tracked,
                        strict=True,
                    )
                )
            else:
                detail = (
                    f"state={detection_state} counts={detection_counts} "
                    "waiting for exactly one apple in every view"
                )
            self.get_logger().info(
                f"frame={self._frames} processing={elapsed_ms:.1f}ms {detail}"
            )
        if self._display:
            cv2.imshow("conv_rsd455 top three-view coloration", top_overlay)
            cv2.imshow("conv_rsd455_01 left coloration", overlays["left"])
            cv2.imshow("conv_rsd455_02 right coloration", overlays["right"])
            if cv2.waitKey(1) & 0xFF in (27, ord("q")):
                rclpy.shutdown()

    def destroy_node(self) -> bool:
        if self._display:
            cv2.destroyAllWindows()
        return super().destroy_node()


def main(args: list[str] | None = None) -> None:
    if rclpy is None:
        raise RuntimeError(
            "ROS 2 Python packages are unavailable; source ROS 2 Jazzy first"
        ) from _ROS_IMPORT_ERROR
    rclpy.init(args=args)
    node = OpenCVColorLiveNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
