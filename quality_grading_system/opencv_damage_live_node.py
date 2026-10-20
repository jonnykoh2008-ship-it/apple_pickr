"""Display OpenCV damage candidates from the conveyor RGB ROS 2 topic.

The GPU PC 2 pilot supports at most two spatially separated apples.  It does
not attempt to split touching/occluded apples and it does not publish final
quality grades.
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


DEFAULT_RGB_TOPIC = "/conveyor_camera/color/image_raw"
DEFAULT_DEPTH_TOPIC = "/conveyor_camera/depth/image_raw"
DEFAULT_MAX_APPLES = 2
DEFAULT_SYNC_QUEUE_SIZE = 20


try:
    import rclpy
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
    _ROS_IMPORT_ERROR = exc


@dataclass(frozen=True)
class AppleInstance:
    contour: np.ndarray
    mask: np.ndarray
    bounding_box: tuple[int, int, int, int]
    center: tuple[float, float]
    area_px2: float


@dataclass(frozen=True)
class LiveDamageResult:
    apple: AppleInstance
    damage: DamageMasks
    damage_pixel_ratio: float


@dataclass
class ExactRgbDepthSynchronizer:
    """Bounded exact-stamp synchronizer for aligned conveyor RGB-D frames."""

    queue_size: int = DEFAULT_SYNC_QUEUE_SIZE

    def __post_init__(self) -> None:
        if self.queue_size < 1:
            raise ValueError("queue_size must be positive")
        self._pending: OrderedDict[int, dict[str, Any]] = OrderedDict()

    def add(self, component: str, message: Any) -> tuple[Any, Any] | None:
        if component not in {"rgb", "depth"}:
            raise ValueError(f"unsupported synchronized component: {component}")
        key = stamp_to_ns(message.header.stamp)
        pending = self._pending.setdefault(key, {})
        pending[component] = message
        self._pending.move_to_end(key)
        while len(self._pending) > self.queue_size:
            self._pending.popitem(last=False)
        if "rgb" not in pending or "depth" not in pending:
            return None
        self._pending.pop(key, None)
        return pending["rgb"], pending["depth"]


@dataclass(frozen=True)
class AppleInstanceConfig:
    red_hue_low_max: int = 12
    yellow_hue_max: int = 40
    red_hue_high_min: int = 168
    min_saturation: int = 70
    min_value: int = 35
    max_value: int = 255
    min_area_ratio: float = 0.0002
    max_area_ratio: float = 0.20
    min_short_to_long_ratio: float = 0.55
    min_circularity: float = 0.45
    min_solidity: float = 0.75
    border_margin_px: int = 2
    morphology_kernel: int = 5
    depth_foreground_margin_mm: int = 8
    max_apples: int = DEFAULT_MAX_APPLES

    def __post_init__(self) -> None:
        if not 0 <= self.red_hue_low_max < self.yellow_hue_max < self.red_hue_high_min <= 179:
            raise ValueError(
                "hue limits must satisfy 0 <= red low < yellow < red high <= 179"
            )
        if not 0 <= self.min_saturation <= 255:
            raise ValueError("min_saturation must be between 0 and 255")
        if not 0 <= self.min_value <= self.max_value <= 255:
            raise ValueError("min/max value must satisfy 0 <= min <= max <= 255")
        if not 0.0 < self.min_area_ratio < self.max_area_ratio < 1.0:
            raise ValueError("area ratios must satisfy 0 < min < max < 1")
        for name, value in (
            ("min_short_to_long_ratio", self.min_short_to_long_ratio),
            ("min_circularity", self.min_circularity),
            ("min_solidity", self.min_solidity),
        ):
            if not 0.0 < value <= 1.0:
                raise ValueError(f"{name} must be in (0, 1]")
        if self.border_margin_px < 0:
            raise ValueError("border_margin_px must be non-negative")
        if self.morphology_kernel < 1 or self.morphology_kernel % 2 == 0:
            raise ValueError("morphology_kernel must be a positive odd integer")
        if self.depth_foreground_margin_mm < 0:
            raise ValueError("depth_foreground_margin_mm must be non-negative")
        if self.max_apples < 1:
            raise ValueError("max_apples must be positive")


def remove_rubber_reflections_by_depth(
    candidate: np.ndarray,
    depth_mm: np.ndarray,
    foreground_margin_mm: int,
) -> np.ndarray:
    """Keep warm pixels measurably closer than their local conveyor surface."""

    mask = np.asarray(candidate)
    depth = np.asarray(depth_mm)
    if mask.ndim != 2 or mask.dtype != np.uint8:
        raise ValueError("candidate must be a uint8 2D mask")
    if depth.shape != mask.shape:
        raise ValueError("depth_mm shape must match the RGB image")
    if foreground_margin_mm < 0:
        raise ValueError("foreground_margin_mm must be non-negative")

    binary = mask > 0
    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        binary.astype(np.uint8), connectivity=8
    )
    filtered = np.zeros(mask.shape, dtype=np.uint8)
    valid_depth = np.isfinite(depth) & (depth > 0)
    for label in range(1, count):
        x = int(stats[label, cv2.CC_STAT_LEFT])
        y = int(stats[label, cv2.CC_STAT_TOP])
        width = int(stats[label, cv2.CC_STAT_WIDTH])
        height = int(stats[label, cv2.CC_STAT_HEIGHT])
        padding = max(width, height)
        x0 = max(0, x - padding)
        y0 = max(0, y - padding)
        x1 = min(mask.shape[1], x + width + padding)
        y1 = min(mask.shape[0], y + height + padding)
        context_valid = valid_depth[y0:y1, x0:x1]
        context_candidate = binary[y0:y1, x0:x1]
        support_values = depth[y0:y1, x0:x1][context_valid & ~context_candidate]
        component = labels == label
        if support_values.size < 16:
            filtered[component] = 255
            continue
        support_depth = float(np.median(support_values))
        foreground = (
            component
            & valid_depth
            & (depth.astype(np.float32) + foreground_margin_mm < support_depth)
        )
        filtered[foreground] = 255
    return filtered


def detect_apple_instances(
    image_bgr: np.ndarray,
    config: AppleInstanceConfig = AppleInstanceConfig(),
    *,
    depth_mm: np.ndarray | None = None,
) -> tuple[AppleInstance, ...]:
    """Return up to ``max_apples`` separated saturated foreground contours."""

    image = np.asarray(image_bgr)
    if image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8:
        raise ValueError("image_bgr must be a uint8 HxWx3 array")
    blurred = cv2.GaussianBlur(image, (5, 5), 0)
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
    warm = cv2.inRange(
        hsv,
        np.asarray((0, config.min_saturation, config.min_value), dtype=np.uint8),
        np.asarray(
            (config.yellow_hue_max, 255, config.max_value), dtype=np.uint8
        ),
    )
    high_red = cv2.inRange(
        hsv,
        np.asarray(
            (config.red_hue_high_min, config.min_saturation, config.min_value),
            dtype=np.uint8,
        ),
        np.asarray((179, 255, config.max_value), dtype=np.uint8),
    )
    # Red wraps around the HSV hue axis.  The contiguous low-hue range also
    # includes orange/yellow skin so a rotating bi-colour apple remains one
    # instance instead of disappearing when its yellow cheek faces the camera.
    candidate = cv2.bitwise_or(warm, high_red)
    if depth_mm is not None:
        candidate = remove_rubber_reflections_by_depth(
            candidate,
            depth_mm,
            config.depth_foreground_margin_mm,
        )
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (config.morphology_kernel, config.morphology_kernel),
    )
    candidate = cv2.morphologyEx(candidate, cv2.MORPH_OPEN, kernel)
    candidate = cv2.morphologyEx(candidate, cv2.MORPH_CLOSE, kernel, iterations=2)
    contours, _ = cv2.findContours(
        candidate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    image_area = image.shape[0] * image.shape[1]
    minimum_area = image_area * config.min_area_ratio
    maximum_area = image_area * config.max_area_ratio
    filtered = []
    for contour in contours:
        area = float(cv2.contourArea(contour))
        if not minimum_area <= area <= maximum_area:
            continue
        x, y, width, height = cv2.boundingRect(contour)
        long_side = max(width, height)
        short_to_long = min(width, height) / long_side if long_side else 0.0
        if short_to_long < config.min_short_to_long_ratio:
            continue
        perimeter = float(cv2.arcLength(contour, True))
        circularity = 4.0 * np.pi * area / (perimeter * perimeter) if perimeter else 0.0
        if circularity < config.min_circularity:
            continue
        hull = cv2.convexHull(contour)
        hull_area = float(cv2.contourArea(hull))
        solidity = area / hull_area if hull_area else 0.0
        if solidity < config.min_solidity:
            continue
        margin = config.border_margin_px
        if (
            x <= margin
            or y <= margin
            or x + width >= image.shape[1] - margin
            or y + height >= image.shape[0] - margin
        ):
            continue
        filtered.append(contour)
    contours = sorted(filtered, key=cv2.contourArea, reverse=True)[
        : config.max_apples
    ]

    instances = []
    for contour in contours:
        hull = cv2.convexHull(contour)
        mask = np.zeros(image.shape[:2], dtype=np.uint8)
        cv2.drawContours(mask, [hull], -1, 255, thickness=cv2.FILLED)
        x, y, width, height = cv2.boundingRect(hull)
        moments = cv2.moments(hull)
        if moments["m00"]:
            center = (
                float(moments["m10"] / moments["m00"]),
                float(moments["m01"] / moments["m00"]),
            )
        else:
            center = (x + width * 0.5, y + height * 0.5)
        instances.append(
            AppleInstance(
                contour=hull,
                mask=mask,
                bounding_box=(int(x), int(y), int(width), int(height)),
                center=center,
                area_px2=float(cv2.contourArea(hull)),
            )
        )
    return tuple(instances)


def process_frame(
    image_bgr: np.ndarray,
    depth_mm: np.ndarray | None = None,
    *,
    instance_config: AppleInstanceConfig = AppleInstanceConfig(),
    damage_config: DamageDetectionConfig = DamageDetectionConfig(),
) -> tuple[np.ndarray, tuple[LiveDamageResult, ...]]:
    """Detect separated apples and return a BGR diagnostic overlay."""

    apples = detect_apple_instances(
        image_bgr,
        instance_config,
        depth_mm=depth_mm,
    )
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    overlay = image_bgr.copy()
    results = []
    colors = ((0, 220, 0), (255, 180, 0))
    for index, apple in enumerate(apples):
        damage = detect_damage(rgb, apple.mask, damage_config)
        apple_pixels = int((apple.mask > 0).sum())
        ratio = float(damage.combined.sum() / apple_pixels) if apple_pixels else 0.0
        result = LiveDamageResult(apple, damage, ratio)
        results.append(result)

        color = colors[index % len(colors)]
        x, y, width, height = apple.bounding_box
        cv2.drawContours(overlay, [apple.contour], -1, color, 2)
        cv2.rectangle(overlay, (x, y), (x + width, y + height), color, 2)
        damage_color = np.zeros_like(overlay)
        damage_color[damage.combined] = (0, 0, 255)
        overlay = cv2.addWeighted(overlay, 1.0, damage_color, 0.55, 0.0)
        cv2.putText(
            overlay,
            f"apple {index + 1} damage_px={ratio * 100.0:.2f}%",
            (x, max(24, y - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2,
            cv2.LINE_AA,
        )
    return overlay, tuple(results)


def _reliable_qos(depth: int = 10) -> Any:
    if QoSProfile is None:
        raise RuntimeError("rclpy is required to construct QoS")
    return QoSProfile(
        reliability=ReliabilityPolicy.RELIABLE,
        durability=DurabilityPolicy.VOLATILE,
        history=HistoryPolicy.KEEP_LAST,
        depth=depth,
    )


class OpenCVDamageLiveNode(Node):  # type: ignore[misc]
    def __init__(self) -> None:
        if _ROS_IMPORT_ERROR is not None:
            raise RuntimeError(
                "ROS 2 Python packages are unavailable; source ROS 2 Jazzy first"
            ) from _ROS_IMPORT_ERROR
        super().__init__(
            "opencv_damage_live_node",
            parameter_overrides=[Parameter("use_sim_time", value=True)],
            automatically_declare_parameters_from_overrides=True,
        )
        defaults = (
            ("rgb_topic", DEFAULT_RGB_TOPIC),
            ("depth_topic", DEFAULT_DEPTH_TOPIC),
            ("display", True),
            ("sync_queue_size", DEFAULT_SYNC_QUEUE_SIZE),
            ("max_apples", DEFAULT_MAX_APPLES),
            ("red_hue_low_max", AppleInstanceConfig().red_hue_low_max),
            ("yellow_hue_max", AppleInstanceConfig().yellow_hue_max),
            ("red_hue_high_min", AppleInstanceConfig().red_hue_high_min),
            ("min_saturation", AppleInstanceConfig().min_saturation),
            ("min_value", AppleInstanceConfig().min_value),
            ("max_value", AppleInstanceConfig().max_value),
            ("min_area_ratio", AppleInstanceConfig().min_area_ratio),
            ("max_area_ratio", AppleInstanceConfig().max_area_ratio),
            (
                "min_short_to_long_ratio",
                AppleInstanceConfig().min_short_to_long_ratio,
            ),
            ("min_circularity", AppleInstanceConfig().min_circularity),
            ("min_solidity", AppleInstanceConfig().min_solidity),
            ("border_margin_px", AppleInstanceConfig().border_margin_px),
            ("morphology_kernel", AppleInstanceConfig().morphology_kernel),
            (
                "depth_foreground_margin_mm",
                AppleInstanceConfig().depth_foreground_margin_mm,
            ),
        )
        for name, default in defaults:
            if not self.has_parameter(name):
                self.declare_parameter(name, default)
        self._display = bool(self.get_parameter("display").value)
        self._instance_config = AppleInstanceConfig(
            red_hue_low_max=int(self.get_parameter("red_hue_low_max").value),
            yellow_hue_max=int(self.get_parameter("yellow_hue_max").value),
            red_hue_high_min=int(self.get_parameter("red_hue_high_min").value),
            min_saturation=int(self.get_parameter("min_saturation").value),
            min_value=int(self.get_parameter("min_value").value),
            max_value=int(self.get_parameter("max_value").value),
            min_area_ratio=float(self.get_parameter("min_area_ratio").value),
            max_area_ratio=float(self.get_parameter("max_area_ratio").value),
            min_short_to_long_ratio=float(
                self.get_parameter("min_short_to_long_ratio").value
            ),
            min_circularity=float(self.get_parameter("min_circularity").value),
            min_solidity=float(self.get_parameter("min_solidity").value),
            border_margin_px=int(self.get_parameter("border_margin_px").value),
            morphology_kernel=int(self.get_parameter("morphology_kernel").value),
            depth_foreground_margin_mm=int(
                self.get_parameter("depth_foreground_margin_mm").value
            ),
            max_apples=int(self.get_parameter("max_apples").value),
        )
        self._synchronizer = ExactRgbDepthSynchronizer(
            queue_size=int(self.get_parameter("sync_queue_size").value)
        )
        self._frames = 0
        rgb_topic = str(self.get_parameter("rgb_topic").value)
        depth_topic = str(self.get_parameter("depth_topic").value)
        self.create_subscription(Image, rgb_topic, self._on_rgb, _reliable_qos())
        self.create_subscription(Image, depth_topic, self._on_depth, _reliable_qos())
        self.get_logger().info(
            f"OpenCV conveyor damage viewer subscribed to {rgb_topic} + {depth_topic}"
        )

    def _on_rgb(self, message: Any) -> None:
        synchronized = self._synchronizer.add("rgb", message)
        if synchronized is not None:
            self._on_rgb_depth(*synchronized)

    def _on_depth(self, message: Any) -> None:
        synchronized = self._synchronizer.add("depth", message)
        if synchronized is not None:
            self._on_rgb_depth(*synchronized)

    def _on_rgb_depth(self, rgb_message: Any, depth_message: Any) -> None:
        started = perf_counter()
        try:
            image = decode_rgb_bgr(rgb_message)
            depth_mm = decode_depth_mm(depth_message)
            if depth_mm.shape != image.shape[:2]:
                raise ValueError("RGB and depth dimensions must match")
            overlay, results = process_frame(
                image,
                depth_mm,
                instance_config=self._instance_config,
            )
        except Exception as exc:
            self.get_logger().error(
                f"Rejected conveyor RGB-D frame: {type(exc).__name__}: {exc}"
            )
            return
        elapsed_ms = (perf_counter() - started) * 1000.0
        self._frames += 1
        if self._frames % 30 == 0:
            ratios = ", ".join(
                f"apple{index + 1}={item.damage_pixel_ratio * 100.0:.2f}%"
                for index, item in enumerate(results)
            ) or "no apples"
            self.get_logger().info(
                f"frame={self._frames} processing={elapsed_ms:.1f}ms {ratios}"
            )
        if self._display:
            cv2.imshow("conv_rsd455 OpenCV damage", overlay)
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
    node = OpenCVDamageLiveNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
