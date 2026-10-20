"""ROS 2 entry point for the GPU PC 2 quality-inspection pipeline."""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass
from enum import Enum
from typing import Any, Generic, TypeVar

from depth_geometry import (
    combine_prediction_with_geometry,
    decode_apple_mask,
    decode_ignore_mask,
)
from inspection_session import (
    InspectionCompletion,
    InspectionContractError,
    InspectionFrame,
    InspectionIdentityMismatch,
    InspectionSession,
    InspectionStore,
)
from predictor import (
    FramePredictor,
    IndexedPrediction,
    PredictorNotConfigured,
    load_measurement_predictor,
    predict_declared_frames,
)
from opencv_color_predictor import decode_rgb
from quality_rules import (
    QualityResult as CoreQualityResult,
    ResultStatus,
    aggregate_measurement_frames,
)


INPUT_TOPIC = "/quality/inspection_images"
COMPLETION_TOPIC = "/quality/inspection_completed"
OUTPUT_TOPIC = "/quality/results"
COLOR_DISTRIBUTION_TOPIC = "/quality/color_distribution_debug/compressed"
RESULT_QOS_DEPTH = 10
COMPLETION_QOS_DEPTH = 10
COLOR_DISTRIBUTION_QOS_DEPTH = 1
# 컨베이어 2는 카메라 3대로 사과의 서로 다른 면을 본다. 한 면만 보고 판정하면
# 3면 구성을 쓰는 의미가 없고 반대편 손상을 놓치므로, 기본값은 전 면 요구다.
# 미달이면 측정값을 만들지 않고 INSUFFICIENT_VIEWS 로 보고한다.
DEFAULT_MIN_VALID_VIEWS = 3
# 등급을 무엇으로 매길지. "size" 는 직경, "color" 는 착색률 기준이다.
# 착색 기준은 color_mask 를 내는 predictor 와 함께 써야 한다
# (model_backend:=opencv_color).
DEFAULT_GRADE_BY = "size"
INPUT_QOS_DEPTH = 6
DEFAULT_CONFIDENCE_THRESHOLD = 0.5
DEFAULT_STALE_SESSION_TIMEOUT_SEC = 3.0
RECENT_FINALIZED_LIMIT = 64
COLOR_DISTRIBUTION_WIDTH = 800
COLOR_DISTRIBUTION_HEIGHT = 520
COLOR_DISTRIBUTION_JPEG_QUALITY = 80
HUE_BIN_WIDTH = 5

try:
    import rclpy
    from appleproj_interfaces.msg import (
        InspectionCompleted,
        InspectionImage,
        QualityResult as QualityResultMessage,
    )
    from rclpy.clock import Clock, ClockType
    from rclpy.node import Node
    from rclpy.parameter import Parameter
    from rclpy.qos import (
        DurabilityPolicy,
        HistoryPolicy,
        QoSProfile,
        ReliabilityPolicy,
    )
    from sensor_msgs.msg import CompressedImage

    _ROS_IMPORT_ERROR: ImportError | None = None
except ImportError as exc:
    rclpy = None  # type: ignore[assignment]
    InspectionCompleted = None  # type: ignore[assignment,misc]
    InspectionImage = None  # type: ignore[assignment,misc]
    QualityResultMessage = None  # type: ignore[assignment,misc]
    CompressedImage = None  # type: ignore[assignment,misc]
    Clock = None  # type: ignore[assignment,misc]
    ClockType = None  # type: ignore[assignment,misc]
    Parameter = None  # type: ignore[assignment,misc]
    QoSProfile = None  # type: ignore[assignment,misc]
    ReliabilityPolicy = None  # type: ignore[assignment,misc]
    DurabilityPolicy = None  # type: ignore[assignment,misc]
    HistoryPolicy = None  # type: ignore[assignment,misc]
    Node = object  # type: ignore[assignment,misc]
    _ROS_IMPORT_ERROR = exc


PredictionT = TypeVar("PredictionT")


class ProcessingState(str, Enum):
    BUFFERING = "BUFFERING"
    DUPLICATE = "DUPLICATE"
    PREDICTED = "PREDICTED"
    PREDICTOR_UNAVAILABLE = "PREDICTOR_UNAVAILABLE"
    TIMEOUT = "TIMEOUT"
    STALE = "STALE"
    RECHECK = "RECHECK"
    FINALIZED = "FINALIZED"


@dataclass(frozen=True)
class ProcessingEvent(Generic[PredictionT]):
    state: ProcessingState
    inspection_id: str
    apple_id: str
    received_count: int
    total_frames: int
    predictions: tuple[IndexedPrediction[PredictionT], ...] = ()
    deadline_time_ns: int | None = None


@dataclass(frozen=True)
class ColorDistribution:
    """Hue and named-colour counts from measurable apple-surface pixels."""

    hue_histogram: tuple[int, ...]
    category_counts: tuple[tuple[str, int], ...]
    apple_pixels: int
    valid_pixels: int
    ignored_pixels: int
    frame_indices: tuple[int, ...]

    @property
    def category_ratios(self) -> tuple[tuple[str, float], ...]:
        if self.valid_pixels <= 0:
            return tuple((name, 0.0) for name, _ in self.category_counts)
        return tuple(
            (name, count / self.valid_pixels)
            for name, count in self.category_counts
        )


def measure_color_distribution(
    frames: tuple[InspectionFrame, ...],
    frame_indices: tuple[int, ...] | list[int],
) -> ColorDistribution:
    """Aggregate colour only from ``apple_mask AND NOT ignore_mask`` pixels."""

    import cv2
    import numpy as np

    selected = set(int(index) for index in frame_indices)
    ordered = tuple(frame for frame in frames if frame.frame_index in selected)
    hue_parts = []
    apple_pixels = 0
    valid_pixels = 0
    ignored_pixels = 0
    used_indices = []

    for frame in ordered:
        rgb = decode_rgb(frame)
        apple_mask = decode_apple_mask(frame)
        ignore_mask = decode_ignore_mask(frame)
        if rgb.shape[:2] != apple_mask.shape or apple_mask.shape != ignore_mask.shape:
            raise InspectionContractError(
                "RGB, apple_mask and ignore_mask dimensions must match for colour graph"
            )
        valid = apple_mask & ~ignore_mask
        apple_pixels += int(apple_mask.sum())
        ignored_pixels += int((apple_mask & ignore_mask).sum())
        frame_valid_pixels = int(valid.sum())
        if frame_valid_pixels <= 0:
            continue
        hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
        hue_parts.append(hsv[..., 0][valid])
        valid_pixels += frame_valid_pixels
        used_indices.append(frame.frame_index)

    if not hue_parts:
        raise InspectionContractError(
            "no measurable apple pixels remain after applying ignore_mask"
        )

    hues = np.concatenate(hue_parts).astype(np.uint8, copy=False)
    bin_edges = np.arange(0, 181, HUE_BIN_WIDTH, dtype=np.int16)
    histogram, _ = np.histogram(hues, bins=bin_edges)
    counts = (
        ("RED", int(((hues <= 10) | (hues >= 170)).sum())),
        ("ORANGE", int(((hues >= 11) & (hues <= 25)).sum())),
        ("YELLOW", int(((hues >= 26) & (hues <= 35)).sum())),
        ("GREEN", int(((hues >= 36) & (hues <= 85)).sum())),
        ("OTHER", int(((hues >= 86) & (hues <= 169)).sum())),
    )
    return ColorDistribution(
        hue_histogram=tuple(int(value) for value in histogram),
        category_counts=counts,
        apple_pixels=apple_pixels,
        valid_pixels=valid_pixels,
        ignored_pixels=ignored_pixels,
        frame_indices=tuple(used_indices),
    )


def _enum_text(value: Any, fallback: str = "NONE") -> str:
    if value is None:
        return fallback
    return str(getattr(value, "value", value))


def render_color_distribution_jpeg(
    distribution: ColorDistribution,
    event: ProcessingEvent[Any],
    result: CoreQualityResult,
) -> bytes:
    """Render a compact graph without matplotlib or per-frame publication."""

    import cv2
    import numpy as np

    width = COLOR_DISTRIBUTION_WIDTH
    height = COLOR_DISTRIBUTION_HEIGHT
    canvas = np.full((height, width, 3), 247, dtype=np.uint8)
    ink = (35, 35, 35)
    muted = (105, 105, 105)
    cv2.putText(
        canvas,
        "Apple colour distribution",
        (24, 32),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.78,
        ink,
        2,
        cv2.LINE_AA,
    )
    confidence = "N/A" if result.confidence is None else f"{result.confidence:.3f}"
    ratio = (
        "N/A"
        if result.measurements is None or result.measurements.color_ratio is None
        else f"{result.measurements.color_ratio:.1%}"
    )
    cv2.putText(
        canvas,
        f"inspection={event.inspection_id}  apple={event.apple_id}",
        (24, 58),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        muted,
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        canvas,
        f"status={_enum_text(result.status)}  grade={_enum_text(result.grade)}  "
        f"target-red={ratio}  confidence={confidence}  frames={list(result.frames_used)}",
        (24, 82),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.47,
        ink,
        1,
        cv2.LINE_AA,
    )

    plot_x, plot_y, plot_w, plot_h = 48, 116, 704, 150
    cv2.putText(
        canvas,
        "Hue histogram (OpenCV H: 0-179)",
        (plot_x, plot_y - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        ink,
        1,
        cv2.LINE_AA,
    )
    cv2.rectangle(canvas, (plot_x, plot_y), (plot_x + plot_w, plot_y + plot_h), muted, 1)
    maximum = max(distribution.hue_histogram, default=0)
    bar_width = plot_w / max(1, len(distribution.hue_histogram))
    for index, count in enumerate(distribution.hue_histogram):
        bar_height = 0 if maximum <= 0 else int((count / maximum) * (plot_h - 4))
        hue = min(179, index * HUE_BIN_WIDTH + HUE_BIN_WIDTH // 2)
        hsv_colour = np.uint8([[[hue, 230, 230]]])
        bgr = tuple(int(value) for value in cv2.cvtColor(hsv_colour, cv2.COLOR_HSV2BGR)[0, 0])
        x1 = plot_x + int(index * bar_width) + 1
        x2 = plot_x + max(int((index + 1) * bar_width), int(index * bar_width) + 1)
        cv2.rectangle(
            canvas,
            (x1, plot_y + plot_h - bar_height - 1),
            (x2, plot_y + plot_h - 1),
            bgr,
            -1,
        )
    for hue_tick in (0, 30, 60, 90, 120, 150, 179):
        tick_x = plot_x + int((hue_tick / 179.0) * plot_w)
        cv2.putText(
            canvas,
            str(hue_tick),
            (tick_x - 8, plot_y + plot_h + 18),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.36,
            muted,
            1,
            cv2.LINE_AA,
        )

    category_colours = {
        "RED": (40, 40, 220),
        "ORANGE": (30, 135, 245),
        "YELLOW": (25, 210, 235),
        "GREEN": (65, 175, 65),
        "OTHER": (155, 105, 120),
    }
    bar_x, bar_width_px = 132, 570
    for row, (name, ratio_value) in enumerate(distribution.category_ratios):
        y = 316 + row * 30
        cv2.putText(
            canvas,
            name,
            (48, y + 15),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            ink,
            1,
            cv2.LINE_AA,
        )
        cv2.rectangle(canvas, (bar_x, y), (bar_x + bar_width_px, y + 17), (220, 220, 220), -1)
        filled = int(bar_width_px * ratio_value)
        if filled > 0:
            cv2.rectangle(
                canvas,
                (bar_x, y),
                (bar_x + filled, y + 17),
                category_colours[name],
                -1,
            )
        cv2.putText(
            canvas,
            f"{ratio_value:6.1%}",
            (710, y + 15),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            ink,
            1,
            cv2.LINE_AA,
        )

    cv2.putText(
        canvas,
        f"measurable={distribution.valid_pixels:,} px   "
        f"ignored={distribution.ignored_pixels:,} px   "
        f"apple-mask total={distribution.apple_pixels:,} px",
        (48, 493),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        ink,
        1,
        cv2.LINE_AA,
    )
    encoded, payload = cv2.imencode(
        ".jpg",
        canvas,
        (cv2.IMWRITE_JPEG_QUALITY, COLOR_DISTRIBUTION_JPEG_QUALITY),
    )
    if not encoded:
        raise RuntimeError("failed to encode colour-distribution JPEG")
    return bytes(payload)


class DiameterOnlyPredictor:
    """No-model predictor used by the approved size-only MVP."""

    def predict(self, frame: InspectionFrame) -> None:
        del frame
        return None


def _header_signature(header: Any) -> tuple[int, int, str]:
    stamp = header.stamp
    return int(stamp.sec), int(stamp.nanosec), str(header.frame_id)


def _stamp_to_ns(stamp: Any) -> int:
    return int(stamp.sec) * 1_000_000_000 + int(stamp.nanosec)


def inspection_frame_from_message(message: Any) -> InspectionFrame:
    """Validate and convert the synchronized custom RGB-D message."""

    components = (
        ("InspectionImage", message.header),
        ("CompressedImage", message.image.header),
        ("apple_mask", message.apple_mask.header),
        ("ignore_mask", message.ignore_mask.header),
        ("aligned_depth", message.aligned_depth.header),
        ("CameraInfo", message.camera_info.header),
    )
    signatures = [(name, _header_signature(header)) for name, header in components]
    expected = signatures[0][1]
    mismatched = [name for name, signature in signatures[1:] if signature != expected]
    if mismatched:
        raise InspectionContractError(
            "all InspectionImage component headers must have identical stamp and frame_id; "
            f"mismatched={mismatched}"
        )

    return InspectionFrame(
        inspection_id=message.inspection_id,
        apple_id=message.apple_id,
        frame_index=int(message.frame_index),
        total_frames=int(message.total_frames),
        image_data=bytes(message.image.data),
        image_format=str(message.image.format),
        apple_mask_data=bytes(message.apple_mask.data),
        apple_mask_format=str(message.apple_mask.format),
        ignore_mask_data=bytes(message.ignore_mask.data),
        ignore_mask_format=str(message.ignore_mask.format),
        depth_data=bytes(message.aligned_depth.data),
        depth_format=str(message.aligned_depth.format),
        camera_width=int(message.camera_info.width),
        camera_height=int(message.camera_info.height),
        camera_k=tuple(float(value) for value in message.camera_info.k),
        camera_p=tuple(float(value) for value in message.camera_info.p),
        stamp_ns=_stamp_to_ns(message.header.stamp),
        frame_id=str(message.header.frame_id),
    )


def inspection_completion_from_message(message: Any) -> InspectionCompletion:
    return InspectionCompletion(
        inspection_id=str(message.inspection_id),
        apple_id=str(message.apple_id),
        total_frames=int(message.total_frames),
        roi_exit_time_ns=_stamp_to_ns(message.header.stamp),
        frame_id=str(message.header.frame_id),
    )


class InspectionCoordinator(Generic[PredictionT]):
    """Own inspection lifecycle without depending on ROS message classes."""

    def __init__(self, predictor: FramePredictor[PredictionT]) -> None:
        self._store = InspectionStore()
        self._predictor = predictor
        self._last_wall_activity_ns: dict[str, int] = {}
        self._recent_finalized: deque[str] = deque(maxlen=RECENT_FINALIZED_LIMIT)
        self._recent_finalized_set: set[str] = set()

    @property
    def store(self) -> InspectionStore:
        return self._store

    def _event(
        self,
        state: ProcessingState,
        session: InspectionSession,
        predictions: tuple[IndexedPrediction[PredictionT], ...] = (),
    ) -> ProcessingEvent[PredictionT]:
        deadline = session.completion.deadline_time_ns if session.completion else None
        return ProcessingEvent(
            state=state,
            inspection_id=session.inspection_id,
            apple_id=session.apple_id,
            received_count=session.received_count,
            total_frames=session.total_frames,
            predictions=predictions,
            deadline_time_ns=deadline,
        )

    def _is_recently_finalized(self, inspection_id: str) -> bool:
        return inspection_id in self._recent_finalized_set

    def _touch(self, inspection_id: str, wall_time_ns: int | None) -> None:
        self._last_wall_activity_ns[inspection_id] = (
            time.monotonic_ns() if wall_time_ns is None else wall_time_ns
        )

    def handle_frame(
        self,
        frame: InspectionFrame,
        simulation_time_ns: int,
        wall_time_ns: int | None = None,
    ) -> ProcessingEvent[PredictionT]:
        if self._is_recently_finalized(frame.inspection_id):
            return ProcessingEvent(
                ProcessingState.FINALIZED,
                frame.inspection_id,
                frame.apple_id,
                0,
                frame.total_frames,
            )
        acceptance = self._store.accept(frame)
        self._touch(frame.inspection_id, wall_time_ns)
        if not acceptance.is_new_frame:
            return self._event(ProcessingState.DUPLICATE, acceptance.session)
        return self._maybe_predict(acceptance.session, simulation_time_ns)

    def handle_completion(
        self,
        completion: InspectionCompletion,
        simulation_time_ns: int,
        wall_time_ns: int | None = None,
    ) -> ProcessingEvent[PredictionT]:
        if self._is_recently_finalized(completion.inspection_id):
            return ProcessingEvent(
                ProcessingState.FINALIZED,
                completion.inspection_id,
                completion.apple_id,
                0,
                completion.total_frames,
                deadline_time_ns=completion.deadline_time_ns,
            )
        session = self._store.complete(completion)
        self._touch(completion.inspection_id, wall_time_ns)
        return self._maybe_predict(session, simulation_time_ns)

    def _maybe_predict(
        self,
        session: InspectionSession,
        simulation_time_ns: int,
    ) -> ProcessingEvent[PredictionT]:
        if session.completion is None:
            return self._event(ProcessingState.BUFFERING, session)
        if session.deadline_reached(simulation_time_ns):
            return self._event(ProcessingState.TIMEOUT, session)
        if not session.has_all_declared_frames:
            return self._event(ProcessingState.BUFFERING, session)
        try:
            predictions = predict_declared_frames(session, self._predictor)
        except PredictorNotConfigured:
            return self._event(ProcessingState.PREDICTOR_UNAVAILABLE, session)
        return self._event(ProcessingState.PREDICTED, session, predictions)

    def expired(self, simulation_time_ns: int) -> tuple[ProcessingEvent[PredictionT], ...]:
        return tuple(
            self._event(ProcessingState.TIMEOUT, session)
            for session in self._store.sessions
            if session.completion is not None
            and session.deadline_reached(simulation_time_ns)
        )

    def stale(
        self,
        wall_time_ns: int,
        timeout_ns: int,
    ) -> tuple[ProcessingEvent[PredictionT], ...]:
        return tuple(
            self._event(ProcessingState.STALE, session)
            for session in self._store.sessions
            if session.completion is None
            if wall_time_ns - self._last_wall_activity_ns.get(
                session.inspection_id,
                wall_time_ns,
            ) >= timeout_ns
        )

    def finalize(self, inspection_id: str) -> None:
        if not inspection_id:
            return
        self._store.pop(inspection_id)
        self._last_wall_activity_ns.pop(inspection_id, None)
        if inspection_id in self._recent_finalized_set:
            return
        if len(self._recent_finalized) == self._recent_finalized.maxlen:
            expired = self._recent_finalized.popleft()
            self._recent_finalized_set.discard(expired)
        self._recent_finalized.append(inspection_id)
        self._recent_finalized_set.add(inspection_id)

    def identity_mismatch_event(
        self,
        inspection_id: str,
        fallback_apple_id: str,
        fallback_total_frames: int,
    ) -> ProcessingEvent[PredictionT]:
        """Describe an ID conflict using the original session identity."""
        session = self._store.get(inspection_id)
        if session is not None:
            return self._event(ProcessingState.RECHECK, session)
        return ProcessingEvent(
            state=ProcessingState.RECHECK,
            inspection_id=inspection_id,
            apple_id=fallback_apple_id,
            received_count=0,
            total_frames=fallback_total_frames,
        )


def _reliable_qos(depth: int) -> Any:
    if QoSProfile is None:
        raise RuntimeError("rclpy is required to construct QoS profiles")
    return QoSProfile(
        reliability=ReliabilityPolicy.RELIABLE,
        durability=DurabilityPolicy.VOLATILE,
        history=HistoryPolicy.KEEP_LAST,
        depth=depth,
    )


def make_input_qos() -> Any:
    return _reliable_qos(INPUT_QOS_DEPTH)


def make_completion_qos() -> Any:
    return _reliable_qos(COMPLETION_QOS_DEPTH)


def make_result_qos() -> Any:
    return _reliable_qos(RESULT_QOS_DEPTH)


def make_color_distribution_qos() -> Any:
    if QoSProfile is None:
        raise RuntimeError("rclpy is required to construct QoS profiles")
    return QoSProfile(
        reliability=ReliabilityPolicy.BEST_EFFORT,
        durability=DurabilityPolicy.VOLATILE,
        history=HistoryPolicy.KEEP_LAST,
        depth=COLOR_DISTRIBUTION_QOS_DEPTH,
    )


class QualityInspectionNode(Node):  # type: ignore[misc]
    def __init__(self, predictor: FramePredictor[Any] | None = None) -> None:
        if _ROS_IMPORT_ERROR is not None:
            raise RuntimeError(
                "ROS 2 Python packages are unavailable; source ROS 2 Jazzy and "
                "the built appleproj_interfaces workspace first"
            ) from _ROS_IMPORT_ERROR

        super().__init__(
            "quality_inspection_node",
            parameter_overrides=[Parameter("use_sim_time", value=True)],
            automatically_declare_parameters_from_overrides=True,
        )
        for name, default in (
            ("model_path", ""),
            ("model_backend", "auto"),
            ("confidence_threshold", DEFAULT_CONFIDENCE_THRESHOLD),
            ("stale_session_timeout_sec", DEFAULT_STALE_SESSION_TIMEOUT_SEC),
            ("min_valid_views", DEFAULT_MIN_VALID_VIEWS),
            ("grade_by", DEFAULT_GRADE_BY),
        ):
            if not self.has_parameter(name):
                self.declare_parameter(name, default)

        configured_predictor = predictor
        if configured_predictor is None:
            model_path = str(self.get_parameter("model_path").value)
            backend = str(self.get_parameter("model_backend").value)
            # opencv_color 는 학습 모델 파일이 없으므로 model_path 없이도 고른다.
            if backend == "opencv_color" or model_path:
                configured_predictor = load_measurement_predictor(
                    model_path, backend=backend
                )
            else:
                configured_predictor = DiameterOnlyPredictor()
        self._confidence_threshold = float(self.get_parameter("confidence_threshold").value)
        self._min_valid_views = int(self.get_parameter("min_valid_views").value)
        if self._min_valid_views < 1:
            raise ValueError("min_valid_views must be positive")
        self._grade_by = str(self.get_parameter("grade_by").value)
        if self._grade_by not in ("size", "color"):
            raise ValueError("grade_by must be 'size' or 'color'")
        self._stale_timeout_ns = int(
            float(self.get_parameter("stale_session_timeout_sec").value) * 1_000_000_000
        )
        self._coordinator = InspectionCoordinator(configured_predictor)
        self._result_publisher = self.create_publisher(
            QualityResultMessage,
            OUTPUT_TOPIC,
            make_result_qos(),
        )
        self._color_distribution_publisher = self.create_publisher(
            CompressedImage,
            COLOR_DISTRIBUTION_TOPIC,
            make_color_distribution_qos(),
        )
        self._inspection_subscription = self.create_subscription(
            InspectionImage,
            INPUT_TOPIC,
            self._on_inspection_image,
            make_input_qos(),
        )
        self._completion_subscription = self.create_subscription(
            InspectionCompleted,
            COMPLETION_TOPIC,
            self._on_inspection_completed,
            make_completion_qos(),
        )
        self._deadline_timer = self.create_timer(
            0.05,
            self._on_deadline_timer,
            clock=self.get_clock(),
        )
        self._steady_clock = Clock(clock_type=ClockType.STEADY_TIME)
        self._stale_timer = self.create_timer(
            0.5,
            self._on_stale_timer,
            clock=self._steady_clock,
        )
        self.get_logger().info(
            f"GPU PC 2 quality node ready: {INPUT_TOPIC} + {COMPLETION_TOPIC} "
            f"-> {OUTPUT_TOPIC}; use_sim_time=true"
        )
        self.get_logger().info(
            f"grading by {self._grade_by}, "
            f"min_valid_views={self._min_valid_views}, "
            f"predictor={type(configured_predictor).__name__}"
        )
        self.get_logger().info(
            f"colour graph on completion: {COLOR_DISTRIBUTION_TOPIC}"
        )

    def _simulation_time_ns(self) -> int:
        return int(self.get_clock().now().nanoseconds)

    def _on_inspection_image(self, message: Any) -> None:
        try:
            frame = inspection_frame_from_message(message)
            event = self._coordinator.handle_frame(
                frame,
                self._simulation_time_ns(),
            )
        except InspectionIdentityMismatch as exc:
            self._handle_identity_mismatch(message, exc)
            return
        except InspectionContractError as exc:
            self.get_logger().error(f"Rejected InspectionImage contract: {exc}")
            return
        except Exception as exc:
            self._coordinator.finalize(str(getattr(message, "inspection_id", "")))
            self.get_logger().error(f"Inspection input failed: {exc}")
            return
        self._handle_event(event)

    def _on_inspection_completed(self, message: Any) -> None:
        try:
            completion = inspection_completion_from_message(message)
            event = self._coordinator.handle_completion(
                completion,
                self._simulation_time_ns(),
            )
        except InspectionIdentityMismatch as exc:
            self._handle_identity_mismatch(message, exc)
            return
        except InspectionContractError as exc:
            self.get_logger().error(f"Rejected InspectionCompleted contract: {exc}")
            return
        except Exception as exc:
            self._coordinator.finalize(str(getattr(message, "inspection_id", "")))
            self.get_logger().error(f"Inspection completion failed: {exc}")
            return
        self._handle_event(event)

    def _handle_event(self, event: ProcessingEvent[Any]) -> None:
        if event.state is ProcessingState.BUFFERING:
            self.get_logger().debug(
                f"Buffering {event.inspection_id}: "
                f"{event.received_count}/{event.total_frames} frames"
            )
            return
        if event.state in (ProcessingState.DUPLICATE, ProcessingState.FINALIZED):
            self.get_logger().warning(
                f"Ignored duplicate/finalized input for inspection {event.inspection_id}"
            )
            return
        if event.state is ProcessingState.PREDICTOR_UNAVAILABLE:
            session = self._coordinator.store.get(event.inspection_id)
            frame_indices = session.frame_indices if session is not None else ()
            result = CoreQualityResult(
                None,
                ResultStatus.UNCLASSIFIED,
                None,
                None,
                frame_indices,
            )
            self._publish_result(event, result)
            self._coordinator.finalize(event.inspection_id)
            return
        if event.state is ProcessingState.TIMEOUT:
            self._publish_timeout(event)
            return
        if event.state is ProcessingState.STALE:
            self.get_logger().warning(
                f"Dropped stale inspection without completion: {event.inspection_id}"
            )
            self._coordinator.finalize(event.inspection_id)
            return
        if event.state is ProcessingState.RECHECK:
            result = CoreQualityResult(
                None,
                ResultStatus.RECHECK,
                None,
                None,
                (),
            )
            self._publish_result(event, result)
            self._coordinator.finalize(event.inspection_id)
            return
        if event.state is ProcessingState.PREDICTED:
            self._finalize_predictions(event)

    def _handle_identity_mismatch(self, message: Any, exc: Exception) -> None:
        inspection_id = str(getattr(message, "inspection_id", ""))
        event = self._coordinator.identity_mismatch_event(
            inspection_id,
            str(getattr(message, "apple_id", "")),
            int(getattr(message, "total_frames", 0)),
        )
        self.get_logger().error(
            f"INSPECTION_IDENTITY_MISMATCH inspection_id={inspection_id} "
            f"received_apple_id={getattr(message, 'apple_id', '')} error={exc}"
        )
        self._handle_event(event)

    def _finalize_predictions(self, event: ProcessingEvent[Any]) -> None:
        session = self._coordinator.store.get(event.inspection_id)
        if session is None:
            return
        measurements = []
        successful_indices = []
        failed_indices = []
        for indexed in event.predictions:
            if not indexed.succeeded:
                failed_indices.append(indexed.frame_index)
                self.get_logger().warning(
                    f"FRAME_INFERENCE_FAILED inspection_id={event.inspection_id} "
                    f"frame_index={indexed.frame_index} error_type={indexed.error_type} "
                    f"error={indexed.error_message}"
                )
                continue

            frame = next(
                item for item in session.ordered_frames
                if item.frame_index == indexed.frame_index
            )
            try:
                measurements.append(
                    combine_prediction_with_geometry(frame, indexed.value)
                )
                successful_indices.append(indexed.frame_index)
            except Exception as exc:
                failed_indices.append(indexed.frame_index)
                self.get_logger().warning(
                    f"FRAME_MEASUREMENT_FAILED inspection_id={event.inspection_id} "
                    f"frame_index={indexed.frame_index} error_type={type(exc).__name__} "
                    f"error={exc}"
                )

        if (
            event.deadline_time_ns is not None
            and self._simulation_time_ns() >= event.deadline_time_ns
        ):
            self.get_logger().warning(
                f"LATE_RESULT inspection_id={event.inspection_id} "
                f"successful_frame_indices={successful_indices} "
                f"failed_frame_indices={failed_indices}"
            )
            self._publish_timeout(event)
            return

        result = aggregate_measurement_frames(
            measurements,
            successful_indices,
            confidence_threshold=self._confidence_threshold,
            min_valid_views=self._min_valid_views,
            grade_by=self._grade_by,
        )
        self._publish_result(event, result)
        if self._grade_by == "color" and successful_indices:
            self._publish_color_distribution(
                event,
                result,
                session,
                successful_indices,
            )
        self._coordinator.finalize(event.inspection_id)

    def _publish_color_distribution(
        self,
        event: ProcessingEvent[Any],
        result: CoreQualityResult,
        session: InspectionSession,
        successful_indices: list[int],
    ) -> None:
        try:
            distribution = measure_color_distribution(
                session.ordered_frames,
                successful_indices,
            )
            payload = render_color_distribution_jpeg(distribution, event, result)
            message = CompressedImage()
            message.header.stamp = self.get_clock().now().to_msg()
            message.header.frame_id = "quality_grading"
            message.format = "bgr8; jpeg compressed bgr8"
            message.data = payload
            self._color_distribution_publisher.publish(message)
            self.get_logger().info(
                f"Published colour graph for {event.inspection_id}: "
                f"frames={list(distribution.frame_indices)}, "
                f"valid={distribution.valid_pixels}, "
                f"ignored={distribution.ignored_pixels}"
            )
        except Exception as exc:
            # 시각화 실패가 품질 판정과 /quality/results 발행을 막아서는 안 된다.
            self.get_logger().warning(
                f"COLOR_GRAPH_FAILED inspection_id={event.inspection_id} "
                f"error_type={type(exc).__name__} error={exc}"
            )

    def _publish_timeout(self, event: ProcessingEvent[Any]) -> None:
        result = CoreQualityResult(
            None,
            ResultStatus.TIMEOUT,
            None,
            None,
            (),
        )
        self._publish_result(event, result)
        self._coordinator.finalize(event.inspection_id)

    def _on_deadline_timer(self) -> None:
        for event in self._coordinator.expired(self._simulation_time_ns()):
            self._handle_event(event)

    def _on_stale_timer(self) -> None:
        now_ns = time.monotonic_ns()
        for event in self._coordinator.stale(now_ns, self._stale_timeout_ns):
            self._handle_event(event)

    def _publish_result(self, event: ProcessingEvent[Any], result: CoreQualityResult) -> None:
        message = QualityResultMessage()
        now = self.get_clock().now().to_msg()
        message.header.stamp = now
        message.header.frame_id = "quality_grading"
        message.inspection_id = event.inspection_id
        message.apple_id = event.apple_id
        message.grade = (
            int(getattr(QualityResultMessage, result.grade.value))
            if result.grade is not None
            else 0
        )
        message.confidence = (
            float(result.confidence) if result.confidence is not None else float("nan")
        )
        measurements = result.measurements
        message.color_ratio = (
            float(measurements.color_ratio)
            if measurements is not None and measurements.color_ratio is not None
            else float("nan")
        )
        message.diameter_mm = (
            float(measurements.diameter_mm)
            if measurements is not None and measurements.diameter_mm is not None
            else float("nan")
        )
        message.damage_area_cm2 = (
            float(measurements.damage_area_cm2)
            if measurements is not None and measurements.damage_area_cm2 is not None
            else float("nan")
        )
        message.frames_used = len(result.frames_used)
        message.frame_indices = list(result.frames_used)
        message.result_timestamp = now
        message.status = int(getattr(QualityResultMessage, result.status.value))
        self._result_publisher.publish(message)
        self.get_logger().info(
            f"Published {result.status.value} for {event.inspection_id}: "
            f"grade={result.grade.value if result.grade else 'NONE'}, "
            f"frames={list(result.frames_used)}"
        )


def main(args: list[str] | None = None) -> None:
    if rclpy is None:
        raise RuntimeError(
            "ROS 2 Python packages are unavailable; source ROS 2 Jazzy and the workspace"
        ) from _ROS_IMPORT_ERROR
    rclpy.init(args=args)
    node = QualityInspectionNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
