from __future__ import annotations

import io
import unittest
from types import SimpleNamespace

import numpy as np
from PIL import Image

from inspection_session import (
    INSPECTION_ROI_FRAME,
    InspectionContractError,
    InspectionFrame,
    InspectionIdentityMismatch,
)
from quality_inspection_node import (
    COMPLETION_QOS_DEPTH,
    COMPLETION_TOPIC,
    COLOR_DISTRIBUTION_HEIGHT,
    COLOR_DISTRIBUTION_QOS_DEPTH,
    COLOR_DISTRIBUTION_TOPIC,
    COLOR_DISTRIBUTION_WIDTH,
    DEFAULT_CONFIDENCE_THRESHOLD,
    INPUT_QOS_DEPTH,
    INPUT_TOPIC,
    OUTPUT_TOPIC,
    RESULT_QOS_DEPTH,
    InspectionCoordinator,
    ProcessingEvent,
    ProcessingState,
    inspection_completion_from_message,
    inspection_frame_from_message,
    measure_color_distribution,
    render_color_distribution_jpeg,
)
from quality_rules import AppleMeasurements, Grade, QualityResult, ResultStatus


CAMERA_K = [500.0, 0.0, 50.0, 0.0, 500.0, 50.0, 0.0, 0.0, 1.0]
CAMERA_P = [500.0, 0.0, 50.0, 0.0, 0.0, 500.0, 50.0, 0.0, 0.0, 0.0, 1.0, 0.0]


def header(sec: int = 10, nanosec: int = 20, frame_id: str = "quality_camera_top_optical_frame"):
    return SimpleNamespace(
        stamp=SimpleNamespace(sec=sec, nanosec=nanosec),
        frame_id=frame_id,
    )


def compressed(data: bytes, image_format: str):
    return SimpleNamespace(
        header=header(),
        data=bytearray(data),
        format=image_format,
    )


def make_message(
    frame_index: int,
    *,
    total_frames: int = 2,
    inspection_id: str = "inspection-ros-001",
    apple_id: str = "apple-ros-001",
):
    return SimpleNamespace(
        header=header(),
        inspection_id=inspection_id,
        apple_id=apple_id,
        frame_index=frame_index,
        total_frames=total_frames,
        image=compressed(f"jpeg-{frame_index}".encode(), "rgb8; jpeg compressed bgr8"),
        apple_mask=compressed(b"png-mask", "mono8; png"),
        ignore_mask=compressed(b"png-ignore", "mono8; png"),
        aligned_depth=compressed(b"png-depth", "16UC1; compressedDepth png"),
        camera_info=SimpleNamespace(
            header=header(),
            width=100,
            height=100,
            k=CAMERA_K,
            p=CAMERA_P,
        ),
    )


def make_completion_message(
    *,
    total_frames: int = 2,
    inspection_id: str = "inspection-ros-001",
    apple_id: str = "apple-ros-001",
    sec: int = 10,
):
    return SimpleNamespace(
        # ROI exit is a conveyor event, so completions use the ROI frame
        # rather than any one of the three camera optical frames.
        header=header(sec=sec, nanosec=0, frame_id=INSPECTION_ROI_FRAME),
        inspection_id=inspection_id,
        apple_id=apple_id,
        total_frames=total_frames,
    )


def png_bytes(array: np.ndarray) -> bytes:
    stream = io.BytesIO()
    Image.fromarray(array).save(stream, format="PNG")
    return stream.getvalue()


def make_colour_frame() -> InspectionFrame:
    # Each column is one named hue category. The GREEN pixel is ignored.
    rgb = np.array(
        [[
            [255, 0, 0],
            [255, 128, 0],
            [255, 255, 0],
            [0, 255, 0],
            [0, 0, 255],
        ]],
        dtype=np.uint8,
    )
    apple_mask = np.full((1, 5), 255, dtype=np.uint8)
    ignore_mask = np.array([[0, 0, 0, 255, 0]], dtype=np.uint8)
    depth = np.full((1, 5), 600, dtype=np.uint16)
    return InspectionFrame(
        inspection_id="inspection-colour-001",
        apple_id="apple-colour-001",
        frame_index=0,
        total_frames=1,
        image_data=png_bytes(rgb),
        image_format="rgb8; png",
        apple_mask_data=png_bytes(apple_mask),
        apple_mask_format="mono8; png",
        ignore_mask_data=png_bytes(ignore_mask),
        ignore_mask_format="mono8; png",
        depth_data=png_bytes(depth),
        depth_format="16UC1; compressedDepth png",
        camera_width=5,
        camera_height=1,
        camera_k=tuple(CAMERA_K),
        camera_p=tuple(CAMERA_P),
        stamp_ns=10_000_000_000,
        frame_id="quality_camera_top_optical_frame",
    )


class MessageAdapterTest(unittest.TestCase):
    def test_converts_synchronized_rgbd_message(self) -> None:
        frame = inspection_frame_from_message(make_message(1))
        self.assertEqual(frame.frame_index, 1)
        self.assertEqual(frame.image_data, b"jpeg-1")
        self.assertEqual(frame.apple_mask_data, b"png-mask")
        self.assertEqual(frame.ignore_mask_data, b"png-ignore")
        self.assertEqual(frame.depth_data, b"png-depth")
        self.assertEqual(frame.camera_width, 100)
        self.assertEqual(frame.frame_id, "quality_camera_top_optical_frame")

    def test_rejects_mismatched_component_header(self) -> None:
        message = make_message(0)
        message.aligned_depth.header = header(nanosec=21)
        with self.assertRaises(InspectionContractError):
            inspection_frame_from_message(message)

    def test_rejects_empty_component_data(self) -> None:
        message = make_message(0)
        message.apple_mask.data = bytearray()
        with self.assertRaises(InspectionContractError):
            inspection_frame_from_message(message)

        message = make_message(0)
        message.ignore_mask.data = bytearray()
        with self.assertRaises(InspectionContractError):
            inspection_frame_from_message(message)

    def test_converts_completion_header_stamp_to_deadline_source(self) -> None:
        completion = inspection_completion_from_message(make_completion_message(sec=12))
        self.assertEqual(completion.roi_exit_time_ns, 12_000_000_000)
        self.assertEqual(completion.frame_id, INSPECTION_ROI_FRAME)


class CoordinatorLifecycleTest(unittest.TestCase):
    def test_all_frames_wait_for_completion_before_prediction(self) -> None:
        class IndexPredictor:
            def predict(self, frame: InspectionFrame) -> str:
                return f"prediction-{frame.frame_index}"

        coordinator = InspectionCoordinator(IndexPredictor())
        first = coordinator.handle_frame(
            inspection_frame_from_message(make_message(1)),
            9_000_000_000,
        )
        second = coordinator.handle_frame(
            inspection_frame_from_message(make_message(0)),
            9_000_000_000,
        )
        complete = coordinator.handle_completion(
            inspection_completion_from_message(make_completion_message()),
            10_100_000_000,
        )
        self.assertEqual(first.state, ProcessingState.BUFFERING)
        self.assertEqual(second.state, ProcessingState.BUFFERING)
        self.assertEqual(complete.state, ProcessingState.PREDICTED)
        self.assertEqual(
            tuple(item.value for item in complete.predictions),
            ("prediction-0", "prediction-1"),
        )

    def test_completion_before_frames_is_supported(self) -> None:
        class IndexPredictor:
            def predict(self, frame: InspectionFrame) -> int:
                return frame.frame_index

        coordinator = InspectionCoordinator(IndexPredictor())
        waiting = coordinator.handle_completion(
            inspection_completion_from_message(make_completion_message()),
            10_000_000_000,
        )
        coordinator.handle_frame(
            inspection_frame_from_message(make_message(0)),
            10_100_000_000,
        )
        ready = coordinator.handle_frame(
            inspection_frame_from_message(make_message(1)),
            10_200_000_000,
        )
        self.assertEqual(waiting.state, ProcessingState.BUFFERING)
        self.assertEqual(ready.state, ProcessingState.PREDICTED)

    def test_deadline_produces_timeout(self) -> None:
        coordinator = InspectionCoordinator(lambda frame: frame.frame_index)
        coordinator.handle_completion(
            inspection_completion_from_message(make_completion_message()),
            10_000_000_000,
        )
        events = coordinator.expired(10_500_000_000)
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0].state, ProcessingState.TIMEOUT)

    def test_unconfigured_predictor_session_can_be_finalized(self) -> None:
        from predictor import UnconfiguredPredictor

        coordinator = InspectionCoordinator(UnconfiguredPredictor())
        coordinator.handle_frame(
            inspection_frame_from_message(make_message(0, total_frames=1)),
            9_000_000_000,
        )
        event = coordinator.handle_completion(
            inspection_completion_from_message(
                make_completion_message(total_frames=1)
            ),
            10_100_000_000,
        )
        self.assertEqual(event.state, ProcessingState.PREDICTOR_UNAVAILABLE)
        coordinator.finalize(event.inspection_id)
        self.assertEqual(len(coordinator.store), 0)

    def test_stale_cleanup_applies_only_without_completion(self) -> None:
        coordinator = InspectionCoordinator(lambda frame: frame.frame_index)
        coordinator.handle_frame(
            inspection_frame_from_message(make_message(0)),
            1,
            wall_time_ns=100,
        )
        events = coordinator.stale(3_000_000_100, 3_000_000_000)
        self.assertEqual(len(events), 1)
        coordinator.handle_completion(
            inspection_completion_from_message(make_completion_message()),
            10_000_000_000,
            wall_time_ns=3_000_000_100,
        )
        self.assertEqual(
            coordinator.stale(7_000_000_100, 3_000_000_000),
            (),
        )

    def test_finalized_id_is_bounded_duplicate_guard(self) -> None:
        coordinator = InspectionCoordinator(lambda frame: frame.frame_index)
        frame = inspection_frame_from_message(make_message(0, total_frames=1))
        coordinator.handle_frame(frame, 1)
        coordinator.finalize(frame.inspection_id)
        duplicate = coordinator.handle_frame(frame, 1)
        self.assertEqual(duplicate.state, ProcessingState.FINALIZED)

    def test_identity_change_is_turned_into_recheck_for_original_apple(self) -> None:
        coordinator = InspectionCoordinator(lambda frame: frame.frame_index)
        original = inspection_frame_from_message(
            make_message(0, total_frames=2, apple_id="apple-original")
        )
        coordinator.handle_frame(original, 1)
        conflicting = inspection_frame_from_message(
            make_message(1, total_frames=2, apple_id="apple-conflict")
        )
        with self.assertRaises(InspectionIdentityMismatch):
            coordinator.handle_frame(conflicting, 2)

        event = coordinator.identity_mismatch_event(
            original.inspection_id,
            conflicting.apple_id,
            conflicting.total_frames,
        )
        self.assertEqual(event.state, ProcessingState.RECHECK)
        self.assertEqual(event.apple_id, "apple-original")
        self.assertEqual(event.received_count, 1)
        coordinator.finalize(event.inspection_id)
        self.assertEqual(len(coordinator.store), 0)


class ColorDistributionGraphTest(unittest.TestCase):
    def test_ignores_only_ignored_pixels_inside_apple_mask(self) -> None:
        distribution = measure_color_distribution((make_colour_frame(),), (0,))
        self.assertEqual(distribution.apple_pixels, 5)
        self.assertEqual(distribution.ignored_pixels, 1)
        self.assertEqual(distribution.valid_pixels, 4)
        self.assertEqual(
            dict(distribution.category_counts),
            {"RED": 1, "ORANGE": 1, "YELLOW": 1, "GREEN": 0, "OTHER": 1},
        )
        self.assertEqual(
            dict(distribution.category_ratios),
            {
                "RED": 0.25,
                "ORANGE": 0.25,
                "YELLOW": 0.25,
                "GREEN": 0.0,
                "OTHER": 0.25,
            },
        )

    def test_renders_decodable_jpeg_at_declared_resolution(self) -> None:
        distribution = measure_color_distribution((make_colour_frame(),), (0,))
        event = ProcessingEvent(
            state=ProcessingState.PREDICTED,
            inspection_id="inspection-colour-001",
            apple_id="apple-colour-001",
            received_count=1,
            total_frames=1,
        )
        result = QualityResult(
            Grade.HIGH,
            ResultStatus.VALID,
            0.95,
            AppleMeasurements(color_ratio=0.85, diameter_mm=80.0),
            (0,),
        )
        payload = render_color_distribution_jpeg(distribution, event, result)
        with Image.open(io.BytesIO(payload)) as image:
            self.assertEqual(image.format, "JPEG")
            self.assertEqual(
                image.size,
                (COLOR_DISTRIBUTION_WIDTH, COLOR_DISTRIBUTION_HEIGHT),
            )


class RosContractTest(unittest.TestCase):
    def test_topic_names_qos_and_threshold(self) -> None:
        self.assertEqual(INPUT_TOPIC, "/quality/inspection_images")
        self.assertEqual(COMPLETION_TOPIC, "/quality/inspection_completed")
        self.assertEqual(OUTPUT_TOPIC, "/quality/results")
        self.assertEqual(
            COLOR_DISTRIBUTION_TOPIC,
            "/quality/color_distribution_debug/compressed",
        )
        self.assertEqual(INPUT_QOS_DEPTH, 6)
        self.assertEqual(COMPLETION_QOS_DEPTH, 10)
        self.assertEqual(RESULT_QOS_DEPTH, 10)
        self.assertEqual(COLOR_DISTRIBUTION_QOS_DEPTH, 1)
        self.assertEqual(DEFAULT_CONFIDENCE_THRESHOLD, 0.5)


if __name__ == "__main__":
    unittest.main()
