from __future__ import annotations

import unittest

from inspection_session import (
    DuplicateFrameConflict,
    INSPECTION_ROI_FRAME,
    InspectionCompletion,
    InspectionContractError,
    InspectionFrame,
    InspectionIdentityMismatch,
    InspectionStore,
    RESULT_DEADLINE_NS,
    TotalFramesMismatch,
)
from predictor import (
    IncompleteInspectionError,
    PredictorNotConfigured,
    UnconfiguredPredictor,
    predict_declared_frames,
)


CAMERA_K = (500.0, 0.0, 50.0, 0.0, 500.0, 50.0, 0.0, 0.0, 1.0)
CAMERA_P = (500.0, 0.0, 50.0, 0.0, 0.0, 500.0, 50.0, 0.0, 0.0, 0.0, 1.0, 0.0)


def make_frame(
    frame_index: int,
    *,
    inspection_id: str = "inspection-001",
    apple_id: str = "apple-001",
    total_frames: int = 3,
    image_data: bytes | None = None,
) -> InspectionFrame:
    return InspectionFrame(
        inspection_id=inspection_id,
        apple_id=apple_id,
        frame_index=frame_index,
        total_frames=total_frames,
        image_data=image_data if image_data is not None else f"jpeg-{frame_index}".encode(),
        image_format="rgb8; jpeg compressed bgr8",
        apple_mask_data=b"png-mask",
        apple_mask_format="mono8; png",
        ignore_mask_data=b"png-ignore",
        ignore_mask_format="mono8; png",
        depth_data=b"png-depth",
        depth_format="16UC1; compressedDepth png",
        camera_width=100,
        camera_height=100,
        camera_k=CAMERA_K,
        camera_p=CAMERA_P,
        stamp_ns=1_000_000_000 + frame_index,
        frame_id="quality_camera_top_optical_frame",
    )


def make_completion(
    *,
    inspection_id: str = "inspection-001",
    apple_id: str = "apple-001",
    total_frames: int = 3,
    roi_exit_time_ns: int = 10_000_000_000,
) -> InspectionCompletion:
    return InspectionCompletion(
        inspection_id,
        apple_id,
        total_frames,
        roi_exit_time_ns,
        INSPECTION_ROI_FRAME,
    )


class InspectionStoreTest(unittest.TestCase):
    def test_accepts_out_of_order_zero_based_frames(self) -> None:
        store = InspectionStore()
        for index in (2, 0, 1):
            self.assertTrue(store.accept(make_frame(index)).is_new_frame)
        session = store.get("inspection-001")
        assert session is not None
        self.assertTrue(session.has_all_declared_frames)
        self.assertEqual(session.frame_indices, (0, 1, 2))

    def test_identical_duplicate_is_idempotent(self) -> None:
        store = InspectionStore()
        frame = make_frame(0)
        self.assertTrue(store.accept(frame).is_new_frame)
        self.assertFalse(store.accept(frame).is_new_frame)

    def test_conflicting_duplicate_is_rejected(self) -> None:
        store = InspectionStore()
        store.accept(make_frame(0, image_data=b"first"))
        with self.assertRaises(DuplicateFrameConflict):
            store.accept(make_frame(0, image_data=b"different"))

    def test_apple_id_and_total_frames_cannot_change(self) -> None:
        store = InspectionStore()
        store.accept(make_frame(0))
        with self.assertRaises(InspectionIdentityMismatch):
            store.accept(make_frame(1, apple_id="apple-002"))
        with self.assertRaises(TotalFramesMismatch):
            store.accept(make_frame(1, total_frames=4))

    def test_frame_index_is_zero_based_and_bounded_by_total(self) -> None:
        with self.assertRaises(InspectionContractError):
            make_frame(3, total_frames=3)
        with self.assertRaises(InspectionContractError):
            make_frame(-1, total_frames=3)

    def test_rejects_missing_rgbd_or_calibration(self) -> None:
        values = make_frame(0).__dict__
        with self.assertRaises(InspectionContractError):
            InspectionFrame(**{**values, "apple_mask_data": b""})
        with self.assertRaises(InspectionContractError):
            InspectionFrame(**{**values, "ignore_mask_data": b""})
        with self.assertRaises(InspectionContractError):
            InspectionFrame(**{**values, "depth_data": b""})
        with self.assertRaises(InspectionContractError):
            InspectionFrame(**{**values, "camera_k": (0.0,) * 9})
        with self.assertRaises(InspectionContractError):
            InspectionFrame(**{**values, "frame_id": "wrong_camera_frame"})

    def test_completion_can_arrive_before_frames_and_uses_sim_deadline(self) -> None:
        store = InspectionStore()
        completion = make_completion(total_frames=4)
        session = store.complete(completion)
        self.assertFalse(session.deadline_reached(completion.deadline_time_ns - 1))
        self.assertTrue(session.deadline_reached(completion.deadline_time_ns))
        self.assertEqual(
            completion.deadline_time_ns,
            completion.roi_exit_time_ns + RESULT_DEADLINE_NS,
        )
        accepted = store.accept(
            make_frame(
                0,
                inspection_id=completion.inspection_id,
                apple_id=completion.apple_id,
                total_frames=completion.total_frames,
            )
        )
        self.assertTrue(accepted.is_new_frame)

    def test_completion_identity_must_match_existing_session(self) -> None:
        store = InspectionStore()
        store.accept(make_frame(0))
        with self.assertRaises(InspectionIdentityMismatch):
            store.complete(make_completion(apple_id="apple-other"))

    def test_store_pop_cleans_session(self) -> None:
        store = InspectionStore()
        store.accept(make_frame(0))
        self.assertIsNotNone(store.pop("inspection-001"))
        self.assertEqual(len(store), 0)


class PredictorBoundaryTest(unittest.TestCase):
    def test_fake_predictor_runs_in_frame_index_order(self) -> None:
        store = InspectionStore()
        for index in (2, 0, 1):
            session = store.accept(make_frame(index)).session

        class ByteCountPredictor:
            def predict(self, frame: InspectionFrame) -> int:
                return len(frame.image_data)

        predictions = predict_declared_frames(session, ByteCountPredictor())
        self.assertEqual(tuple(item.frame_index for item in predictions), (0, 1, 2))

    def test_inference_rejects_incomplete_batch(self) -> None:
        session = InspectionStore().accept(make_frame(0)).session
        with self.assertRaises(IncompleteInspectionError):
            predict_declared_frames(session, UnconfiguredPredictor())

    def test_one_frame_failure_does_not_abort_other_predictions(self) -> None:
        store = InspectionStore()
        for index in range(5):
            session = store.accept(make_frame(index, total_frames=5)).session

        class OneFrameFailurePredictor:
            def predict(self, frame: InspectionFrame) -> int:
                if frame.frame_index == 1:
                    raise RuntimeError("synthetic frame failure")
                return frame.frame_index

        predictions = predict_declared_frames(session, OneFrameFailurePredictor())
        self.assertEqual(len(predictions), 5)
        self.assertFalse(predictions[1].succeeded)
        self.assertEqual(predictions[1].error_type, "RuntimeError")
        self.assertEqual(
            tuple(item.value for item in predictions if item.succeeded), (0, 2, 3, 4)
        )


    def test_unconfigured_predictor_never_fabricates_result(self) -> None:
        store = InspectionStore()
        for index in range(3):
            session = store.accept(make_frame(index)).session
        with self.assertRaises(PredictorNotConfigured):
            predict_declared_frames(session, UnconfiguredPredictor())


if __name__ == "__main__":
    unittest.main()
