"""Replaceable segmentation inference boundary for GPU PC 2."""

from __future__ import annotations

import io
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Generic, Protocol, TypeVar, runtime_checkable

from inspection_session import InspectionFrame, InspectionSession


PredictionT = TypeVar("PredictionT")
MODEL_ARCHITECTURE = "quality_segmentation_v1"
MASK_TARGETS = ("target_color_mask", "damage_mask")
CONFIDENCE_TARGETS = ("color", "damage", "severe")


class PredictorNotConfigured(RuntimeError):
    """Raised when runtime inference is requested without an approved model."""


class IncompleteInspectionError(ValueError):
    """Raised when batch inference is requested before all declared frames arrive."""


@runtime_checkable
class FramePredictor(Protocol[PredictionT]):
    def predict(self, frame: InspectionFrame) -> PredictionT:
        """Return one model-specific prediction for frame."""


@dataclass(frozen=True)
class IndexedPrediction(Generic[PredictionT]):
    frame_index: int
    value: PredictionT | None = None
    error_type: str | None = None
    error_message: str | None = None

    @property
    def succeeded(self) -> bool:
        return self.error_type is None


@dataclass(frozen=True)
class FrameModelPrediction:
    """Per-pixel quality masks and image-level severe-defect output."""

    color_mask: Any | None = None
    damage_mask: Any | None = None
    severe_defect: bool | None = None
    color_confidence: float | None = None
    damage_confidence: float | None = None
    severe_confidence: float | None = None


@dataclass(frozen=True)
class LetterboxTransform:
    original_width: int
    original_height: int
    resized_width: int
    resized_height: int
    offset_x: int
    offset_y: int
    image_size: int


class UnconfiguredPredictor:
    def predict(self, frame: InspectionFrame) -> None:
        del frame
        raise PredictorNotConfigured("no approved quality model is configured")


def create_measurement_model():
    """Create the segmentation network shared by training and inference."""

    try:
        from torch import nn
    except ImportError as exc:
        raise PredictorNotConfigured("PyTorch is required for model construction") from exc

    class QualitySegmentationNet(nn.Module):
        def __init__(self) -> None:
            super().__init__()

            def down(input_channels, output_channels):
                return nn.Sequential(
                    nn.Conv2d(input_channels, output_channels, 3, stride=2, padding=1),
                    nn.BatchNorm2d(output_channels),
                    nn.SiLU(inplace=True),
                )

            def up(input_channels, output_channels):
                return nn.Sequential(
                    nn.ConvTranspose2d(input_channels, output_channels, 4, stride=2, padding=1),
                    nn.BatchNorm2d(output_channels),
                    nn.SiLU(inplace=True),
                )

            self.encoder = nn.Sequential(
                down(3, 16),
                down(16, 32),
                down(32, 64),
                down(64, 96),
            )
            self.decoder = nn.Sequential(
                up(96, 64),
                up(64, 32),
                up(32, 16),
                nn.ConvTranspose2d(16, len(MASK_TARGETS), 4, stride=2, padding=1),
            )
            self.pool = nn.AdaptiveAvgPool2d(1)
            self.severe_head = nn.Linear(96, 1)
            self.confidence_head = nn.Linear(96, len(CONFIDENCE_TARGETS))

        def forward(self, images):
            features = self.encoder(images)
            masks = self.decoder(features).sigmoid()
            pooled = self.pool(features).flatten(1)
            severe = self.severe_head(pooled).sigmoid()
            confidences = self.confidence_head(pooled).sigmoid()
            return masks, severe, confidences

    return QualitySegmentationNet()


def _image_bytes_to_array(image_data: bytes, image_size: int):
    try:
        import numpy as np
        from PIL import Image
    except ImportError as exc:
        raise PredictorNotConfigured(
            "NumPy and Pillow are required for image inference"
        ) from exc

    try:
        image = Image.open(io.BytesIO(image_data)).convert("RGB")
    except Exception as exc:
        raise ValueError(
            "InspectionImage does not contain a decodable compressed RGB image"
        ) from exc
    scale = min(image_size / image.width, image_size / image.height)
    resized_width = max(1, round(image.width * scale))
    resized_height = max(1, round(image.height * scale))
    resized = image.resize((resized_width, resized_height), Image.Resampling.BILINEAR)
    offset_x = (image_size - resized_width) // 2
    offset_y = (image_size - resized_height) // 2
    canvas = Image.new("RGB", (image_size, image_size), (114, 114, 114))
    canvas.paste(resized, (offset_x, offset_y))
    pixels = np.asarray(canvas, dtype=np.float32) / 255.0
    array = np.transpose(pixels, (2, 0, 1))[None].copy()
    transform = LetterboxTransform(
        original_width=image.width,
        original_height=image.height,
        resized_width=resized_width,
        resized_height=resized_height,
        offset_x=offset_x,
        offset_y=offset_y,
        image_size=image_size,
    )
    return array, transform


def _restore_mask(mask, transform: LetterboxTransform):
    import numpy as np
    from PIL import Image

    array = (
        mask.detach().cpu().numpy()
        if hasattr(mask, "detach")
        else np.asarray(mask)
    )
    cropped = array[
        transform.offset_y : transform.offset_y + transform.resized_height,
        transform.offset_x : transform.offset_x + transform.resized_width,
    ]
    if cropped.size == 0:
        raise ValueError("model mask is empty after removing letterbox padding")
    restored = Image.fromarray(np.asarray(cropped, dtype=np.float32)).resize(
        (transform.original_width, transform.original_height),
        Image.Resampling.BILINEAR,
    )
    return np.asarray(restored, dtype=np.float32)


def _prediction_from_outputs(
    masks,
    severe,
    confidences,
    transform: LetterboxTransform,
    trained_targets: frozenset[str],
) -> FrameModelPrediction:
    color_mask = (
        _restore_mask(masks[0, 0], transform)
        if "target_color_mask" in trained_targets
        else None
    )
    damage_mask = (
        _restore_mask(masks[0, 1], transform)
        if "damage_mask" in trained_targets
        else None
    )
    severe_value = (
        float(severe[0, 0])
        if "severe_defect" in trained_targets
        else None
    )
    raw_confidences = [float(value) for value in confidences[0]]
    return FrameModelPrediction(
        color_mask=color_mask,
        damage_mask=damage_mask,
        severe_defect=severe_value >= 0.5 if severe_value is not None else None,
        color_confidence=(
            raw_confidences[0] if "target_color_mask" in trained_targets else None
        ),
        damage_confidence=(
            raw_confidences[1] if "damage_mask" in trained_targets else None
        ),
        severe_confidence=(
            raw_confidences[2] if "severe_defect" in trained_targets else None
        ),
    )


class TorchMeasurementPredictor:
    def __init__(self, checkpoint_path: str | Path, *, device: str = "auto") -> None:
        try:
            import torch
        except ImportError as exc:
            raise PredictorNotConfigured("PyTorch is not installed") from exc

        checkpoint_path = Path(checkpoint_path).expanduser().resolve()
        if not checkpoint_path.is_file():
            raise PredictorNotConfigured(f"model checkpoint not found: {checkpoint_path}")
        selected_device = (
            "cuda"
            if device == "auto" and torch.cuda.is_available()
            else "cpu"
            if device == "auto"
            else device
        )
        checkpoint = torch.load(checkpoint_path, map_location=selected_device, weights_only=False)
        if checkpoint.get("architecture") != MODEL_ARCHITECTURE:
            raise PredictorNotConfigured(
                f"checkpoint must use architecture {MODEL_ARCHITECTURE}"
            )
        self._model = create_measurement_model()
        self._model.load_state_dict(checkpoint["model_state"])
        self._model.to(selected_device).eval()
        self._device = selected_device
        self._image_size = int(checkpoint.get("image_size", 640))
        self._trained_targets = frozenset(checkpoint.get("trained_targets", []))

    def predict(self, frame: InspectionFrame) -> FrameModelPrediction:
        import torch

        array, transform = _image_bytes_to_array(frame.image_data, self._image_size)
        tensor = torch.from_numpy(array)
        with torch.inference_mode():
            masks, severe, confidences = self._model(tensor.to(self._device))
        return _prediction_from_outputs(
            masks.detach().cpu(),
            severe.detach().cpu(),
            confidences.detach().cpu(),
            transform,
            self._trained_targets,
        )


class OnnxMeasurementPredictor:
    def __init__(self, model_path: str | Path) -> None:
        try:
            import onnxruntime as ort
        except ImportError as exc:
            raise PredictorNotConfigured(
                "onnxruntime-gpu is required on GPU PC 2 for ONNX inference"
            ) from exc
        import json

        model_path = Path(model_path).expanduser().resolve()
        metadata_path = model_path.with_suffix(model_path.suffix + ".json")
        if not model_path.is_file() or not metadata_path.is_file():
            raise PredictorNotConfigured("ONNX model and its .json metadata sidecar are required")
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata.get("architecture") != MODEL_ARCHITECTURE:
            raise PredictorNotConfigured(
                f"ONNX metadata must use architecture {MODEL_ARCHITECTURE}"
            )
        providers = (
            ["CUDAExecutionProvider", "CPUExecutionProvider"]
            if "CUDAExecutionProvider" in ort.get_available_providers()
            else ["CPUExecutionProvider"]
        )
        self._session = ort.InferenceSession(str(model_path), providers=providers)
        self._image_size = int(metadata.get("image_size", 640))
        self._trained_targets = frozenset(metadata.get("trained_targets", []))

    def predict(self, frame: InspectionFrame) -> FrameModelPrediction:
        array, transform = _image_bytes_to_array(frame.image_data, self._image_size)
        raw_masks, raw_severe, raw_confidences = self._session.run(
            None,
            {"images": array},
        )
        return _prediction_from_outputs(
            raw_masks,
            raw_severe,
            raw_confidences,
            transform,
            self._trained_targets,
        )


def load_measurement_predictor(model_path: str | Path, *, backend: str = "auto"):
    # 학습 모델 없이 OpenCV 임계값만으로 착색 mask를 만드는 경로. 모델 파일이
    # 필요 없으므로 model_path 는 무시한다.
    if backend == "opencv_color":
        from opencv_color_predictor import OpenCvColorPredictor

        return OpenCvColorPredictor()
    path = Path(model_path)
    selected = (
        "onnx" if path.suffix.lower() == ".onnx" else "torch"
    ) if backend == "auto" else backend
    if selected == "onnx":
        return OnnxMeasurementPredictor(path)
    if selected == "torch":
        return TorchMeasurementPredictor(path)
    raise ValueError(
        "backend must be auto, onnx, torch, or opencv_color"
    )


def predict_declared_frames(
    session: InspectionSession,
    predictor: FramePredictor[PredictionT],
) -> tuple[IndexedPrediction[PredictionT], ...]:
    if not session.has_all_declared_frames:
        raise IncompleteInspectionError(
            f"inspection {session.inspection_id!r} has {session.received_count}/"
            f"{session.total_frames} declared frames"
        )

    results: list[IndexedPrediction[PredictionT]] = []
    for frame in session.ordered_frames:
        try:
            value = predictor.predict(frame)
        except PredictorNotConfigured:
            raise
        except Exception as exc:
            results.append(
                IndexedPrediction(
                    frame_index=frame.frame_index,
                    error_type=type(exc).__name__,
                    error_message=str(exc),
                )
            )
        else:
            results.append(IndexedPrediction(frame.frame_index, value=value))
    return tuple(results)
