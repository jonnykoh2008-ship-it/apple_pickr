"""OpenCV-only synthetic apple damage detection and dataset evaluation.

This module is the GPU PC 2 pilot path for detecting three rendered damage
appearances without a learned segmentation model.  It deliberately reports
pixel masks and validation metrics; final HIGH/MEDIUM/LOW damage thresholds
remain a separate, approved quality-rule decision.
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np


@dataclass(frozen=True)
class DamageDetectionConfig:
    """Initial synthetic-render thresholds in OpenCV channel scales."""

    bright_wound_hsv_low: tuple[int, int, int] = (7, 75, 135)
    bright_wound_hsv_high: tuple[int, int, int] = (23, 175, 230)
    bright_wound_lab_a_max: int = 166
    bright_wound_local_l_rise: int = 12
    browning_hsv_low: tuple[int, int, int] = (4, 85, 72)
    browning_hsv_high: tuple[int, int, int] = (23, 218, 170)
    browning_lab_a_max: int = 172
    browning_local_l_drop: int = 4
    bruise_value_low: int = 40
    bruise_value_high: int = 138
    bruise_saturation_min: int = 50
    bruise_local_l_drop: int = 18
    bruise_lab_a_max: int = 170
    bruise_lab_b_max: int = 158
    local_context_kernel: int = 51
    edge_exclusion_px: int = 5
    close_kernel: int = 3
    min_component_area_px: int = 3
    # 사과 표면 대비 손상 성분의 크기 상한. 0.04는 등급 판정에 필요한 크기를
    # 오히려 잘라냈다. 논의된 경계값 2.5cm2 는 80mm 사과 투영면적(약 50cm2)의
    # 5%라 그 상한을 넘어, 큰 손상일수록 "손상 없음"으로 판정되는 정반대 동작을
    # 만들었다. 같은 프레임에서 상한만 올리자 재현율 0.44 -> 0.63,
    # IoU 0.39 -> 0.55 로 올랐고 0.10 이후로는 변화가 없었다.
    # 표면의 1/3을 넘는 성분은 여전히 검출 실패로 보고 거부한다.
    max_component_area_ratio: float = 0.35

    def __post_init__(self) -> None:
        if self.local_context_kernel < 3 or self.local_context_kernel % 2 == 0:
            raise ValueError("local_context_kernel must be an odd integer >= 3")
        if self.close_kernel < 1 or self.close_kernel % 2 == 0:
            raise ValueError("close_kernel must be an odd positive integer")
        if self.edge_exclusion_px < 0:
            raise ValueError("edge_exclusion_px must be non-negative")
        if self.min_component_area_px < 1:
            raise ValueError("min_component_area_px must be positive")
        if not 0.0 < self.max_component_area_ratio <= 1.0:
            raise ValueError("max_component_area_ratio must be in (0, 1]")


@dataclass(frozen=True)
class DamageMasks:
    bright_wound: np.ndarray
    browning: np.ndarray
    bruise: np.ndarray
    combined: np.ndarray


@dataclass(frozen=True)
class SegmentationMetrics:
    true_positive: int
    false_positive: int
    false_negative: int
    true_negative: int

    @property
    def precision(self) -> float:
        denominator = self.true_positive + self.false_positive
        return self.true_positive / denominator if denominator else 1.0

    @property
    def recall(self) -> float:
        denominator = self.true_positive + self.false_negative
        return self.true_positive / denominator if denominator else 1.0

    @property
    def iou(self) -> float:
        denominator = self.true_positive + self.false_positive + self.false_negative
        return self.true_positive / denominator if denominator else 1.0

    @property
    def dice(self) -> float:
        denominator = 2 * self.true_positive + self.false_positive + self.false_negative
        return 2 * self.true_positive / denominator if denominator else 1.0

    def to_dict(self) -> dict[str, float | int]:
        result = asdict(self)
        result.update(
            precision=self.precision,
            recall=self.recall,
            iou=self.iou,
            dice=self.dice,
        )
        return result


def _binary_mask(mask: np.ndarray, shape: tuple[int, int], name: str) -> np.ndarray:
    value = np.asarray(mask)
    if value.shape != shape:
        raise ValueError(f"{name} shape {value.shape} does not match RGB shape {shape}")
    return value > 0


def _filter_components(
    mask: np.ndarray, minimum_area: int, maximum_area: int
) -> np.ndarray:
    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask.astype(np.uint8), connectivity=8
    )
    result = np.zeros(mask.shape, dtype=bool)
    for label in range(1, count):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if minimum_area <= area <= maximum_area:
            result |= labels == label
    return result


def detect_damage(
    image_rgb: np.ndarray,
    apple_mask: np.ndarray,
    config: DamageDetectionConfig | None = None,
) -> DamageMasks:
    """Detect synthetic bright wounds, browning and bruises inside one apple mask."""

    config = config or DamageDetectionConfig()
    image = np.asarray(image_rgb)
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("image_rgb must have shape (height, width, 3)")
    if image.dtype != np.uint8:
        raise ValueError("image_rgb must use uint8 RGB pixels")

    surface = _binary_mask(apple_mask, image.shape[:2], "apple_mask")
    analysis_surface = surface
    if config.edge_exclusion_px:
        size = config.edge_exclusion_px * 2 + 1
        edge_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
        eroded = cv2.erode(surface.astype(np.uint8), edge_kernel) > 0
        if eroded.any():
            analysis_surface = eroded
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    hue, saturation, value = cv2.split(hsv)
    lightness, lab_a, lab_b = cv2.split(lab)

    surface_weight = surface.astype(np.float32)
    weighted_lightness = cv2.GaussianBlur(
        lightness.astype(np.float32) * surface_weight,
        (config.local_context_kernel, config.local_context_kernel),
        sigmaX=0,
    )
    local_weight = cv2.GaussianBlur(
        surface_weight,
        (config.local_context_kernel, config.local_context_kernel),
        sigmaX=0,
    )
    local_lightness = np.divide(
        weighted_lightness,
        local_weight,
        out=lightness.astype(np.float32).copy(),
        where=local_weight > 1e-6,
    )
    lightness_delta = lightness.astype(np.float32) - local_lightness

    bright = cv2.inRange(
        hsv,
        np.asarray(config.bright_wound_hsv_low, dtype=np.uint8),
        np.asarray(config.bright_wound_hsv_high, dtype=np.uint8),
    ) > 0
    bright &= lab_a <= config.bright_wound_lab_a_max
    bright &= lightness_delta >= config.bright_wound_local_l_rise

    browning = cv2.inRange(
        hsv,
        np.asarray(config.browning_hsv_low, dtype=np.uint8),
        np.asarray(config.browning_hsv_high, dtype=np.uint8),
    ) > 0
    browning &= lab_a <= config.browning_lab_a_max
    browning &= -lightness_delta >= config.browning_local_l_drop

    local_drop = -lightness_delta
    bruise = (
        (value >= config.bruise_value_low)
        & (value <= config.bruise_value_high)
        & (saturation >= config.bruise_saturation_min)
        & (local_drop >= config.bruise_local_l_drop)
        & (lab_a <= config.bruise_lab_a_max)
        & (lab_b <= config.bruise_lab_b_max)
    )

    bright &= analysis_surface
    browning &= analysis_surface
    bruise &= analysis_surface

    combined = bright | browning | bruise
    if config.close_kernel > 1:
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (config.close_kernel, config.close_kernel)
        )
        combined = cv2.morphologyEx(
            combined.astype(np.uint8), cv2.MORPH_CLOSE, kernel
        ) > 0
    maximum_area = max(
        config.min_component_area_px,
        round(int(surface.sum()) * config.max_component_area_ratio),
    )
    combined = _filter_components(
        combined & analysis_surface,
        config.min_component_area_px,
        maximum_area,
    )

    # Type masks are diagnostic views and must agree with the cleaned union.
    return DamageMasks(
        bright_wound=bright & combined,
        browning=browning & combined,
        bruise=bruise & combined,
        combined=combined,
    )


def segmentation_metrics(
    prediction: np.ndarray,
    ground_truth: np.ndarray,
    evaluation_mask: np.ndarray | None = None,
) -> SegmentationMetrics:
    """Calculate binary segmentation metrics, optionally only on the apple surface."""

    predicted = np.asarray(prediction) > 0
    truth = np.asarray(ground_truth) > 0
    if predicted.shape != truth.shape:
        raise ValueError("prediction and ground_truth shapes must match")
    valid = (
        np.ones(predicted.shape, dtype=bool)
        if evaluation_mask is None
        else _binary_mask(evaluation_mask, predicted.shape, "evaluation_mask")
    )
    return SegmentationMetrics(
        true_positive=int((predicted & truth & valid).sum()),
        false_positive=int((predicted & ~truth & valid).sum()),
        false_negative=int((~predicted & truth & valid).sum()),
        true_negative=int((~predicted & ~truth & valid).sum()),
    )


def draw_damage_overlay(image_rgb: np.ndarray, masks: DamageMasks) -> np.ndarray:
    """Return an RGB overlay: wound=yellow, browning=orange, bruise=blue."""

    overlay = np.asarray(image_rgb).copy()
    color_layer = overlay.copy()
    color_layer[masks.bright_wound] = (255, 235, 0)
    color_layer[masks.browning] = (255, 120, 0)
    color_layer[masks.bruise] = (40, 100, 255)
    blended = cv2.addWeighted(overlay, 0.58, color_layer, 0.42, 0.0)
    contours, _ = cv2.findContours(
        masks.combined.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    cv2.drawContours(blended, contours, -1, (255, 255, 255), 1)
    return blended


def _load_rgb(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"cannot read image: {path}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def _load_mask(path: Path) -> np.ndarray:
    mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise ValueError(f"cannot read mask: {path}")
    return mask > 0


def evaluate_dataset(
    dataset_root: str | Path,
    *,
    config: DamageDetectionConfig | None = None,
    output_dir: str | Path | None = None,
) -> dict:
    """Evaluate every annotated RGB frame without splitting scenario groups."""

    root = Path(dataset_root).expanduser().resolve()
    annotation_path = root / "quality_annotations.json"
    annotations = json.loads(annotation_path.read_text(encoding="utf-8"))
    if not isinstance(annotations, dict) or not annotations:
        raise ValueError("quality_annotations.json must contain annotated RGB frames")

    destination = Path(output_dir).expanduser().resolve() if output_dir else None
    if destination:
        destination.mkdir(parents=True, exist_ok=True)
        (destination / "overlays").mkdir(exist_ok=True)
        (destination / "masks").mkdir(exist_ok=True)

    rows: list[dict] = []
    totals = SegmentationMetrics(0, 0, 0, 0)
    for rgb_name, annotation in sorted(annotations.items()):
        rgb_path = root / rgb_name
        apple_path = root / str(annotation["apple_mask"])
        truth_path = root / str(annotation["damage_mask"])
        image = _load_rgb(rgb_path)
        apple = _load_mask(apple_path)
        truth = _load_mask(truth_path)
        masks = detect_damage(image, apple, config)
        metrics = segmentation_metrics(masks.combined, truth, apple)
        totals = SegmentationMetrics(
            totals.true_positive + metrics.true_positive,
            totals.false_positive + metrics.false_positive,
            totals.false_negative + metrics.false_negative,
            totals.true_negative + metrics.true_negative,
        )
        row = {"image": rgb_name, **metrics.to_dict()}
        rows.append(row)

        if destination:
            stem = Path(rgb_name).stem
            overlay = cv2.cvtColor(draw_damage_overlay(image, masks), cv2.COLOR_RGB2BGR)
            cv2.imwrite(str(destination / "overlays" / f"{stem}_damage.png"), overlay)
            cv2.imwrite(
                str(destination / "masks" / f"{stem}_damage_mask.png"),
                masks.combined.astype(np.uint8) * 255,
            )

    report = {
        "dataset": str(root),
        "frames": len(rows),
        "aggregate": totals.to_dict(),
        "mean_frame_iou": float(np.mean([row["iou"] for row in rows])),
        "mean_frame_dice": float(np.mean([row["dice"] for row in rows])),
        "config": asdict(config or DamageDetectionConfig()),
    }
    if destination:
        with (destination / "frame_metrics.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
        (destination / "summary.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", help="Boundary-pilot dataset run directory")
    parser.add_argument(
        "--output-dir",
        help="Optional directory for masks, overlays, frame CSV and summary JSON",
    )
    return parser


def main(argv: Iterable[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    print(
        json.dumps(
            evaluate_dataset(args.dataset, output_dir=args.output_dir),
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
