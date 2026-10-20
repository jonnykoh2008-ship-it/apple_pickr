"""Train segmentation quality heads without inventing missing annotations."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from apple_quality_dataset import (
    AppleQualityTorchDataset,
    inspect_capabilities,
    load_dataset_records,
)
from predictor import MODEL_ARCHITECTURE, create_measurement_model


TARGET_NAMES = ("target_color_mask", "damage_mask", "severe_defect")


def split_records(records, validation_fraction: float, seed: int):
    """Split by apple scenario so its five views never leak across splits."""

    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation_fraction must be between 0 and 1")
    groups: dict[str, list] = {}
    for record in records:
        groups.setdefault(record.group_key, []).append(record)
    keys = list(groups)
    random.Random(seed).shuffle(keys)
    validation_group_count = max(1, round(len(keys) * validation_fraction))
    validation_keys = set(keys[:validation_group_count])
    train = tuple(
        record
        for key, values in groups.items()
        if key not in validation_keys
        for record in values
    )
    validation = tuple(
        record
        for key, values in groups.items()
        if key in validation_keys
        for record in values
    )
    if not train or not validation:
        raise RuntimeError("dataset split must contain both training and validation groups")
    return train, validation


def _batch_loss(
    mask_probabilities,
    severe_probability,
    confidences,
    mask_targets,
    severe_target,
    target_valid,
    valid_masks,
):
    import torch
    import torch.nn.functional as functional

    probabilities = mask_probabilities.clamp(1e-6, 1.0 - 1e-6)
    mask_weight = (
        valid_masks.to(probabilities.dtype)
        * target_valid[:, :2, None, None].to(probabilities.dtype)
    )
    positive = (mask_targets * mask_weight).sum(dim=(0, 2, 3))
    negative = ((1.0 - mask_targets) * mask_weight).sum(dim=(0, 2, 3))
    positive_weight = (negative / positive.clamp_min(1.0)).clamp(1.0, 50.0)
    pixel_loss = -(
        positive_weight[None, :, None, None]
        * mask_targets
        * probabilities.log()
        + (1.0 - mask_targets) * (1.0 - probabilities).log()
    )
    mask_denominator = mask_weight.sum().clamp_min(1.0)
    balanced_bce = (pixel_loss * mask_weight).sum() / mask_denominator

    intersection = (probabilities * mask_targets * mask_weight).sum(dim=(2, 3))
    denominator = (
        (probabilities + mask_targets) * mask_weight
    ).sum(dim=(2, 3))
    dice_loss = 1.0 - (2.0 * intersection + 1.0) / (denominator + 1.0)
    valid_target_weight = target_valid[:, :2].to(dice_loss.dtype)
    dice_loss = (
        dice_loss * valid_target_weight
    ).sum() / valid_target_weight.sum().clamp_min(1.0)
    segmentation_loss = balanced_bce + dice_loss

    severe_loss = functional.binary_cross_entropy(
        severe_probability[:, 0].clamp(1e-6, 1.0 - 1e-6),
        severe_target,
        reduction="none",
    )
    severe_weight = target_valid[:, 2].to(severe_loss.dtype)
    severe_loss = (severe_loss * severe_weight).sum() / severe_weight.sum().clamp_min(1.0)

    with torch.no_grad():
        predicted = mask_probabilities >= 0.5
        truth = mask_targets >= 0.5
        valid = valid_masks.to(torch.bool)
        intersection = (predicted & truth & valid).sum(dim=(2, 3)).to(torch.float32)
        union = ((predicted | truth) & valid).sum(dim=(2, 3)).to(torch.float32)
        mask_confidence_target = torch.where(
            union > 0,
            intersection / union.clamp_min(1.0),
            torch.ones_like(union),
        )
        severe_confidence_target = (
            1.0 - torch.abs(severe_probability[:, 0] - severe_target)
        ).detach()
        confidence_target = torch.cat(
            (mask_confidence_target, severe_confidence_target[:, None]),
            dim=1,
        )
    confidence_loss = (confidences - confidence_target).square()
    confidence_weight = target_valid.to(confidence_loss.dtype)
    confidence_loss = (
        confidence_loss * confidence_weight
    ).sum() / confidence_weight.sum().clamp_min(1.0)
    return segmentation_loss + severe_loss + 0.25 * confidence_loss


def _evaluate(model, loader, device):
    import torch

    intersections = torch.zeros(2, device=device)
    unions = torch.zeros(2, device=device)
    severe_correct = torch.tensor(0.0, device=device)
    severe_count = torch.tensor(0.0, device=device)
    model.eval()
    with torch.inference_mode():
        for images, mask_targets, severe_target, target_valid, valid_masks in loader:
            images = images.to(device)
            mask_targets = mask_targets.to(device)
            severe_target = severe_target.to(device)
            target_valid = target_valid.to(device)
            valid_masks = valid_masks.to(device)
            masks, severe, _ = model(images)
            predicted = masks >= 0.5
            truth = mask_targets >= 0.5
            for index in range(2):
                valid_samples = target_valid[:, index, None, None]
                valid_pixels = valid_masks[:, index : index + 1] & valid_samples[:, None]
                intersections[index] += (
                    predicted[:, index : index + 1]
                    & truth[:, index : index + 1]
                    & valid_pixels
                ).sum()
                unions[index] += (
                    (predicted[:, index : index + 1] | truth[:, index : index + 1])
                    & valid_pixels
                ).sum()
            severe_valid = target_valid[:, 2]
            severe_correct += (
                ((severe[:, 0] >= 0.5) == (severe_target >= 0.5))
                & severe_valid
            ).sum()
            severe_count += severe_valid.sum()
    return {
        "target_color_mask_iou": (
            float((intersections[0] / unions[0]).cpu()) if unions[0] else None
        ),
        "damage_mask_iou": (
            float((intersections[1] / unions[1]).cpu()) if unions[1] else None
        ),
        "severe_defect_accuracy": (
            float((severe_correct / severe_count).cpu()) if severe_count else None
        ),
    }


def train(args) -> dict:
    import torch
    from torch.utils.data import DataLoader

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    records = load_dataset_records(args.dataset)
    if args.max_samples:
        records = records[: args.max_samples]
    capabilities = inspect_capabilities(records)
    if args.require_complete_quality_labels and not capabilities.supports_complete_quality_training:
        raise RuntimeError(
            "complete quality training requires target_color_mask, damage_mask and "
            "severe_defect annotations"
        )
    if (
        args.require_three_metric_labels
        and not capabilities.supports_three_metric_training
    ):
        raise RuntimeError(
            "three-metric training requires diameter, target_color_mask and "
            "damage_mask labels for every sample"
        )

    label_counts = {
        "target_color_mask": capabilities.color_mask_labels,
        "damage_mask": capabilities.damage_mask_labels,
        "severe_defect": capabilities.severe_defect_labels,
    }
    trained_targets = [
        name for name in TARGET_NAMES if label_counts[name] > 0
    ]
    if not trained_targets:
        raise RuntimeError(
            "the dataset has diameter metadata but no segmentation quality annotations; "
            "add mask paths to quality_annotations.json before training"
        )

    train_records, validation_records = split_records(
        records,
        args.validation_fraction,
        args.seed,
    )
    loader_options = {
        "batch_size": args.batch_size,
        "num_workers": args.workers,
        "pin_memory": torch.cuda.is_available(),
    }
    train_loader = DataLoader(
        AppleQualityTorchDataset(train_records, args.image_size),
        shuffle=True,
        **loader_options,
    )
    validation_loader = DataLoader(
        AppleQualityTorchDataset(validation_records, args.image_size),
        shuffle=False,
        **loader_options,
    )
    device = (
        "cuda"
        if args.device == "auto" and torch.cuda.is_available()
        else "cpu"
        if args.device == "auto"
        else args.device
    )
    model = create_measurement_model().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        batches = 0
        for batch in train_loader:
            images, mask_targets, severe_target, target_valid, valid_masks = (
                value.to(device) for value in batch
            )
            optimizer.zero_grad(set_to_none=True)
            masks, severe, confidences = model(images)
            loss = _batch_loss(
                masks,
                severe,
                confidences,
                mask_targets,
                severe_target,
                target_valid,
                valid_masks,
            )
            loss.backward()
            optimizer.step()
            running_loss += float(loss.detach().cpu())
            batches += 1
        metrics = _evaluate(model, validation_loader, device)
        print(
            f"epoch {epoch:03d}/{args.epochs:03d} "
            f"loss={running_loss / max(1, batches):.5f} validation={metrics}"
        )

    metrics = _evaluate(model, validation_loader, device)
    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "architecture": MODEL_ARCHITECTURE,
            "model_state": model.state_dict(),
            "image_size": args.image_size,
            "trained_targets": trained_targets,
            "validation_metrics": metrics,
            "dataset_capabilities": capabilities.to_dict(),
            "quality_criteria": ["color_ratio", "diameter_mm", "damage_area_cm2"],
            "severe_defect_trained": "severe_defect" in trained_targets,
            "seed": args.seed,
        },
        output,
    )

    if args.onnx_output:
        onnx_output = Path(args.onnx_output).expanduser().resolve()
        onnx_output.parent.mkdir(parents=True, exist_ok=True)
        dummy = torch.zeros(1, 3, args.image_size, args.image_size, device=device)
        try:
            torch.onnx.export(
                model,
                dummy,
                onnx_output,
                input_names=["images"],
                output_names=["masks", "severe", "confidences"],
                dynamic_axes={
                    "images": {0: "batch"},
                    "masks": {0: "batch"},
                    "severe": {0: "batch"},
                    "confidences": {0: "batch"},
                },
                opset_version=17,
            )
        except ModuleNotFoundError as exc:
            raise RuntimeError("install the onnx package before ONNX export") from exc
        onnx_output.with_suffix(onnx_output.suffix + ".json").write_text(
            json.dumps(
                {
                    "architecture": MODEL_ARCHITECTURE,
                    "image_size": args.image_size,
                    "trained_targets": trained_targets,
                    "quality_criteria": [
                        "color_ratio",
                        "diameter_mm",
                        "damage_area_cm2",
                    ],
                    "severe_defect_trained": "severe_defect" in trained_targets,
                },
                indent=2,
            ),
            encoding="utf-8",
        )

    report = {
        "checkpoint": str(output),
        "architecture": MODEL_ARCHITECTURE,
        "device": device,
        "trained_targets": trained_targets,
        "validation_metrics": metrics,
        "capabilities": capabilities.to_dict(),
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="apple_data")
    parser.add_argument(
        "--output",
        default="quality_grading_system/models/apple_quality_segmentation.pt",
    )
    parser.add_argument("--onnx-output")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--image-size", type=int, default=640)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20260823)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--require-complete-quality-labels", action="store_true")
    parser.add_argument("--require-three-metric-labels", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> None:
    train(build_parser().parse_args(argv))


if __name__ == "__main__":
    main()
