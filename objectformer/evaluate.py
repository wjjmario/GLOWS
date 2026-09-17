import csv
import os

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from PIL import Image

from .metrics import segmentation_report
from .model import semantic_logits


def five_significant(value):
    value = float(value)
    if not np.isfinite(value):
        return value
    return float(f"{value:.5g}")


@torch.inference_mode()
def evaluate_image_only(
    model,
    encoder,
    dataset,
    class_ids,
    device,
    epoch=None,
    output_dir=None,
    max_images=-1,
    save_predictions=False,
):
    """Leakage-safe evaluation: prediction is completed before GT is read."""
    model.eval()
    predictions, ground_truth = [], []
    items = dataset.items if max_images < 0 else dataset.items[:max_images]
    if not items:
        raise ValueError("No evaluation images found")
    for item_index, item in enumerate(
        tqdm(items, desc=f"image-only eval {dataset.split}")
    ):
        image = (
            dataset.read_item_image(item)
            if hasattr(dataset, "read_item_image")
            else dataset.read_image(item["image"])
        )
        tokens = [
            x.to(device).float() for x in encoder.extract_intermediate_tokens(image)
        ]
        patch_h, patch_w = encoder.patch_hw
        outputs = model(tokens, patch_h, patch_w)
        logits = F.interpolate(
            semantic_logits(outputs),
            size=(image.height, image.width),
            mode="bilinear",
            align_corners=False,
        )
        indices = logits.argmax(1)[0].cpu().numpy()
        prediction = np.zeros(indices.shape, dtype=np.uint16)
        for index, cid in enumerate(class_ids):
            prediction[indices == index] = int(cid)
        # Deliberately read labels only after prediction has been materialized.
        target = (
            dataset.read_item_label(item)
            if hasattr(dataset, "read_item_label")
            else dataset.read_label(item["label"])
        )
        if target is not None:
            predictions.append(prediction)
            ground_truth.append(target)
        if save_predictions and output_dir:
            os.makedirs(output_dir, exist_ok=True)
            Image.fromarray(prediction.astype(np.uint8)).save(
                os.path.join(output_dir, item["stem"] + ".png")
            )
    report = (
        segmentation_report(predictions, ground_truth, class_ids)
        if ground_truth
        else {}
    )
    if output_dir and report:
        os.makedirs(output_dir, exist_ok=True)
        path = os.path.join(output_dir, "image_only_metrics.csv")
        write_header = not os.path.exists(path)
        with open(path, "a", newline="", encoding="utf-8") as handle:
            fieldnames = (
                ["epoch", "split", "mIoU", "mF1", "OA"]
                + [f"IoU_{cid}" for cid in class_ids]
                + [f"F1_{cid}" for cid in class_ids]
            )
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            if write_header:
                writer.writeheader()
            row = {
                "epoch": epoch,
                "split": dataset.split,
                "mIoU": five_significant(report["mIoU"]),
                "mF1": five_significant(report["mF1"]),
                "OA": five_significant(report["OA"]),
            }
            row.update(
                {
                    f"IoU_{cid}": five_significant(
                        report["IoU"].get(int(cid), float("nan"))
                    )
                    for cid in class_ids
                }
            )
            row.update(
                {
                    f"F1_{cid}": five_significant(
                        report["F1"].get(int(cid), float("nan"))
                    )
                    for cid in class_ids
                }
            )
            writer.writerow(row)
    model.train()
    return report
