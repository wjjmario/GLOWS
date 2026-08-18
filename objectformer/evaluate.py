import csv
import os

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from .metrics import segmentation_report
from .model import semantic_logits
from .visualization import evenly_spaced_indices, save_label_panels


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
    visualization=None,
):
    """Leakage-safe evaluation: prediction is completed before GT is read."""
    model.eval()
    predictions, ground_truth = [], []
    items = dataset.items if max_images < 0 else dataset.items[:max_images]
    vcfg = visualization or {}
    visualize = bool(vcfg.get("enabled", False)) and output_dir is not None
    visualize = visualize and int(epoch or 0) % max(1, int(vcfg.get("every", 1))) == 0
    selected = set(evenly_spaced_indices(len(items), int(vcfg.get("num_test_images", 6)))) if visualize else set()
    for item_index, item in enumerate(tqdm(items, desc=f"image-only eval {dataset.split}")):
        image = dataset.read_item_image(item) if hasattr(dataset, "read_item_image") else dataset.read_image(item["image"])
        tokens = [x.to(device).float() for x in encoder.extract_intermediate_tokens(image)]
        patch_h, patch_w = encoder.patch_hw
        outputs = model(tokens, patch_h, patch_w)
        logits = F.interpolate(semantic_logits(outputs), size=(image.height, image.width), mode="bilinear", align_corners=False)
        indices = logits.argmax(1)[0].cpu().numpy()
        prediction = np.zeros(indices.shape, dtype=np.uint16)
        for index, cid in enumerate(class_ids):
            prediction[indices == index] = int(cid)
        predictions.append(prediction)
        # Deliberately read labels only after prediction has been materialized.
        target = dataset.read_item_label(item) if hasattr(dataset, "read_item_label") else dataset.read_label(item["label"])
        ground_truth.append(target)
        if item_index in selected:
            path = os.path.join(
                output_dir,
                "visualizations",
                "inference",
                f"epoch_{int(epoch or 0):04d}",
                item["stem"] + "_prediction_gt.png",
            )
            save_label_panels([prediction, target], path)
    report = segmentation_report(predictions, ground_truth, class_ids)
    if output_dir:
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
            row.update({
                f"IoU_{cid}": five_significant(report["IoU"].get(int(cid), float("nan")))
                for cid in class_ids
            })
            row.update({
                f"F1_{cid}": five_significant(report["F1"].get(int(cid), float("nan")))
                for cid in class_ids
            })
            writer.writerow(row)
    model.train()
    return report
