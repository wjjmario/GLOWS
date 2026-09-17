"""Image-only prediction; SAM2 and point annotations are not loaded."""

import argparse
from pathlib import Path

import torch

from objectformer.data import SegDataset, class_ids_from_cfg
from objectformer.encoders import DINOv3Encoder
from objectformer.evaluate import evaluate_image_only
from objectformer.io import load_yaml
from objectformer.model import build_model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--split", default=None)
    parser.add_argument(
        "--images", help="Optional directory of images; no labels required"
    )
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--max-images", type=int, default=-1)
    args = parser.parse_args()
    cfg = load_yaml(args.config)
    split = args.split or cfg["dataset"]["infer_split"]
    if args.images:
        cfg["dataset"].update(
            root=str(Path(args.images).resolve().parent),
            image_dir=str(Path(args.images).resolve()),
            label_dir="__no_labels__",
        )
        split = "infer"
    dataset = SegDataset(cfg, split)
    class_ids = class_ids_from_cfg(cfg)
    device = torch.device(args.device)
    encoder = DINOv3Encoder(cfg["encoder"], device)
    model = build_model(cfg, len(class_ids), encoder.embed_dim).to(device)
    # Load only trusted checkpoints (torch pickle format).
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(state["ema"] if "ema" in state else state["model"])
    report = evaluate_image_only(
        model,
        encoder,
        dataset,
        class_ids,
        device,
        state.get("epoch"),
        args.output_dir or cfg["infer"]["output_dir"],
        args.max_images,
        save_predictions=True,
    )
    print(report if report else "Predictions saved; no ground truth supplied.")


if __name__ == "__main__":
    main()
