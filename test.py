import argparse

import torch

from objectformer.data import SegDataset, class_ids_from_cfg
from objectformer.encoders import DINOv3Encoder
from objectformer.evaluate import evaluate_image_only
from objectformer.io import load_yaml
from objectformer.model import build_model


def main():
    parser = argparse.ArgumentParser(description="Strictly image-only evaluation")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--split", default="test")
    parser.add_argument("--max-images", type=int, default=-1)
    args = parser.parse_args()
    cfg = load_yaml(args.config)
    device = torch.device(args.device)
    class_ids = class_ids_from_cfg(cfg)
    dataset = SegDataset(cfg, args.split)
    encoder = DINOv3Encoder(cfg["encoder"], device)
    model = build_model(cfg, len(class_ids), encoder.embed_dim).to(device)
    state = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(state.get("ema", state["model"]))
    report = evaluate_image_only(model, encoder, dataset, class_ids, device, state.get("epoch"), cfg.get("infer", {}).get("output_dir"), args.max_images)
    print(report)


if __name__ == "__main__":
    main()

'''
  python infer.py \
    --config configs/vaihingen_point_objectformer.yaml \
    --checkpoint outputs/vaihingen_hybrid_closed_loop_v1/checkpoints/best.pt \
    --device cuda:2 \
    --split test

  python infer.py \
    --config configs/potsdam_point_objectformer.yaml \
    --checkpoint outputs/potsdam_hybrid_closed_loop_v1/checkpoints/best.pt \
    --device cuda:3 \
    --split test
'''