import os
import sys

import numpy as np
import torch
from PIL import Image, UnidentifiedImageError

from objectformer.io import ensure_dir, save_tif


class SAM2ProposalGenerator:
    def __init__(self, cfg, device, dataset_name="default"):
        self.cfg = cfg
        self.device = device
        self.cache = bool(cfg.get("cache", True))
        self.cache_dir = cfg.get("cache_dir", "./cache/sam2")
        self.dataset_name = dataset_name
        repo = cfg["repo"]
        if repo not in sys.path:
            sys.path.insert(0, repo)
        from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
        from sam2.build_sam import build_sam2

        checkpoint = cfg["checkpoint"]
        if not os.path.exists(checkpoint):
            raise FileNotFoundError(checkpoint)
        print(f"loading SAM2: {checkpoint}")
        sam2 = build_sam2(cfg["model_cfg"], checkpoint, device=device, apply_postprocessing=False)
        self.generator = SAM2AutomaticMaskGenerator(
            model=sam2,
            points_per_side=int(cfg.get("points_per_side", 32)),
            pred_iou_thresh=float(cfg.get("pred_iou_thresh", 0.9)),
            stability_score_thresh=float(cfg.get("stability_score_thresh", 0.7)),
            box_nms_thresh=float(cfg.get("box_nms_thresh", 0.95)),
            crop_n_layers=int(cfg.get("crop_n_layers", 1)),
            crop_nms_thresh=float(cfg.get("crop_nms_thresh", 0.95)),
            crop_n_points_downscale_factor=int(cfg.get("crop_n_points_downscale_factor", 1)),
            min_mask_region_area=0,
        )

    def _cache_path(self, split, stem):
        return os.path.join(self.cache_dir, self.dataset_name, split, f"{stem}_instance_id.tif")

    def _filter_anns(self, anns, h, w):
        min_area = int(self.cfg.get("min_area", 10))
        max_area_ratio = float(self.cfg.get("max_area_ratio", 0.98))
        image_area = h * w
        out = []
        for ann in anns:
            area = int(ann.get("area", ann["segmentation"].sum()))
            if area < min_area:
                continue
            if area > image_area * max_area_ratio:
                continue
            out.append(ann)
        return out

    @staticmethod
    def anns_to_instance_map(anns, h, w):
        inst = np.zeros((h, w), dtype=np.uint16)
        anns = sorted(anns, key=lambda x: float(x.get("predicted_iou", 0.0)))
        for idx, ann in enumerate(anns, start=1):
            inst[ann["segmentation"].astype(bool)] = idx
        return inst

    def _filter_visible_instances(self, instance_map):
        """Remove tiny visible fragments left after overlapping masks compose.

        ``min_area`` filters the original SAM masks before composition, but a
        later overlapping mask can reduce an accepted mask to only a handful
        of visible pixels.  The optional second threshold removes those
        fragments and compacts IDs.  It is disabled unless explicitly set, so
        existing dataset configurations retain their historical behavior.
        """
        min_area = int(self.cfg.get("visible_min_area", 0))
        if min_area <= 0:
            return instance_map
        ids, counts = np.unique(instance_map, return_counts=True)
        keep = ids[(ids > 0) & (counts >= min_area)]
        output = np.zeros(instance_map.shape, dtype=np.uint16)
        for new_id, old_id in enumerate(keep, start=1):
            output[instance_map == old_id] = new_id
        return output

    def generate(self, image_pil, stem="image", split="unknown"):
        if self.cache:
            path = self._cache_path(split, stem)
            if os.path.exists(path):
                try:
                    with Image.open(path) as cached:
                        return np.array(cached)
                except (UnidentifiedImageError, OSError, ValueError):
                    # A previous worker may have been interrupted while
                    # writing. Recompute instead of poisoning future runs.
                    pass
        image_np = np.array(image_pil.convert("RGB"))
        h, w = image_np.shape[:2]
        with torch.inference_mode():
            if self.device.type == "cuda":
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    try:
                        anns = self.generator.generate(image_np)
                    except IndexError:
                        # SAM2's automatic generator currently raises inside
                        # box_area when a crop yields no masks.  An empty
                        # proposal set is a valid result for uniform/no-data
                        # remote-sensing patches.
                        anns = []
            else:
                try:
                    anns = self.generator.generate(image_np)
                except IndexError:
                    anns = []
        inst = self.anns_to_instance_map(self._filter_anns(anns, h, w), h, w)
        inst = self._filter_visible_instances(inst)
        if self.cache:
            path = self._cache_path(split, stem)
            ensure_dir(os.path.dirname(path))
            temporary = f"{path}.{os.getpid()}.tmp.tif"
            save_tif(inst, temporary)
            os.replace(temporary, path)
        return inst
