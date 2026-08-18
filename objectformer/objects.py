from dataclasses import asdict, dataclass

import numpy as np
import torch
import torch.nn.functional as F


@dataclass
class PseudoInstance:
    mask_id: int
    class_id: int
    confidence: float
    source: str
    anchored: bool = False
    first_epoch: int = 0
    last_epoch: int = 0
    group_id: int = -1

    def to_dict(self):
        return asdict(self)


def proposal_nodes(feature_map, instance_map):
    """Pool one normalized DINO descriptor for every SAM proposal."""
    inst = np.asarray(instance_map, dtype=np.int64)
    _, channels, hf, wf = feature_map.shape
    small = torch.as_tensor(inst, device=feature_map.device, dtype=torch.float32)[None, None]
    small = F.interpolate(small, size=(hf, wf), mode="nearest")[0, 0].long()
    nodes = []
    feat = feature_map[0].float()
    for mid in np.unique(inst):
        mid = int(mid)
        if mid <= 0:
            continue
        token_mask = small == mid
        pixel_mask = inst == mid
        if not token_mask.any() or not pixel_mask.any():
            continue
        embedding = F.normalize(feat[:, token_mask].mean(dim=1), dim=0)
        nodes.append({
            "mask_id": mid,
            "embedding": embedding,
            "mask": pixel_mask,
            "area": int(pixel_mask.sum()),
        })
    return nodes


def instances_to_targets(records, instance_map, class_to_index, device, nodes=None, point_fallbacks=None):
    labels, masks, weights, mask_ids, embeddings = [], [], [], [], []
    node_by_id = {int(n["mask_id"]): n for n in (nodes or [])}
    inst = np.asarray(instance_map)
    for record in records:
        cid = int(record.class_id)
        if cid not in class_to_index:
            continue
        mask = inst == int(record.mask_id)
        if not mask.any():
            continue
        labels.append(class_to_index[cid])
        masks.append(torch.from_numpy(mask.copy()))
        confidence = float(record.confidence)
        weights.append(float(np.clip(confidence, 0.05, 1.0)) if np.isfinite(confidence) else 0.05)
        mask_ids.append(int(record.mask_id))
        node = node_by_id.get(int(record.mask_id))
        if node is not None:
            embeddings.append(node["embedding"].detach())
    h, w = inst.shape
    # Dataset-specific point fallback: an unmatched point is still a valid
    # weak instance target.  The caller supplies a small pixel mask rather
    # than inserting a synthetic id into the SAM instance map or pseudo bank.
    for index, fallback in enumerate(point_fallbacks or []):
        cid = int(fallback["class_id"])
        if cid not in class_to_index:
            continue
        cx, cy = int(round(float(fallback["x"]))), int(round(float(fallback["y"])))
        radius = max(0, int(fallback.get("radius", 1)))
        mask = np.zeros((h, w), dtype=np.bool_)
        x0, x1 = max(0, cx - radius), min(w, cx + radius + 1)
        y0, y1 = max(0, cy - radius), min(h, cy + radius + 1)
        if x1 <= x0 or y1 <= y0:
            continue
        mask[y0:y1, x0:x1] = True
        labels.append(class_to_index[cid])
        masks.append(torch.from_numpy(mask))
        weights.append(float(np.clip(fallback.get("confidence", 1.0), 0.05, 1.0)))
        mask_ids.append(-(index + 1))
    return {
        "labels": torch.tensor(labels, dtype=torch.long, device=device),
        "masks": torch.stack(masks).to(device=device, dtype=torch.float32) if masks else torch.zeros((0, h, w), device=device),
        "weights": torch.tensor(weights, dtype=torch.float32, device=device),
        "mask_ids": mask_ids,
        "embeddings": torch.stack(embeddings).to(device) if len(embeddings) == len(labels) and embeddings else torch.zeros((0, 0), device=device),
    }


def semantic_from_instances(records, instance_map, ignore_index=255):
    out = np.full(np.asarray(instance_map).shape, int(ignore_index), dtype=np.uint16)
    for record in sorted(records, key=lambda x: float(x.confidence)):
        out[np.asarray(instance_map) == int(record.mask_id)] = int(record.class_id)
    return out
