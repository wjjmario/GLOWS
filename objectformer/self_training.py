import copy
import hashlib
import json

import numpy as np
import torch
import torch.nn.functional as F

from .objects import PseudoInstance


class EMAModel:
    def __init__(self, model, decay=0.999):
        self.decay = float(decay)
        self.model = copy.deepcopy(model).eval()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)

    @torch.no_grad()
    def update(self, student):
        student_state = student.state_dict()
        for name, value in self.model.state_dict().items():
            source = student_state[name].detach()
            if value.is_floating_point():
                value.mul_(self.decay).add_(source, alpha=1.0 - self.decay)
            else:
                value.copy_(source)

    def state_dict(self):
        return self.model.state_dict()

    def load_state_dict(self, state):
        self.model.load_state_dict(state)


def point_signature(points):
    normalized = {str(int(k)): [[round(float(x), 4), round(float(y), 4)] for x, y in v] for k, v in sorted(points.items())}
    return hashlib.sha256(json.dumps(normalized, sort_keys=True).encode("utf-8")).hexdigest()


@torch.no_grad()
def proposal_decoder_scores(outputs, instance_map, class_ids):
    """Project EMA query predictions onto fixed SAM proposal objects."""
    class_prob = outputs["pred_logits"][0].softmax(-1)[..., :-1]
    masks = F.interpolate(outputs["pred_masks"], size=np.asarray(instance_map).shape, mode="bilinear", align_corners=False)[0].sigmoid()
    scores = {}
    inst = np.asarray(instance_map)
    for mid in np.unique(inst):
        mid = int(mid)
        if mid <= 0:
            continue
        target = torch.from_numpy((inst == mid).copy()).to(masks.device)
        area = target.sum().float().clamp_min(1.0)
        intersection = (masks * target[None]).sum((-2, -1))
        union = masks.sum((-2, -1)) + area - intersection
        soft_iou = intersection / union.clamp_min(1e-6)
        scores[mid] = {}
        for index, cid in enumerate(class_ids):
            value = torch.nan_to_num(class_prob[:, index] * soft_iou, nan=0.0, posinf=0.0, neginf=0.0)
            scores[mid][int(cid)] = float(value.max().clamp(0.0, 1.0).item())
    return scores


def reconcile_instances(previous, proposed, anchors, epoch, cfg, return_stats=False):
    """Hysteresis update: anchors immutable, additions harder than retention."""
    ucfg = cfg.get("pseudo_update", {})
    add_threshold = float(ucfg.get("add_threshold", 0.75))
    keep_threshold = float(ucfg.get("keep_threshold", 0.55))
    class_change_margin = float(ucfg.get("class_change_margin", 0.15))
    previous_by_id = {int(x.mask_id): x for x in (previous or [])}
    proposed_by_id = {int(x.mask_id): x for x in proposed}
    result = []
    stats = {
        "bank_anchored": 0,
        "bank_added": 0,
        "bank_retained": 0,
        "bank_relabelled": 0,
        "bank_dropped": 0,
        "bank_rejected_add": 0,
    }

    for mid, cid in anchors.items():
        old = previous_by_id.get(int(mid))
        result.append(PseudoInstance(int(mid), int(cid), 1.0, "point", True, old.first_epoch if old else 0, epoch))
        stats["bank_anchored"] += 1

    for mid in sorted(set(previous_by_id) | set(proposed_by_id)):
        if mid in anchors:
            continue
        old, new = previous_by_id.get(mid), proposed_by_id.get(mid)
        if old is None and new is not None and float(new.confidence) >= add_threshold:
            new.first_epoch = epoch
            new.last_epoch = epoch
            result.append(new)
            stats["bank_added"] += 1
            continue
        if old is None:
            stats["bank_rejected_add"] += 1
            continue
        if new is None:
            if float(old.confidence) >= keep_threshold:
                old.confidence *= float(ucfg.get("missing_decay", 0.9))
                old.last_epoch = epoch
                result.append(old)
                stats["bank_retained"] += 1
            else:
                stats["bank_dropped"] += 1
            continue
        if int(new.class_id) != int(old.class_id) and float(new.confidence) < float(old.confidence) + class_change_margin:
            old.confidence = max(float(old.confidence) * 0.95, keep_threshold)
            old.last_epoch = epoch
            result.append(old)
            stats["bank_retained"] += 1
        elif float(new.confidence) >= keep_threshold:
            new.first_epoch = old.first_epoch
            new.last_epoch = epoch
            result.append(new)
            if int(new.class_id) != int(old.class_id):
                stats["bank_relabelled"] += 1
            else:
                stats["bank_retained"] += 1
        else:
            stats["bank_dropped"] += 1
    return (result, stats) if return_stats else result
