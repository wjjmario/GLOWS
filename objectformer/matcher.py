import numpy as np
import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment


def sample_points(mask_logits, coords):
    """Sample BxNxHxW tensors at normalized BxPx2 xy coordinates."""
    grid = coords.mul(2.0).sub(1.0).unsqueeze(2)
    sampled = F.grid_sample(mask_logits, grid, align_corners=False)
    return sampled[:, 0, :, 0]


def shared_random_points(num_points, device, generator=None):
    return torch.rand((1, int(num_points), 2), device=device, generator=generator)


def pairwise_sigmoid_ce(inputs, targets):
    # inputs QxP, targets TxP -> QxT
    positive = F.binary_cross_entropy_with_logits(inputs[:, None, :], torch.ones_like(inputs[:, None, :]), reduction="none")
    negative = F.binary_cross_entropy_with_logits(inputs[:, None, :], torch.zeros_like(inputs[:, None, :]), reduction="none")
    return (positive * targets[None] + negative * (1.0 - targets[None])).mean(-1)


def pairwise_dice(inputs, targets):
    probs = inputs.sigmoid()
    numerator = 2.0 * torch.einsum("qp,tp->qt", probs, targets)
    denominator = probs.sum(-1)[:, None] + targets.sum(-1)[None]
    return 1.0 - (numerator + 1.0) / (denominator + 1.0)


class HungarianMatcher(torch.nn.Module):
    def __init__(self, cost_class=2.0, cost_mask=5.0, cost_dice=5.0, cost_proto=1.0, num_points=4096):
        super().__init__()
        if cost_class == cost_mask == cost_dice == 0:
            raise ValueError("At least one matching cost must be non-zero")
        self.cost_class = float(cost_class)
        self.cost_mask = float(cost_mask)
        self.cost_dice = float(cost_dice)
        self.cost_proto = float(cost_proto)
        self.num_points = int(num_points)

    @torch.no_grad()
    def forward(self, outputs, targets):
        logits, masks = outputs["pred_logits"], outputs["pred_masks"]
        indices = []
        for batch_index, target in enumerate(targets):
            count = int(target["labels"].numel())
            if count == 0:
                empty = torch.empty(0, dtype=torch.int64, device=logits.device)
                indices.append((empty, empty))
                continue
            class_cost = -logits[batch_index].softmax(-1)[:, target["labels"]]
            coords = shared_random_points(self.num_points, masks.device)
            pred = sample_points(masks[batch_index][:, None], coords.expand(masks.shape[1], -1, -1))
            tgt = sample_points(target["masks"][:, None], coords.expand(count, -1, -1))
            mask_cost = pairwise_sigmoid_ce(pred.float(), tgt.float())
            dice_cost = pairwise_dice(pred.float(), tgt.float())
            confidence = target.get("weights", torch.ones(count, device=logits.device)).clamp(0.05, 1.0)
            cost = self.cost_class * class_cost + confidence[None] * (self.cost_mask * mask_cost + self.cost_dice * dice_cost)
            if (
                self.cost_proto > 0
                and outputs.get("pred_prototypes") is not None
                and target.get("embeddings") is not None
                and target["embeddings"].numel()
            ):
                proto_cost = 1.0 - outputs["pred_prototypes"][batch_index] @ F.normalize(target["embeddings"].float(), dim=1).T
                cost = cost + self.cost_proto * proto_cost
            cost = torch.nan_to_num(cost, nan=1e4, posinf=1e4, neginf=-1e4).cpu().numpy()
            src, dst = linear_sum_assignment(cost)
            indices.append((
                torch.as_tensor(src, dtype=torch.int64, device=logits.device),
                torch.as_tensor(dst, dtype=torch.int64, device=logits.device),
            ))
        return indices
