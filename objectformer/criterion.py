import torch
import torch.nn as nn
import torch.nn.functional as F

from .matcher import sample_points, shared_random_points


def dice_loss(logits, targets, weights):
    probs = logits.sigmoid()
    numerator = 2.0 * (probs * targets).sum(1)
    denominator = probs.sum(1) + targets.sum(1)
    loss = 1.0 - (numerator + 1.0) / (denominator + 1.0)
    return (loss * weights).sum() / weights.sum().clamp_min(1e-6)


class SetCriterion(nn.Module):
    def __init__(self, num_classes, matcher, weight_dict, eos_coef=0.1, num_points=4096):
        super().__init__()
        self.num_classes = int(num_classes)
        self.matcher = matcher
        self.weight_dict = dict(weight_dict)
        self.num_points = int(num_points)
        class_weights = torch.ones(self.num_classes + 1)
        class_weights[-1] = float(eos_coef)
        self.register_buffer("class_weights", class_weights)

    @staticmethod
    def _permutation(indices):
        batch = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        source = torch.cat([src for src, _ in indices])
        return batch, source

    def loss_labels(self, outputs, targets, indices):
        logits = outputs["pred_logits"]
        target_classes = torch.full(logits.shape[:2], self.num_classes, dtype=torch.long, device=logits.device)
        for i, (src, dst) in enumerate(indices):
            if src.numel():
                target_classes[i, src] = targets[i]["labels"][dst]
        return {"loss_ce": F.cross_entropy(logits.transpose(1, 2), target_classes, self.class_weights)}

    def loss_masks(self, outputs, targets, indices):
        batch_idx, src_idx = self._permutation(indices)
        if src_idx.numel() == 0:
            zero = outputs["pred_masks"].sum() * 0.0
            return {"loss_mask": zero, "loss_dice": zero}
        source_masks = outputs["pred_masks"][batch_idx, src_idx].float()
        target_masks, weights = [], []
        for target, (_, dst) in zip(targets, indices):
            target_masks.append(target["masks"][dst])
            default_weights = torch.ones(target["labels"].numel(), device=source_masks.device)
            weights.append(target.get("weights", default_weights)[dst])
        target_masks = torch.cat(target_masks, dim=0).to(source_masks)
        weights = torch.cat(weights, dim=0).to(source_masks).clamp(0.05, 1.0)
        coords = shared_random_points(self.num_points, source_masks.device)
        coords = coords.expand(source_masks.shape[0], -1, -1)
        pred = sample_points(source_masks[:, None], coords)
        tgt = sample_points(target_masks[:, None], coords)
        bce = F.binary_cross_entropy_with_logits(pred, tgt, reduction="none").mean(1)
        loss_mask = (bce * weights).sum() / weights.sum().clamp_min(1e-6)
        return {"loss_mask": loss_mask, "loss_dice": dice_loss(pred, tgt, weights)}

    def loss_prototypes(self, outputs, targets, indices):
        batch_idx, src_idx = self._permutation(indices)
        if (
            outputs.get("pred_prototypes") is None
            or src_idx.numel() == 0
            or not all(t.get("embeddings") is not None and t["embeddings"].numel() for t in targets)
        ):
            return {"loss_proto": outputs["pred_logits"].sum() * 0.0}
        target_embeddings, weights = [], []
        for target, (_, dst) in zip(targets, indices):
            target_embeddings.append(target["embeddings"][dst])
            weights.append(target.get("weights", torch.ones(target["labels"].numel(), device=src_idx.device))[dst])
        target_embeddings = F.normalize(torch.cat(target_embeddings).to(outputs["pred_prototypes"]), dim=1)
        predicted = F.normalize(outputs["pred_prototypes"][batch_idx, src_idx].float(), dim=1)
        weights = torch.cat(weights).to(predicted).clamp(0.05, 1.0)
        loss = (1.0 - (predicted * target_embeddings).sum(1))
        value = (loss * weights).sum() / weights.sum().clamp_min(1e-6)
        return {"loss_proto": torch.nan_to_num(value, nan=0.0, posinf=10.0, neginf=0.0)}

    def _one_level(self, outputs, targets):
        indices = self.matcher(outputs, targets)
        losses = {}
        losses.update(self.loss_labels(outputs, targets, indices))
        losses.update(self.loss_masks(outputs, targets, indices))
        losses.update(self.loss_prototypes(outputs, targets, indices))
        return losses

    def forward(self, outputs, targets):
        losses = self._one_level(outputs, targets)
        for i, auxiliary in enumerate(outputs.get("aux_outputs", [])):
            for name, value in self._one_level(auxiliary, targets).items():
                losses[f"{name}_{i}"] = value
        weighted = {}
        total = outputs["pred_logits"].sum() * 0.0
        for name, value in losses.items():
            base_name = name.rsplit("_", 1)[0] if name.rsplit("_", 1)[-1].isdigit() else name
            weight = float(self.weight_dict.get(base_name, 0.0))
            weighted[name] = value
            total = total + weight * value
        weighted["loss_total"] = total
        return weighted
