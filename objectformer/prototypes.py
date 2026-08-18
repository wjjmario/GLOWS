from collections import defaultdict

import torch
import torch.nn.functional as F


def resolve_class_prototypes(
    local_prototypes,
    global_prototypes=None,
    enabled=False,
    fusion_enabled=False,
    local_weight=0.5,
    global_weight=0.5,
):
    """Resolve per-tile prototypes with optional local/global weighted fusion.

    ``fusion_enabled=False`` preserves the historical behavior: an available
    epoch-frozen global prototype replaces the local prototype.  When fusion is
    enabled, both inputs are normalized before their weighted residual mixture
    is normalized again.
    """
    resolved = {}
    for cid, local in (local_prototypes or {}).items():
        cid = int(cid)
        local = F.normalize(local.to(local.device).float(), dim=0)
        global_vector = (global_prototypes or {}).get(cid) if enabled else None
        if global_vector is None:
            resolved[cid] = local
            continue
        global_vector = F.normalize(global_vector.to(local.device).float(), dim=0)
        if not fusion_enabled:
            resolved[cid] = global_vector
            continue
        local_weight = float(local_weight)
        global_weight = float(global_weight)
        if local_weight < 0.0 or global_weight < 0.0 or local_weight + global_weight <= 0.0:
            raise ValueError("Local/global prototype fusion weights must be non-negative and sum to > 0")
        resolved[cid] = F.normalize(
            local_weight * local + global_weight * global_vector,
            dim=0,
        )
    return resolved


class EpochPrototypeAccumulator:
    """Constant-memory collection followed by exactly one prototype update per epoch."""

    def __init__(self, class_ids):
        self.class_ids = [int(cid) for cid in class_ids]
        self.anchor_sum = {}
        self.anchor_weight = defaultdict(float)
        self.anchor_count = defaultdict(int)
        self.candidate_sum = {}
        self.candidate_weight = defaultdict(float)
        self.candidate_count = defaultdict(int)
        self.seen = 0
        self.accepted = 0
        self.rejected_confidence = 0
        self.rejected_dino = 0
        self.rejected_ema = 0

    @staticmethod
    def _add(sums, weights, counts, cid, vector, weight):
        cid, weight = int(cid), float(weight)
        normalized = F.normalize(vector.detach().float(), dim=0)
        value = normalized * weight
        sums[cid] = value.clone() if cid not in sums else sums[cid] + value
        weights[cid] += weight
        counts[cid] += 1

    def add_anchors(self, nodes, anchors):
        node_by_id = {int(node["mask_id"]): node for node in nodes}
        for mid, cid in anchors.items():
            node = node_by_id.get(int(mid))
            if node is not None:
                self._add(
                    self.anchor_sum, self.anchor_weight, self.anchor_count,
                    cid, node["embedding"], 1.0,
                )

    @staticmethod
    def _top_class(scores):
        ranked = sorted(
            ((float(score), int(cid)) for cid, score in scores.items()), reverse=True,
        )
        if not ranked:
            return None, -1.0, -1.0
        best_score, best_class = ranked[0]
        second_score = ranked[1][0] if len(ranked) > 1 else -1.0
        return best_class, best_score, best_score - second_score

    def add_candidates(self, nodes, records, prototypes, decoder_scores, cfg):
        pcfg = cfg.get("global_prototype", {})
        gate = cfg.get("agreement_gate", {})
        threshold = float(pcfg.get("candidate_threshold", 0.75))
        dino_threshold = float(gate.get("dino_threshold", 0.55))
        dino_margin = float(gate.get("dino_margin", 0.05))
        ema_threshold = float(gate.get("ema_threshold", 0.75))
        gate_enabled = bool(gate.get("enabled", False))
        node_by_id = {int(node["mask_id"]): node for node in nodes}

        for record in records:
            if bool(record.anchored):
                continue
            node = node_by_id.get(int(record.mask_id))
            if node is None:
                continue
            self.seen += 1
            if float(record.confidence) < threshold:
                self.rejected_confidence += 1
                continue

            dino_scores = {
                int(cid): float(torch.dot(node["embedding"].float(), proto.float()))
                for cid, proto in (prototypes or {}).items()
            }
            dino_class, dino_score, margin = self._top_class(dino_scores)
            if gate_enabled and (
                dino_class != int(record.class_id)
                or dino_score < dino_threshold
                or margin < dino_margin
            ):
                self.rejected_dino += 1
                continue

            ema_class, ema_score, _ = self._top_class(
                (decoder_scores or {}).get(int(record.mask_id), {})
            )
            if gate_enabled and (
                ema_class != int(record.class_id) or ema_score < ema_threshold
            ):
                self.rejected_ema += 1
                continue

            self._add(
                self.candidate_sum, self.candidate_weight, self.candidate_count,
                record.class_id, node["embedding"], record.confidence,
            )
            self.accepted += 1

    @staticmethod
    def _centroids(sums, weights):
        return {
            int(cid): F.normalize(total / max(float(weights[cid]), 1e-6), dim=0)
            for cid, total in sums.items()
        }

    def finalize(self, current, anchor_base, cfg):
        pcfg = cfg.get("global_prototype", {})
        momentum = float(pcfg.get("momentum", 0.95))
        anchor_mix = float(pcfg.get("anchor_weight", 0.7))
        candidate_mix = float(pcfg.get("candidate_weight", 0.3))
        epoch_anchors = self._centroids(self.anchor_sum, self.anchor_weight)
        candidates = self._centroids(self.candidate_sum, self.candidate_weight)
        anchors = dict(anchor_base or {})
        anchors.update({cid: vector.detach() for cid, vector in epoch_anchors.items()})
        updated = dict(current or {})
        drift = {}

        for cid in sorted(set(anchors) | set(candidates)):
            parts, weights = [], []
            if cid in anchors:
                parts.append(anchors[cid])
                weights.append(anchor_mix)
            if cid in candidates:
                parts.append(candidates[cid])
                weights.append(candidate_mix)
            if not parts:
                continue
            target = F.normalize(
                sum(vector * weight for vector, weight in zip(parts, weights))
                / max(sum(weights), 1e-6),
                dim=0,
            )
            old = updated.get(cid)
            if old is None:
                new = target.detach()
                drift[cid] = 0.0
            else:
                old = F.normalize(old.to(target.device).float(), dim=0)
                new = F.normalize(momentum * old + (1.0 - momentum) * target, dim=0).detach()
                drift[cid] = float((1.0 - torch.dot(old, new)).clamp_min(0.0).item())
            updated[cid] = new

        diagnostics = {
            "prototype_classes": len(updated),
            "prototype_candidates_seen": self.seen,
            "prototype_candidates_accepted": self.accepted,
            "prototype_rejected_confidence": self.rejected_confidence,
            "prototype_rejected_dino": self.rejected_dino,
            "prototype_rejected_ema": self.rejected_ema,
            "prototype_anchor_count": sum(self.anchor_count.values()),
            "prototype_candidate_count": sum(self.candidate_count.values()),
            "prototype_drift_mean": sum(drift.values()) / max(len(drift), 1),
        }
        for cid in self.class_ids:
            diagnostics[f"anchor_count_{cid}"] = self.anchor_count[cid]
            diagnostics[f"candidate_count_{cid}"] = self.candidate_count[cid]
            diagnostics[f"prototype_drift_{cid}"] = drift.get(cid, 0.0)
        return updated, anchors, diagnostics
