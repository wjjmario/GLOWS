import torch
import torch.nn.functional as F

from .objects import PseudoInstance


def _class_prototypes(nodes, records, class_ids, global_prototypes=None):
    node_by_id = {int(n["mask_id"]): n for n in nodes}
    prototypes = {}
    for cid in class_ids:
        vectors, weights = [], []
        for record in records:
            if int(record.class_id) == int(cid) and int(record.mask_id) in node_by_id:
                vectors.append(node_by_id[int(record.mask_id)]["embedding"])
                weights.append(float(record.confidence))
        if vectors:
            x = torch.stack(vectors)
            w = torch.tensor(weights, device=x.device, dtype=x.dtype)
            prototypes[int(cid)] = F.normalize((x * w[:, None]).sum(0) / w.sum().clamp_min(1e-6), dim=0)
        elif global_prototypes and int(cid) in global_prototypes:
            prototypes[int(cid)] = F.normalize(global_prototypes[int(cid)].to(nodes[0]["embedding"].device), dim=0)
    return prototypes


def propagate_objects(nodes, anchors, class_ids, cfg, epoch=0, global_prototypes=None, decoder_scores=None, seed_records=None):
    """Confidence-aware object graph propagation from immutable point anchors."""
    gcfg = cfg.get("object_graph", {})
    records = [PseudoInstance(mid, cid, 1.0, "point", True, 0, epoch) for mid, cid in anchors.items()]
    anchored_ids = set(anchors)
    # Previously accepted objects carry knowledge forward, but unlike point
    # anchors they remain removable/relabelable by the hysteresis reconciler.
    for old in seed_records or []:
        if int(old.mask_id) in anchored_ids:
            continue
        records.append(PseudoInstance(
            int(old.mask_id), int(old.class_id), float(old.confidence),
            "history", False, int(old.first_epoch), epoch,
        ))
    assigned = {int(record.mask_id) for record in records}
    if not nodes or not records:
        return records
    embeddings = F.normalize(torch.stack([n["embedding"] for n in nodes]), dim=1)
    sim = embeddings @ embeddings.T
    k = min(int(gcfg.get("knn", 12)), max(len(nodes) - 1, 1))
    accept = float(gcfg.get("initial_threshold", 0.72))
    margin_threshold = float(gcfg.get("margin_threshold", 0.08))
    graph_weight = float(gcfg.get("graph_weight", 0.35))
    decoder_weight = float(gcfg.get("decoder_weight", 0.25))
    proto_weight = float(gcfg.get("prototype_weight", 0.65))
    max_rounds = int(gcfg.get("rounds", 3))

    for _ in range(max_rounds):
        prototypes = _class_prototypes(nodes, records, class_ids, global_prototypes)
        if not prototypes:
            break
        additions = []
        record_by_mid = {int(r.mask_id): r for r in records}
        for i, node in enumerate(nodes):
            mid = int(node["mask_id"])
            if mid in assigned:
                continue
            top = torch.topk(sim[i], k=min(k + 1, len(nodes))).indices.tolist()
            neighbor_votes = {int(cid): 0.0 for cid in class_ids}
            for j in top:
                if j == i:
                    continue
                neighbor = record_by_mid.get(int(nodes[j]["mask_id"]))
                if neighbor is not None:
                    neighbor_votes[int(neighbor.class_id)] += max(float(sim[i, j]), 0.0) * float(neighbor.confidence)
            norm = max(sum(neighbor_votes.values()), 1e-6)
            scores = []
            for cid in class_ids:
                cid = int(cid)
                proto_score = max(float(torch.dot(node["embedding"], prototypes[cid])) if cid in prototypes else -1.0, 0.0)
                graph_score = neighbor_votes[cid] / norm
                decoder_score = float(decoder_scores.get(mid, {}).get(cid, 0.0)) if decoder_scores else 0.0
                score = proto_weight * proto_score + graph_weight * graph_score + decoder_weight * decoder_score
                scores.append((score, cid))
            scores.sort(reverse=True)
            best, second = scores[0], scores[1] if len(scores) > 1 else (-1.0, -1)
            if best[0] >= accept and best[0] - second[0] >= margin_threshold:
                additions.append(PseudoInstance(mid, best[1], min(best[0], 1.0), "graph", False, epoch, epoch))
        if not additions:
            break
        for record in additions:
            assigned.add(int(record.mask_id))
        records.extend(additions)
    return records


def update_global_prototypes(current, nodes, records, momentum=0.9):
    class_ids = sorted({int(r.class_id) for r in records})
    observed = _class_prototypes(nodes, records, class_ids)
    out = dict(current or {})
    for cid, vector in observed.items():
        old = out.get(cid)
        out[cid] = vector.detach() if old is None else F.normalize(momentum * old.to(vector.device) + (1.0 - momentum) * vector, dim=0).detach()
    return out
