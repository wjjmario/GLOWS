import numpy as np
import torch
import torch.nn.functional as F

from .model import semantic_logits
from .objects import PseudoInstance


def _graph_hw(feature, max_side):
    height, width = feature.shape[-2:]
    scale = min(1.0, float(max_side) / max(height, width))
    return max(1, int(round(height * scale))), max(1, int(round(width * scale)))


def _resize_feature(feature, size):
    resized = F.interpolate(feature.float(), size=size, mode="bilinear", align_corners=False)
    return F.normalize(resized, dim=1)


@torch.no_grad()
def build_patch_graph(feature, instance_map, cfg):
    """Build the normalized LPOSS patch graph from frozen DINOv3 features.

    The graph is intentionally small enough to be rebuilt in the training loop.
    SAM memberships boost edges within an object without forbidding useful
    cross-object propagation.
    """
    lcfg = cfg.get("lposs", {})
    graph_size = _graph_hw(feature, int(lcfg.get("graph_max_side", 32)))
    graph_feature = _resize_feature(feature, graph_size)
    vectors = graph_feature[0].permute(1, 2, 0).reshape(-1, graph_feature.shape[1])
    vectors = F.normalize(vectors, dim=1)
    count = vectors.shape[0]

    similarity = (vectors @ vectors.T).clamp_min(0.0)
    similarity.pow_(float(lcfg.get("gamma", 3.0)))

    yy, xx = torch.meshgrid(
        torch.arange(graph_size[0], device=feature.device, dtype=torch.float32),
        torch.arange(graph_size[1], device=feature.device, dtype=torch.float32),
        indexing="ij",
    )
    locations = torch.stack((yy.flatten(), xx.flatten()), dim=1)
    distance = torch.cdist(locations[None], locations[None], p=2)[0]
    distance = distance.pow(float(lcfg.get("spatial_distance_power", 1.0)))
    similarity.mul_(torch.exp(-float(lcfg.get("spatial_sigma", 0.01)) * distance))

    instance = torch.as_tensor(np.asarray(instance_map).copy(), device=feature.device, dtype=torch.float32)[None, None]
    instance = F.interpolate(instance, size=graph_size, mode="nearest")[0, 0].long().flatten()
    sam_boost = float(lcfg.get("sam_same_object_boost", 0.5))
    if sam_boost > 0:
        same_object = (instance[:, None] == instance[None, :]) & (instance[:, None] > 0)
        similarity.mul_(1.0 + sam_boost * same_object.to(similarity.dtype))

    similarity.fill_diagonal_(0.0)
    k = min(max(1, int(lcfg.get("knn", 64))), max(count - 1, 1))
    values, indices = torch.topk(similarity, k=k, dim=1)
    adjacency = torch.zeros_like(similarity)
    adjacency.scatter_(1, indices, values)
    adjacency = torch.maximum(adjacency, adjacency.T)
    adjacency.fill_diagonal_(0.0)

    degree = adjacency.sum(1).clamp_min(1e-6)
    inv_sqrt = degree.rsqrt()
    normalized = inv_sqrt[:, None] * adjacency * inv_sqrt[None, :]
    return normalized, graph_feature, graph_size, instance.reshape(graph_size)


def _available_class_mask(prototypes, class_ids, device):
    return torch.tensor([int(cid) in prototypes for cid in class_ids], dtype=torch.bool, device=device)


@torch.no_grad()
def prototype_initialization(graph_feature, prototypes, class_ids, temperature):
    vectors = graph_feature[0].permute(1, 2, 0).reshape(-1, graph_feature.shape[1])
    vectors = F.normalize(vectors, dim=1)
    logits = torch.full(
        (vectors.shape[0], len(class_ids)), -1e4,
        device=vectors.device, dtype=torch.float32,
    )
    similarities = torch.full_like(logits, -1.0)
    for class_index, cid in enumerate(class_ids):
        if int(cid) not in prototypes:
            continue
        prototype = F.normalize(prototypes[int(cid)].to(vectors).float(), dim=0)
        score = vectors @ prototype
        similarities[:, class_index] = score
        logits[:, class_index] = score / max(float(temperature), 1e-4)
    probabilities = logits.softmax(1)
    return probabilities, similarities


def _teacher_probabilities(outputs, graph_size, class_ids, available):
    if outputs is None:
        return None
    logits = F.interpolate(
        semantic_logits(outputs).float(), size=graph_size,
        mode="bilinear", align_corners=False,
    )[0].permute(1, 2, 0).reshape(-1, len(class_ids))
    probabilities = logits.softmax(1)
    probabilities[:, ~available] = 0.0
    return probabilities / probabilities.sum(1, keepdim=True).clamp_min(1e-6)


def _history_probabilities(previous, graph_instance, class_ids):
    if not previous:
        return None
    class_to_index = {int(cid): index for index, cid in enumerate(class_ids)}
    out = torch.zeros(
        (graph_instance.numel(), len(class_ids)),
        device=graph_instance.device, dtype=torch.float32,
    )
    flat_instance = graph_instance.flatten()
    for record in previous:
        class_index = class_to_index.get(int(record.class_id))
        if class_index is None:
            continue
        selected = flat_instance == int(record.mask_id)
        if selected.any():
            out[selected, class_index] = max(float(record.confidence), 0.0)
    valid = out.sum(1, keepdim=True) > 0
    normalized = out / out.sum(1, keepdim=True).clamp_min(1e-6)
    return normalized, valid


def _point_indices(points, class_ids, image_size, graph_size, device):
    class_to_index = {int(cid): index for index, cid in enumerate(class_ids)}
    indices, labels = [], []
    image_width, image_height = image_size
    graph_height, graph_width = graph_size
    for cid, locations in points.items():
        if int(cid) not in class_to_index:
            continue
        for x, y in locations:
            gx = int(round(float(x) / max(image_width - 1, 1) * (graph_width - 1)))
            gy = int(round(float(y) / max(image_height - 1, 1) * (graph_height - 1)))
            indices.append(gy * graph_width + gx)
            labels.append(class_to_index[int(cid)])
    return (
        torch.tensor(indices, dtype=torch.long, device=device),
        torch.tensor(labels, dtype=torch.long, device=device),
    )


@torch.no_grad()
def propagate_patch_labels(feature, instance_map, points, prototypes, class_ids, cfg, teacher_outputs=None, previous=None):
    lcfg = cfg.get("lposs", {})
    graph, graph_feature, graph_size, graph_instance = build_patch_graph(feature, instance_map, cfg)
    y0, prototype_similarity = prototype_initialization(
        graph_feature, prototypes, class_ids,
        temperature=float(lcfg.get("prototype_temperature", 0.07)),
    )
    available = _available_class_mask(prototypes, class_ids, feature.device)
    if not bool(available.any()):
        height, width = np.asarray(instance_map).shape
        return {
            "probabilities": torch.zeros((len(class_ids), height, width), device=feature.device),
            "semantic": np.full((height, width), 255, dtype=np.uint16),
            "confidence": np.zeros((height, width), dtype=np.float32),
            "margin": np.zeros((height, width), dtype=np.float32),
            "graph_size": graph_size,
        }

    teacher = _teacher_probabilities(teacher_outputs, graph_size, class_ids, available)
    teacher_weight = float(lcfg.get("teacher_weight", 0.35)) if teacher is not None else 0.0
    if teacher is not None and teacher_weight > 0:
        y0 = (1.0 - teacher_weight) * y0 + teacher_weight * teacher

    history = _history_probabilities(previous, graph_instance, class_ids)
    if history is not None:
        history_values, history_valid = history
        history_weight = float(lcfg.get("history_weight", 0.20))
        y0 = torch.where(
            history_valid,
            (1.0 - history_weight) * y0 + history_weight * history_values,
            y0,
        )

    # A point-matched SAM object is a strong initialization, while only the
    # exact point is clamped throughout propagation.
    class_to_index = {int(cid): index for index, cid in enumerate(class_ids)}
    anchor_strength = float(lcfg.get("sam_anchor_strength", 0.8))
    flat_instance = graph_instance.flatten()
    anchor_classes = {}
    for cid, locations in points.items():
        for x, y in locations:
            gx = int(round(float(x) / max(instance_map.shape[1] - 1, 1) * (graph_size[1] - 1)))
            gy = int(round(float(y) / max(instance_map.shape[0] - 1, 1) * (graph_size[0] - 1)))
            mid = int(graph_instance[gy, gx])
            if mid > 0:
                anchor_classes.setdefault(mid, set()).add(int(cid))
    for mid, classes in anchor_classes.items():
        if len(classes) != 1:
            continue
        cid = next(iter(classes))
        if cid not in class_to_index:
            continue
        selected = flat_instance == mid
        one_hot = F.one_hot(
            torch.full((int(selected.sum()),), class_to_index[cid], device=feature.device),
            num_classes=len(class_ids),
        ).float()
        y0[selected] = (1.0 - anchor_strength) * y0[selected] + anchor_strength * one_hot

    point_index, point_label = _point_indices(
        points, class_ids, (instance_map.shape[1], instance_map.shape[0]), graph_size, feature.device,
    )
    if point_index.numel():
        y0[point_index] = F.one_hot(point_label, num_classes=len(class_ids)).float()

    alpha = float(lcfg.get("alpha", 0.95))
    labels = y0.clone()
    for _ in range(max(1, int(lcfg.get("iterations", 10)))):
        labels = alpha * (graph @ labels) + (1.0 - alpha) * y0
        labels.clamp_min_(0.0)
        if point_index.numel():
            labels[point_index] = F.one_hot(point_label, num_classes=len(class_ids)).float()

    # Row normalization is applied only after the linear fixed-point updates.
    # Normalizing inside the loop makes the propagation nonlinear and lets
    # high-degree semantic modes dominate the graph.
    labels = labels / labels.sum(1, keepdim=True).clamp_min(1e-6)

    graph_height, graph_width = graph_size
    probabilities = labels.reshape(graph_height, graph_width, len(class_ids)).permute(2, 0, 1)[None]
    probabilities = F.interpolate(
        probabilities, size=np.asarray(instance_map).shape,
        mode="bilinear", align_corners=False,
    )[0]
    probabilities = probabilities / probabilities.sum(0, keepdim=True).clamp_min(1e-6)

    values, indices = probabilities.topk(k=min(2, len(class_ids)), dim=0)
    confidence = values[0]
    margin = values[0] - (values[1] if values.shape[0] > 1 else 0.0)
    prototype_support = prototype_similarity.max(1).values.reshape(graph_height, graph_width)[None, None]
    prototype_support = F.interpolate(
        prototype_support, size=np.asarray(instance_map).shape,
        mode="bilinear", align_corners=False,
    )[0, 0]
    valid = (
        (confidence >= float(lcfg.get("semantic_confidence_threshold", 0.60)))
        & (margin >= float(lcfg.get("semantic_margin_threshold", 0.10)))
        & (prototype_support >= float(lcfg.get("minimum_prototype_similarity", 0.35)))
    )
    semantic = np.full(np.asarray(instance_map).shape, 255, dtype=np.uint16)
    index_np = indices[0].cpu().numpy()
    valid_np = valid.cpu().numpy()
    for class_index, cid in enumerate(class_ids):
        semantic[valid_np & (index_np == class_index)] = int(cid)

    return {
        "probabilities": probabilities,
        "semantic": semantic,
        "confidence": confidence.cpu().numpy().astype(np.float32),
        "margin": margin.cpu().numpy().astype(np.float32),
        "graph_size": graph_size,
    }


@torch.no_grad()
def proposal_records_from_probabilities(probabilities, nodes, anchors, class_ids, cfg, epoch):
    lcfg = cfg.get("lposs", {})
    threshold = float(lcfg.get("proposal_confidence_threshold", 0.60))
    margin_threshold = float(lcfg.get("proposal_margin_threshold", 0.08))
    records = []
    for node in nodes:
        mid = int(node["mask_id"])
        if mid in anchors:
            records.append(PseudoInstance(mid, int(anchors[mid]), 1.0, "point", True, epoch, epoch))
            continue
        mask = torch.as_tensor(node["mask"].copy(), device=probabilities.device, dtype=torch.bool)
        if not mask.any():
            continue
        score = probabilities[:, mask].mean(1)
        values, indices = score.topk(k=min(2, len(class_ids)))
        margin = float(values[0] - (values[1] if values.numel() > 1 else 0.0))
        confidence = float(values[0])
        if confidence < threshold or margin < margin_threshold:
            continue
        records.append(PseudoInstance(
            mid, int(class_ids[int(indices[0])]), confidence,
            "lposs", False, epoch, epoch,
        ))
    return records


@torch.no_grad()
def refine_pseudo_labels(pseudo, instance_map, points, class_ids, cfg, epoch, teacher_outputs=None, previous=None):
    propagated = propagate_patch_labels(
        pseudo["feature"], instance_map, points, pseudo["prototypes"], class_ids, cfg,
        teacher_outputs=teacher_outputs, previous=previous,
    )
    records = proposal_records_from_probabilities(
        propagated["probabilities"], pseudo["nodes"], pseudo["anchors"], class_ids, cfg, epoch,
    )
    propagated["records"] = records
    return propagated
