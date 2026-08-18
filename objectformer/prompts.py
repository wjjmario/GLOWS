"""Point-only weak annotations.

This module intentionally exposes no box construction API.  Keeping the point
protocol separate makes accidental supervision-strength leakage impossible.
"""

import numpy as np


def _distance_transform(mask):
    mask = (np.asarray(mask) > 0).astype(np.uint8)
    try:
        import cv2
        return cv2.distanceTransform(mask, cv2.DIST_L2, 5)
    except ImportError:
        # Exact Euclidean distance transform is available in the training env.
        from scipy.ndimage import distance_transform_edt
        return distance_transform_edt(mask).astype(np.float32)


def center_biased_points(label, class_ids, points_per_class=1, seed=42, topk_ratio=0.2):
    """Generate deterministic point prompts.

    The one-point branch is intentionally byte-for-byte equivalent in its
    sampling decisions to the original implementation.  For multi-point
    prompts, a deterministic component-aware prefix is used so that increasing
    ``points_per_class`` adds points to the same annotation set instead of
    silently resampling a different set.
    """
    rng = np.random.default_rng(int(seed))
    label = np.asarray(label)
    points = {}
    for cid in class_ids:
        ys, xs = np.where(label == int(cid))
        if len(xs) == 0 or int(points_per_class) <= 0:
            points[int(cid)] = []
            continue
        dist = _distance_transform(label == int(cid))
        order = np.argsort(dist[ys, xs])[::-1]
        topn = max(int(points_per_class), int(len(order) * float(topk_ratio)))
        candidates = order[:topn]
        if int(points_per_class) == 1:
            # Preserve the historical 1-point protocol exactly.
            chosen = rng.choice(candidates, size=min(int(points_per_class), len(candidates)), replace=False)
        else:
            chosen = _multi_point_prefix(
                label == int(cid), ys, xs, candidates, int(points_per_class), rng,
            )
        points[int(cid)] = [[float(xs[i]), float(ys[i])] for i in chosen]
    return points


def _multi_point_prefix(class_mask, ys, xs, candidates, count, rng):
    """Select a deterministic, spatially spread prefix for a class.

    Connected components are used as a lightweight proxy for distinct objects.
    When a class has fewer components than requested points, remaining points
    are selected from unused high-confidence candidates.  The returned order
    depends only on the class mask and seed, not on the requested count (except
    for the final truncation), making 3-point and 5-point runs comparable.
    """
    count = min(int(count), len(candidates))
    if count <= 0:
        return np.empty((0,), dtype=np.int64)
    try:
        from scipy import ndimage
        components, _ = ndimage.label(np.asarray(class_mask, dtype=np.uint8))
    except ImportError:
        components = np.asarray(class_mask, dtype=np.int32)

    groups = {}
    for candidate in np.asarray(candidates, dtype=np.int64):
        component = int(components[int(ys[candidate]), int(xs[candidate])])
        groups.setdefault(component, []).append(int(candidate))
    component_ids = np.asarray(list(groups), dtype=np.int64)
    rng.shuffle(component_ids)
    selected = []
    for component in component_ids:
        values = np.asarray(groups[int(component)], dtype=np.int64)
        rng.shuffle(values)
        selected.append(int(values[0]))
    selected_set = set(selected)
    remaining = [int(x) for x in np.asarray(candidates, dtype=np.int64) if int(x) not in selected_set]
    rng.shuffle(remaining)
    selected.extend(remaining)
    return np.asarray(selected[:count], dtype=np.int64)


def build_point_prompts(label, cfg, class_ids, seed_offset=0):
    pcfg = cfg.get("prompt", {})
    if bool(pcfg.get("use_box", False)) or bool(pcfg.get("use_text", False)):
        raise ValueError("Point-only protocol forbids use_box/use_text")
    allowed = [int(x) for x in pcfg.get("point_class_ids", class_ids)]
    return center_biased_points(
        label,
        allowed,
        points_per_class=int(pcfg.get("points_per_class", 1)),
        seed=int(pcfg.get("seed", 42)) + int(seed_offset),
        topk_ratio=float(pcfg.get("center_topk_ratio", 0.2)),
    )


def nearest_instance_id(instance_map, x, y, radius=20):
    inst = np.asarray(instance_map)
    h, w = inst.shape
    xi = int(np.clip(round(float(x)), 0, w - 1))
    yi = int(np.clip(round(float(y)), 0, h - 1))
    if int(inst[yi, xi]) > 0:
        return int(inst[yi, xi])
    y0, y1 = max(0, yi - radius), min(h, yi + radius + 1)
    x0, x1 = max(0, xi - radius), min(w, xi + radius + 1)
    patch = inst[y0:y1, x0:x1]
    best = (float("inf"), 0)
    for mid in np.unique(patch):
        if int(mid) <= 0:
            continue
        ys, xs = np.where(patch == mid)
        d2 = float(np.min((xs + x0 - xi) ** 2 + (ys + y0 - yi) ** 2))
        best = min(best, (d2, int(mid)))
    return best[1]


def point_anchor_assignments(instance_map, points, radius=20):
    """Return unambiguous mask_id -> class_id anchor assignments."""
    candidates = {}
    for cid, class_points in points.items():
        for x, y in class_points:
            mid = nearest_instance_id(instance_map, x, y, radius=int(radius))
            if mid > 0:
                candidates.setdefault(mid, set()).add(int(cid))
    return {mid: next(iter(cids)) for mid, cids in candidates.items() if len(cids) == 1}


def canonicalize_point_prompts(points, instance_map, points_per_class=1, radius=20):
    """Merge same-class points that hit the same SAM proposal.

    The historical one-point protocol is a strict no-op. For multi-point
    prompts, each class/proposal pair contributes one representative point.
    Points from different classes that hit one proposal are retained so the
    anchor builder can mark that proposal ambiguous and route it to the
    patch-only fallback rather than silently assigning a class.
    """
    if int(points_per_class) <= 1:
        return points
    seen = set()
    output = {int(cid): [] for cid in points}
    for cid, class_points in points.items():
        cid = int(cid)
        for x, y in class_points:
            mid = nearest_instance_id(instance_map, x, y, radius=int(radius))
            key = (cid, int(mid)) if int(mid) > 0 else None
            if key is not None and key in seen:
                continue
            if key is not None:
                seen.add(key)
            output[cid].append([float(x), float(y)])
    return output
