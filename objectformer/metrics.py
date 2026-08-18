import numpy as np


def semantic_metrics(pred, gt, class_ids):
    pred = np.asarray(pred)
    gt = np.asarray(gt)
    rows = {}
    ious = []
    for cid in class_ids:
        p = pred == int(cid)
        g = gt == int(cid)
        inter = float((p & g).sum())
        union = float((p | g).sum())
        iou = inter / union if union > 0 else np.nan
        rows[str(cid)] = iou
        if not np.isnan(iou):
            ious.append(iou)
    return {"mIoU": float(np.mean(ious)) if ious else 0.0, "IoU": rows}


def segmentation_report(preds, gts, class_ids):
    class_ids = [int(x) for x in class_ids]
    tp = {cid: 0.0 for cid in class_ids}
    fp = {cid: 0.0 for cid in class_ids}
    fn = {cid: 0.0 for cid in class_ids}
    correct = 0.0
    total = 0.0

    for pred, gt in zip(preds, gts):
        pred = np.asarray(pred)
        gt = np.asarray(gt)
        valid = np.zeros(gt.shape, dtype=bool)
        for cid in class_ids:
            valid |= gt == cid
        correct += float(((pred == gt) & valid).sum())
        total += float(valid.sum())
        for cid in class_ids:
            p = (pred == cid) & valid
            g = (gt == cid) & valid
            tp[cid] += float((p & g).sum())
            fp[cid] += float((p & ~g).sum())
            fn[cid] += float((~p & g).sum())

    out = {"IoU": {}, "F1": {}}
    ious = []
    f1s = []
    for cid in class_ids:
        denom_iou = tp[cid] + fp[cid] + fn[cid]
        denom_f1 = 2.0 * tp[cid] + fp[cid] + fn[cid]
        iou = tp[cid] / denom_iou if denom_iou > 0 else np.nan
        f1 = 2.0 * tp[cid] / denom_f1 if denom_f1 > 0 else np.nan
        out["IoU"][cid] = float(iou) if not np.isnan(iou) else np.nan
        out["F1"][cid] = float(f1) if not np.isnan(f1) else np.nan
        if not np.isnan(iou):
            ious.append(iou)
        if not np.isnan(f1):
            f1s.append(f1)
    out["mIoU"] = float(np.mean(ious)) if ious else 0.0
    out["mF1"] = float(np.mean(f1s)) if f1s else 0.0
    out["OA"] = float(correct / total) if total > 0 else 0.0
    return out
