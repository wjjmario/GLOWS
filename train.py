import argparse
import csv
import os
import platform
import random
import subprocess

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from objectformer.bank import PseudoInstanceBank
from objectformer.criterion import SetCriterion
from objectformer.data import SegDataset, class_ids_from_cfg
from objectformer.encoders import DINOv3Encoder
from objectformer.evaluate import evaluate_image_only
from objectformer.hybrid import build_hybrid_pseudo, rotation_consistent_features
from objectformer.io import ensure_dir, load_yaml, save_json
from objectformer.matcher import HungarianMatcher
from objectformer.model import build_model, semantic_logits
from objectformer.objects import instances_to_targets, proposal_nodes, semantic_from_instances
from objectformer.prompts import build_point_prompts, canonicalize_point_prompts, point_anchor_assignments
from objectformer.prototypes import EpochPrototypeAccumulator
from objectformer.proposals import SAM2ProposalGenerator
from objectformer.self_training import EMAModel, point_signature, proposal_decoder_scores, reconcile_instances
from objectformer.visualization import evenly_spaced_indices, save_label_panels


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def git_metadata():
    root = os.path.dirname(os.path.abspath(__file__))

    def run_git(*args):
        result = subprocess.run(
            ["git", *args],
            cwd=root,
            capture_output=True,
            text=True,
            check=False,
        )
        return result.stdout.strip() if result.returncode == 0 else ""

    commit = run_git("rev-parse", "HEAD")
    status = run_git("status", "--porcelain")
    return {"commit": commit or None, "dirty": bool(status) if commit else None}


def build_criterion(cfg, num_classes):
    lcfg = cfg.get("loss", {})
    matcher = HungarianMatcher(
        cost_class=lcfg.get("class_weight", 2.0),
        cost_mask=lcfg.get("mask_weight", 5.0),
        cost_dice=lcfg.get("dice_weight", 5.0),
        cost_proto=lcfg.get("prototype_weight", 1.0),
        num_points=lcfg.get("num_points", 4096),
    )
    return SetCriterion(
        num_classes,
        matcher,
        {"loss_ce": lcfg.get("class_weight", 2.0), "loss_mask": lcfg.get("mask_weight", 5.0), "loss_dice": lcfg.get("dice_weight", 5.0), "loss_proto": lcfg.get("prototype_weight", 1.0)},
        eos_coef=lcfg.get("no_object_weight", 0.1),
        num_points=lcfg.get("num_points", 4096),
    )


def should_update(epoch, cfg):
    ucfg = cfg.get("pseudo_update", {})
    start, interval = int(ucfg.get("start_epoch", 3)), max(1, int(ucfg.get("interval", 1)))
    return epoch >= start and (epoch - start) % interval == 0


def slice_outputs(outputs, index):
    """Keep one image from batched decoder outputs, including auxiliaries."""
    sliced = {
        key: value[index:index + 1] if torch.is_tensor(value) else value
        for key, value in outputs.items() if key != "aux_outputs"
    }
    sliced["aux_outputs"] = [
        {key: value[index:index + 1] for key, value in auxiliary.items()}
        for auxiliary in outputs.get("aux_outputs", [])
    ]
    return sliced


def semantic_batch_loss(outputs, pseudos, class_to_index, device, confidence_weighted=False):
    semantic_targets, semantic_weights = [], []
    for pseudo in pseudos:
        semantic_array = torch.as_tensor(pseudo["semantic"], device=device)
        target = torch.full(semantic_array.shape, 255, dtype=torch.long, device=device)
        for cid, class_index in class_to_index.items():
            target[semantic_array == int(cid)] = int(class_index)
        semantic_targets.append(target)
        if confidence_weighted:
            weight = torch.as_tensor(
                pseudo["semantic_confidence"], device=device, dtype=torch.float32,
            ).clamp(0.05, 1.0)
        else:
            weight = torch.ones_like(target, dtype=torch.float32)
        semantic_weights.append(weight)
    target = torch.stack(semantic_targets)
    weights = torch.stack(semantic_weights)
    prediction = F.interpolate(
        semantic_logits(outputs), size=target.shape[-2:], mode="bilinear", align_corners=False,
    )
    valid = target != 255
    if not bool(valid.any()):
        return prediction.sum() * 0.0
    loss_map = F.cross_entropy(prediction.float(), target, ignore_index=255, reduction="none")
    weights = weights * valid
    # Average image-wise losses so images with more labelled pixels do not
    # dominate the batch compared with the historical single-image path.
    numerator = (loss_map * weights).flatten(1).sum(1)
    denominator = weights.flatten(1).sum(1).clamp_min(1e-6)
    return torch.nan_to_num((numerator / denominator).mean(), nan=0.0, posinf=10.0, neginf=0.0)


def instance_target_records(records, supervision_cfg):
    """Select pseudo instances used by Hungarian instance supervision."""
    mode = str(supervision_cfg.get("instance_target_mode", "all_records")).lower()
    if mode == "all_records":
        return list(records)
    if mode == "anchored_only":
        return [record for record in records if bool(record.anchored)]
    raise ValueError(
        "supervision.instance_target_mode must be 'all_records' or 'anchored_only', "
        f"got {mode!r}"
    )


def disabled_instance_losses(outputs):
    """Zero-valued instance losses without running matching or no-object CE."""
    zero = outputs["pred_logits"].sum() * 0.0
    return {
        "loss_total": zero,
        "loss_ce": zero,
        "loss_mask": zero,
        "loss_dice": zero,
        "loss_proto": zero,
    }


def checkpoint(
    path, epoch, model, ema, optimizer, scaler,
    global_prototypes, anchor_prototypes, cfg, best_miou,
):
    torch.save({
        "epoch": int(epoch), "model": model.state_dict(), "ema": ema.state_dict(),
        "optimizer": optimizer.state_dict(), "scaler": scaler.state_dict(),
        "global_prototypes": {int(k): v.cpu() for k, v in global_prototypes.items()},
        "anchor_prototypes": {int(k): v.cpu() for k, v in anchor_prototypes.items()},
        "config": cfg, "best_miou": float(best_miou),
    }, path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--max-images", type=int, default=-1)
    parser.add_argument("--eval-max-images", type=int, default=-1)
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--resume", default="")
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    seed_everything(int(cfg.get("seed", 42)))
    device = torch.device(args.device)
    class_ids = class_ids_from_cfg(cfg)
    class_to_index = {int(cid): i for i, cid in enumerate(class_ids)}
    train_set = SegDataset(cfg, cfg["dataset"].get("train_split", "train"))
    eval_set = SegDataset(cfg, cfg["dataset"].get("infer_split", "test"))
    items = train_set.items if args.max_images < 0 else train_set.items[:args.max_images]
    tcfg = cfg.get("train", {})
    vcfg = cfg.get("visualization", {})
    train_visual_indices = set(
        evenly_spaced_indices(len(items), int(vcfg.get("num_train_images", 4)))
    ) if bool(vcfg.get("enabled", False)) else set()
    # Optional explicit collection of training pseudo-labels.  This is useful
    # for following one fixed tile across epochs instead of relying on the
    # evenly-spaced visualization samples.
    train_visual_stems = {str(stem) for stem in vcfg.get("train_stems", [])}
    if bool(vcfg.get("enabled", False)) and train_visual_stems:
        train_visual_indices.update(
            index for index, item in enumerate(items)
            if item.get("stem") in train_visual_stems
        )
    out_dir = ensure_dir(args.output_dir or tcfg.get("output_dir", "outputs/objectformer"))
    ckpt_dir = ensure_dir(os.path.join(out_dir, "checkpoints"))
    if bool(vcfg.get("enabled", False)):
        save_json({
            "inference_panel_order": ["prediction", "ground_truth"],
            "pseudo_panel_order": ["instance_pseudo", "dense_dino_pseudo", "ground_truth"],
            "class_names": cfg["task"].get("target_classes", {}),
            "background_id": cfg["task"].get("background_id"),
            "every_epochs": int(vcfg.get("every", 1)),
            "num_train_images": int(vcfg.get("num_train_images", 4)),
            "num_test_images": int(vcfg.get("num_test_images", 6)),
        }, os.path.join(out_dir, "visualizations", "layout.json"))
    configured_bank = cfg.get("pseudo_bank", {}).get("root", os.path.join(out_dir, "pseudo_bank"))
    bank_root = os.path.join(out_dir, "pseudo_bank") if args.output_dir else configured_bank
    bank = PseudoInstanceBank(bank_root)
    save_json({
        "config": args.config,
        "device": args.device,
        "protocol": "point-only/image-only-test",
        "max_images": args.max_images,
        "eval_max_images": args.eval_max_images,
        "requested_epochs": args.epochs,
        "output_dir": out_dir,
        "python": platform.python_version(),
        "torch": torch.__version__,
        "git": git_metadata(),
    }, os.path.join(out_dir, "experiment_meta.json"))
    save_json(cfg, os.path.join(out_dir, "resolved_config.json"))

    encoder = DINOv3Encoder(cfg["encoder"], device)
    sam2 = SAM2ProposalGenerator(cfg["proposal"], device, dataset_name=os.path.basename(cfg["dataset"]["root"]))
    model = build_model(cfg, len(class_ids), encoder.embed_dim).to(device)
    ema = EMAModel(model, decay=float(cfg.get("pseudo_update", {}).get("ema_decay", 0.999)))
    criterion = build_criterion(cfg, len(class_ids)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(tcfg.get("lr", 1e-4)), weight_decay=float(tcfg.get("weight_decay", 0.05)))
    amp_enabled = bool(tcfg.get("amp", True)) and device.type == "cuda"
    amp_dtype_name = str(tcfg.get("amp_dtype", "bfloat16")).lower()
    if amp_dtype_name in {"bf16", "bfloat16"}:
        amp_dtype = torch.bfloat16
    elif amp_dtype_name in {"fp16", "float16", "half"}:
        amp_dtype = torch.float16
    else:
        raise ValueError(f"Unsupported train.amp_dtype: {amp_dtype_name}")
    # BF16 has a wide exponent range and does not need loss scaling.  FP16 is
    # retained as an opt-in compatibility mode, but is guarded against scale
    # underflow and non-finite optimizer updates below.
    scaler_enabled = amp_enabled and amp_dtype == torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=scaler_enabled)
    start_epoch, best_miou, global_prototypes, anchor_prototypes = 1, -1.0, {}, {}
    if args.resume:
        state = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(state["model"])
        ema.load_state_dict(state.get("ema", state["model"]))
        optimizer.load_state_dict(state["optimizer"])
        if "scaler" in state:
            scaler.load_state_dict(state["scaler"])
        global_prototypes = {int(k): v.to(device) for k, v in state.get("global_prototypes", {}).items()}
        anchor_prototypes = {int(k): v.to(device) for k, v in state.get("anchor_prototypes", {}).items()}
        best_miou = float(state.get("best_miou", -1.0))
        start_epoch = int(state["epoch"]) + 1

    epochs = int(args.epochs or tcfg.get("epochs", 20))
    batch_size = max(1, int(tcfg.get("batch_size", 1)))
    accumulation = max(1, int(tcfg.get("grad_accum_steps", 8)))
    supervision_cfg = cfg.get("supervision", {})
    instance_enabled = bool(supervision_cfg.get("instance_enabled", True))
    # Validate once before the long-running training loop.
    instance_target_records([], supervision_cfg)
    fieldnames = ["epoch", "loss", "loss_ce", "loss_mask", "loss_dice", "loss_proto", "loss_semantic", "pseudo_instances", "supervised_instances", "batch_size", "optimizer_steps", "bank_updates", "nonfinite_batches"]
    bank_fields = [
        "bank_anchored", "bank_added", "bank_retained", "bank_relabelled",
        "bank_dropped", "bank_rejected_add",
    ]
    prototype_fields = [
        "prototype_classes", "prototype_candidates_seen", "prototype_candidates_accepted",
        "prototype_rejected_confidence", "prototype_rejected_dino", "prototype_rejected_ema",
        "prototype_anchor_count", "prototype_candidate_count", "prototype_drift_mean",
    ]
    for cid in class_ids:
        prototype_fields.extend([f"anchor_count_{cid}", f"candidate_count_{cid}", f"prototype_drift_{cid}"])
    closed_loop_fields = ["epoch", "global_enabled", "persistent_bank", "agreement_gate"] + bank_fields + prototype_fields
    log_path = os.path.join(out_dir, "train_log.csv")
    if os.path.exists(log_path):
        with open(log_path, "r", encoding="utf-8") as existing:
            existing_header = existing.readline().strip().split(",")
        if existing_header != fieldnames:
            log_path = os.path.join(out_dir, "train_log_stable.csv")
    write_header = not os.path.exists(log_path)
    closed_loop_path = os.path.join(out_dir, "closed_loop_log.csv")
    closed_loop_write_header = not os.path.exists(closed_loop_path)
    with open(log_path, "a", newline="", encoding="utf-8") as log_file, open(
        closed_loop_path, "a", newline="", encoding="utf-8",
    ) as closed_loop_file:
        writer = csv.DictWriter(log_file, fieldnames=fieldnames)
        closed_loop_writer = csv.DictWriter(closed_loop_file, fieldnames=closed_loop_fields)
        if write_header:
            writer.writeheader()
        if closed_loop_write_header:
            closed_loop_writer.writeheader()
        for epoch in range(start_epoch, epochs + 1):
            model.train()
            optimizer.zero_grad(set_to_none=True)
            rows, bank_updates, nonfinite_batches, optimizer_steps = [], 0, 0, 0
            pcfg = cfg.get("global_prototype", {})
            global_enabled = bool(pcfg.get("enabled", False))
            global_use_epoch = int(pcfg.get("use_start_epoch", 2))
            global_update_epoch = int(pcfg.get("update_start_epoch", 3))
            epoch_global_prototypes = (
                {cid: vector.detach().clone() for cid, vector in global_prototypes.items()}
                if global_enabled and epoch >= global_use_epoch else {}
            )
            prototype_accumulator = EpochPrototypeAccumulator(class_ids)
            bank_epoch_stats = {field: 0 for field in bank_fields}
            batches = [(start, items[start:start + batch_size]) for start in range(0, len(items), batch_size)]
            progress = tqdm(batches, total=len(batches), desc=f"epoch {epoch}/{epochs}")
            for batch_index, (batch_start, batch_items) in enumerate(progress):
                images = [train_set.read_item_image(item) for item in batch_items]
                labels = [train_set.read_item_label(item) for item in batch_items]
                points_batch, maps, previous_batch, signatures = [], [], [], []
                for local_index, (item, image, label) in enumerate(zip(batch_items, images, labels)):
                    image_index = batch_start + local_index
                    points = build_point_prompts(label, cfg, class_ids, seed_offset=image_index)
                    previous_payload = bank.load(train_set.split, item["stem"])
                    previous_records = []
                    instance_map = sam2.generate(image, stem=item["stem"], split=train_set.split)
                    points = canonicalize_point_prompts(
                        points, instance_map,
                        points_per_class=int(cfg.get("prompt", {}).get("points_per_class", 1)),
                        radius=int(cfg.get("prompt", {}).get("point_match_radius", 20)),
                    )
                    signature = point_signature(points)
                    if previous_payload is not None and previous_payload.get("point_signature") == signature:
                        previous_records = previous_payload.get("instances", [])
                    points_batch.append(points); maps.append(instance_map)
                    previous_batch.append(previous_records); signatures.append(signature)

                tokens = [x.to(device).float() for x in encoder.extract_intermediate_tokens_batch(images)]
                ema_batch_outputs = None
                if should_update(epoch, cfg):
                    with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=amp_enabled):
                        ema_batch_outputs = ema.model(tokens, *encoder.patch_hw)
                rotation_cache_root = cfg.get("hybrid_anchor", {}).get("cache_dir", "")
                rotation_cache_paths = [
                    os.path.join(rotation_cache_root, train_set.split, item["stem"] + ".pt")
                    if rotation_cache_root else None for item in batch_items
                ]
                features = rotation_consistent_features(encoder, images, rotation_cache_paths)
                pseudos, targets, record_counts, supervised_record_counts = [], [], [], []

                for local_index, (item, image, label, points, instance_map, previous_records, signature, feature) in enumerate(zip(
                    batch_items, images, labels, points_batch, maps, previous_batch, signatures, features,
                )):
                    image_index = batch_start + local_index
                    ema_outputs = slice_outputs(ema_batch_outputs, local_index) if ema_batch_outputs is not None else None
                    decoder_scores = proposal_decoder_scores(ema_outputs, instance_map, class_ids) if ema_outputs is not None else None
                    pseudo = build_hybrid_pseudo(
                        image, encoder, instance_map, points, class_ids, cfg,
                        epoch=epoch, decoder_scores=decoder_scores,
                        global_prototypes=epoch_global_prototypes, feature=feature,
                    )
                    nodes, proposed_records, anchors = pseudo["nodes"], pseudo["records"], pseudo["anchors"]
                    persistent_bank = bool(cfg.get("pseudo_update", {}).get("persistent_bank", False))
                    if persistent_bank:
                        records, image_bank_stats = reconcile_instances(
                            previous_records, proposed_records, anchors, epoch, cfg, return_stats=True,
                        )
                        for key in bank_fields:
                            bank_epoch_stats[key] += int(image_bank_stats[key])
                    else:
                        records = proposed_records
                    pseudo["records"] = records
                    bank.save(train_set.split, item["stem"], records, epoch, signature)
                    bank_updates += int(persistent_bank or decoder_scores is not None)

                    visualize_epoch = epoch % max(1, int(vcfg.get("every", 1))) == 0
                    if bool(vcfg.get("enabled", False)) and visualize_epoch and image_index in train_visual_indices:
                        point_records = [record for record in records if bool(record.anchored)]
                        point_anchor_pseudo = semantic_from_instances(point_records, instance_map)
                        learned_instance_pseudo = semantic_from_instances(records, instance_map)
                        path = os.path.join(out_dir, "visualizations", "pseudo", f"epoch_{epoch:04d}", item["stem"] + "_instance_dense_gt.png")
                        save_label_panels([learned_instance_pseudo, pseudo["semantic"], label], path)
                        save_label_panels([point_anchor_pseudo], os.path.join(out_dir, "visualizations", "pseudo", f"epoch_{epoch:04d}", item["stem"] + "_point_anchor_instance.png"))
                        save_label_panels([learned_instance_pseudo], os.path.join(out_dir, "visualizations", "pseudo", f"epoch_{epoch:04d}", item["stem"] + "_learned_instance_pseudo.png"))

                    if global_enabled:
                        prototype_accumulator.add_anchors(nodes, anchors)
                        if epoch >= global_update_epoch:
                            prototype_accumulator.add_candidates(nodes, records, pseudo["prototypes"], decoder_scores, cfg)
                    max_targets = int(cfg.get("model", {}).get("num_queries", 100))
                    supervised_records = instance_target_records(records, supervision_cfg)
                    target_records = sorted(
                        supervised_records,
                        key=lambda record: (int(record.anchored), float(np.nan_to_num(record.confidence, nan=0.0))),
                        reverse=True,
                    )[:max_targets]
                    targets.append(
                        instances_to_targets(
                            target_records,
                            instance_map,
                            class_to_index,
                            device,
                            nodes=nodes,
                            point_fallbacks=pseudo.get("point_fallbacks", []),
                        )
                    )
                    pseudos.append(pseudo); record_counts.append(len(records))
                    supervised_record_counts.append(len(supervised_records) if instance_enabled else 0)

                with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=amp_enabled):
                    outputs = model(tokens, *encoder.patch_hw)
                    # SetCriterion performs one Hungarian match per image, then
                    # aggregates matched instances over the full batch. Images
                    # containing more pseudo instances therefore contribute
                    # proportionally more to mask/dice supervision.
                    losses = criterion(outputs, targets) if instance_enabled else disabled_instance_losses(outputs)
                    semantic_weight = float(cfg.get("loss", {}).get("semantic_weight", 1.0))
                    if semantic_weight:
                        losses["loss_semantic"] = semantic_batch_loss(
                            outputs, pseudos, class_to_index, device,
                            confidence_weighted=bool(cfg.get("loss", {}).get("semantic_confidence_weighted", False)),
                        )
                    else:
                        losses["loss_semantic"] = outputs["pred_logits"].sum() * 0.0
                    losses["loss_total"] = losses["loss_total"] + semantic_weight * losses["loss_semantic"]
                    loss = losses["loss_total"] / accumulation
                if not bool(torch.isfinite(loss).all()):
                    nonfinite_batches += 1
                    optimizer.zero_grad(set_to_none=True)
                    continue
                if scaler_enabled:
                    scaler.scale(loss).backward()
                else:
                    loss.backward()
                step = (batch_index + 1) % accumulation == 0 or batch_index + 1 == len(batches)
                if step:
                    if scaler_enabled:
                        scaler.unscale_(optimizer)
                    finite_gradients = all(parameter.grad is None or bool(torch.isfinite(parameter.grad).all()) for parameter in model.parameters())
                    if not finite_gradients:
                        nonfinite_batches += 1; optimizer.zero_grad(set_to_none=True)
                        if scaler_enabled: scaler.update()
                        continue
                    try:
                        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), float(tcfg.get("clip_grad_norm", 0.1)), error_if_nonfinite=True)
                    except RuntimeError:
                        nonfinite_batches += 1; optimizer.zero_grad(set_to_none=True)
                        if scaler_enabled: scaler.update()
                        continue
                    if scaler_enabled:
                        scaler.step(optimizer); scaler.update()
                    else:
                        optimizer.step()
                    optimizer.zero_grad(set_to_none=True); ema.update(model); optimizer_steps += 1
                row = {
                    "loss": float(losses["loss_total"].detach()), "loss_ce": float(losses["loss_ce"].detach()),
                    "loss_mask": float(losses["loss_mask"].detach()), "loss_dice": float(losses["loss_dice"].detach()),
                    "loss_proto": float(losses["loss_proto"].detach()), "loss_semantic": float(losses["loss_semantic"].detach()),
                    "pseudo_instances": float(np.mean(record_counts)),
                    "supervised_instances": float(np.mean(supervised_record_counts)),
                    "batch_size": len(batch_items),
                }
                rows.append(row)
                progress.set_postfix(loss=f"{row['loss']:.3f}", batch=len(batch_items), updates=bank_updates)

            if not rows:
                raise RuntimeError(f"Epoch {epoch} produced no finite batches")
            mean = {key: float(np.mean([row[key] for row in rows])) for key in rows[0]}
            mean.update({"epoch": epoch, "optimizer_steps": optimizer_steps, "bank_updates": bank_updates, "nonfinite_batches": nonfinite_batches})
            writer.writerow(mean)
            log_file.flush()
            if global_enabled:
                global_prototypes, anchor_prototypes, prototype_stats = prototype_accumulator.finalize(
                    global_prototypes, anchor_prototypes, cfg,
                )
            else:
                prototype_stats = {field: 0 for field in prototype_fields}
            closed_loop_row = {
                "epoch": epoch,
                "global_enabled": int(global_enabled),
                "persistent_bank": int(bool(cfg.get("pseudo_update", {}).get("persistent_bank", False))),
                "agreement_gate": int(bool(cfg.get("agreement_gate", {}).get("enabled", False))),
                **bank_epoch_stats,
                **prototype_stats,
            }
            closed_loop_writer.writerow(closed_loop_row)
            closed_loop_file.flush()
            save_json(closed_loop_row, os.path.join(out_dir, "closed_loop", f"epoch_{epoch:04d}.json"))
            checkpoint(
                os.path.join(ckpt_dir, "last.pt"), epoch, model, ema, optimizer, scaler,
                global_prototypes, anchor_prototypes, cfg, best_miou,
            )
            if epoch % int(tcfg.get("save_every", 1)) == 0:
                checkpoint(
                    os.path.join(ckpt_dir, f"epoch_{epoch:04d}.pt"), epoch,
                    model, ema, optimizer, scaler,
                    global_prototypes, anchor_prototypes, cfg, best_miou,
                )
            if int(tcfg.get("eval_every", 1)) > 0 and epoch % int(tcfg.get("eval_every", 1)) == 0:
                report = evaluate_image_only(
                    ema.model,
                    encoder,
                    eval_set,
                    class_ids,
                    device,
                    epoch,
                    out_dir,
                    args.eval_max_images,
                    visualization=vcfg,
                )
                if float(report["mIoU"]) > best_miou:
                    best_miou = float(report["mIoU"])
                    checkpoint(
                        os.path.join(ckpt_dir, "best.pt"), epoch,
                        model, ema, optimizer, scaler,
                        global_prototypes, anchor_prototypes, cfg, best_miou,
                    )
                print({"epoch": epoch, "image_only_test": report, "closed_loop": closed_loop_row, **mean})


if __name__ == "__main__":
    main()
