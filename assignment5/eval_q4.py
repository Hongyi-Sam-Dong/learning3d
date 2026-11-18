import argparse
import csv
import hashlib
import json
import os
import sys

import numpy as np
import torch

from eval_cls import CLASS_NAMES
from models_q4 import ARCH, build_model
from train_q4 import run_name
from utils import (SEG_BACKGROUND, SEG_COLOR_NAMES, cls_colors, create_dir, gather_points, point_orders, predict,
                   render_view, save_montage, seg_colors, viz_cls, viz_seg, write_json)

TRAIN_KEYS = ("num_points", "num_epochs", "batch_size", "lr", "val_frac", "seed")


def create_parser(prog=None):
    parser = argparse.ArgumentParser(
        prog=prog,
        description="Q4 inference (never trains): evaluates Point Transformer checkpoints on the full test set and compares "
                    "with saved Q1/Q2 results when available. Results go to <output_dir>/<task>_n<num_points>[_tag]/.")
    parser.add_argument('--task', choices=["cls", "seg", "all"], default="all")
    parser.add_argument('--num_points', type=int, default=10000, help='Selects the configuration and the input points per object')
    parser.add_argument('--tag', type=str, default='')
    parser.add_argument('--checkpoint_root', type=str, default='./checkpoints/q4')
    parser.add_argument('--cls_checkpoint', type=str, default=None, help='Default: <checkpoint_root>/cls_n<N>[_tag]/best_model.pt')
    parser.add_argument('--seg_checkpoint', type=str, default=None, help='Default: <checkpoint_root>/seg_n<N>[_tag]/best_model.pt')
    parser.add_argument('--main_dir', type=str, default='./data/')
    parser.add_argument('--output_dir', type=str, default='./output/q4')
    parser.add_argument('--baseline_output_dir', type=str, default='./output', help='Where existing Q1/Q2 results (q1_cls/, q2_seg/) are read from')
    parser.add_argument('--baseline_cls_checkpoint', type=str, default='./checkpoints/cls/best_model.pt')
    parser.add_argument('--baseline_seg_checkpoint', type=str, default='./checkpoints/seg/best_model.pt')
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--seed', type=int, default=0, help='Point selection seed; keep equal to the Q1/Q2 inference seed for matched comparison')
    parser.add_argument('--device', type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument('--cls_num_random', type=int, default=5)
    parser.add_argument('--num_diff', type=int, default=2, help='Objects shown per disagreement type (Q4 better / Q4 worse)')
    parser.add_argument('--seg_num_worst', type=int, default=2)
    parser.add_argument('--seg_num_random', type=int, default=3)
    parser.add_argument('--seg_full_chunked', action='store_true',
                        help='Additionally (seg, num_points < available) predict every point via disjoint N-point chunks; reported separately')
    return parser


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def load_q4_model(path, task, device):
    if not os.path.isfile(path):
        sys.exit("missing Q4 checkpoint {}. Inference does not train; run python main.py q4 train --task {} first.".format(path, task))
    ckpt = torch.load(path, map_location=device)
    if ckpt.get("arch") != ARCH or ckpt.get("task") != task:
        sys.exit("{} is not a {} '{}' checkpoint (arch={}, task={})".format(path, ARCH, task, ckpt.get("arch"), ckpt.get("task")))
    model = build_model(ckpt["model_config"])
    model.load_state_dict(ckpt["model_state_dict"])
    meta = {"path": path, "sha256": sha256(path), "epoch": ckpt.get("epoch"), "val_acc": ckpt.get("val_acc"),
            "model_config": ckpt["model_config"], "config": ckpt.get("config")}
    return model.to(device).eval(), meta


def eval_settings(args, meta, num_available):
    train_points = (meta.get("config") or {}).get("num_points")
    return {
        "checkpoint": meta, "num_points": args.num_points, "num_available_points": num_available, "seed": args.seed,
        "batch_size": args.batch_size, "device": str(args.device), "train_num_points": train_points,
        "train_eval_points_match": train_points == args.num_points,
        "point_sampling": "same as Q1/Q2: per-object permutation from numpy default_rng(seed) via utils.point_orders; "
                          "the first num_points indices form one single model input per object",
        "mode": "model.eval(), torch.no_grad(), batched over objects",
    }


def load_baseline(task, args, q4_meta):
    sub = "q1_cls" if task == "cls" else "q2_seg"
    folder = os.path.join(args.baseline_output_dir, sub)
    files = [os.path.join(folder, "metrics.json"), os.path.join(folder, "predictions.csv" if task == "cls" else "predictions.npz")]
    missing = [f for f in files if not os.path.isfile(f)]
    if missing:
        return None, {"available": False, "reason": "comparison unavailable: missing {}; run python main.py inference --task {} "
                      "(baseline evaluation is never rerun automatically)".format(", ".join(missing), task)}
    with open(files[0]) as f:
        metrics = json.load(f)
    s = metrics.get("settings", {})
    report = {"available": True, "baseline_dir": folder, "baseline_accuracy": metrics.get("accuracy"),
              "baseline_eval_checkpoint": s.get("checkpoint"), "baseline_eval_checkpoint_epoch": s.get("checkpoint_epoch"),
              "baseline_eval_seed": s.get("seed"), "baseline_eval_num_points": s.get("num_points"),
              "q4_eval_seed": args.seed, "q4_eval_num_points": args.num_points,
              "seed_match": s.get("seed") == args.seed, "num_points_match": s.get("num_points") == args.num_points}
    ckpt_path = args.baseline_cls_checkpoint if task == "cls" else args.baseline_seg_checkpoint
    if os.path.isfile(ckpt_path):
        current = torch.load(ckpt_path, map_location="cpu")
        report["baseline_checkpoint_file"] = ckpt_path
        report["baseline_checkpoint_sha256"] = sha256(ckpt_path)
        report["checkpoint_identity"] = ("consistent: current file epoch/val_acc equal the values recorded by the baseline evaluation"
                                         if (current.get("epoch"), current.get("val_acc")) == (s.get("checkpoint_epoch"), s.get("checkpoint_val_acc"))
                                         else "MISMATCH: {} now holds epoch {} (val_acc {}), baseline results used epoch {} (val_acc {})".format(
                                             ckpt_path, current.get("epoch"), current.get("val_acc"), s.get("checkpoint_epoch"), s.get("checkpoint_val_acc")))
    else:
        report["checkpoint_identity"] = "unverified: {} not found".format(ckpt_path)
    base_cfg, q4_cfg = s.get("checkpoint_config") or {}, q4_meta.get("config") or {}
    report["training_differences"] = {k: {"baseline": base_cfg.get(k), "q4": q4_cfg.get(k)} for k in TRAIN_KEYS if base_cfg.get(k) != q4_cfg.get(k)}
    report["model"] = {"baseline": "PointNet without T-Nets (models.py)", "q4": ARCH}
    protocol = []
    if not report["seed_match"]:
        protocol.append("point-selection seed differs ({} vs {})".format(s.get("seed"), args.seed))
    if not report["num_points_match"]:
        protocol.append("baseline evaluated {} points per object, Q4 {} points per object".format(s.get("num_points"), args.num_points))
    if s.get("batch_size") != args.batch_size:
        protocol.append("inference batch size {} vs {} (no effect on predictions in eval mode)".format(s.get("batch_size"), args.batch_size))
    report["protocol_differences"] = protocol
    return metrics, report


def comparison_lines(report):
    if not report["available"]:
        return ["", "**Comparison:** " + report["reason"]]
    lines = ["", "Checkpoint identity (baseline): " + report["checkpoint_identity"],
             "Seed match: {}  |  points per object: baseline {}, Q4 {}".format(report["seed_match"], report["baseline_eval_num_points"], report["q4_eval_num_points"])]
    if report["training_differences"]:
        lines.append("Training differences: " + "; ".join("{} {} vs {}".format(k, v["baseline"], v["q4"]) for k, v in report["training_differences"].items()))
    lines += ["Protocol difference: " + p for p in report["protocol_differences"]]
    return lines


def run_cls(args):
    name = run_name("cls", args.num_points, args.tag)
    ckpt = args.cls_checkpoint or os.path.join(args.checkpoint_root, name, "best_model.pt")
    out_dir = os.path.join(args.output_dir, name)
    model, meta = load_q4_model(ckpt, "cls", args.device)
    create_dir(os.path.join(out_dir, "gifs"))
    data = np.load(os.path.join(args.main_dir, "cls", "data_test.npy"))
    labels = np.load(os.path.join(args.main_dir, "cls", "label_test.npy")).astype(np.int64)
    idx = point_orders(len(data), data.shape[1], args.num_points, args.seed)[:, :args.num_points]
    probs = torch.softmax(predict(model, data, args.device, args.batch_size, idx), dim=-1).numpy()
    preds = probs.argmax(1)
    correct = preds == labels
    confusion = np.zeros((len(CLASS_NAMES), len(CLASS_NAMES)), dtype=np.int64)
    np.add.at(confusion, (labels, preds), 1)

    baseline, report = load_baseline("cls", args, meta)
    base_pred = None
    if baseline is not None:
        with open(os.path.join(report["baseline_dir"], "predictions.csv")) as f:
            rows = list(csv.DictReader(f))
        if len(rows) == len(data) and all(int(r["gt_id"]) == labels[i] and int(r["object_id"]) == i for i, r in enumerate(rows)):
            base_pred = np.array([int(r["pred_id"]) for r in rows])
            base_correct = base_pred == labels
            report.update(
                matched_objects=True,
                point_indices=("identical: both use all {} points".format(data.shape[1]) if args.num_points == data.shape[1] == report["baseline_eval_num_points"]
                               else "derived from recorded seed/num_points (Q1 does not store cls indices); Q4 uses the first {} of the same per-object permutation"
                               .format(args.num_points) if report["seed_match"] else "different point selection (seed mismatch)"),
                baseline_accuracy_recomputed=float(base_correct.mean()), q4_accuracy=float(correct.mean()),
                both_correct=int((correct & base_correct).sum()), q4_only_correct=int((correct & ~base_correct).sum()),
                baseline_only_correct=int((~correct & base_correct).sum()), both_wrong=int((~correct & ~base_correct).sum()),
                same_prediction=int((preds == base_pred).sum()))
        else:
            report.update(available=False, reason="comparison unavailable: Q1 predictions.csv does not match the test set object order/labels")

    with open(os.path.join(out_dir, "predictions.csv"), "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["object_id", "gt_id", "gt_name", "pred_id", "pred_name", "correct"] + ["prob_" + n for n in CLASS_NAMES]
                        + (["q1_pred_id", "q1_correct"] if base_pred is not None else []))
        for i in range(len(data)):
            writer.writerow([i, labels[i], CLASS_NAMES[labels[i]], preds[i], CLASS_NAMES[preds[i]], int(correct[i])]
                            + ["{:.6f}".format(p) for p in probs[i]]
                            + ([base_pred[i], int(base_pred[i] == labels[i])] if base_pred is not None else []))

    groups = [("random", np.sort(np.random.default_rng([args.seed, 1]).choice(len(data), min(args.cls_num_random, len(data)), replace=False)).tolist())]
    conf = probs[np.arange(len(data)), preds]
    if base_pred is not None:
        for kind, mask in (("q4_fixes_q1", correct & (base_pred != labels)), ("q4_breaks_q1", ~correct & (base_pred == labels))):
            ids = np.where(mask)[0]
            groups.append((kind, ids[np.argsort(-conf[ids], kind="stable")][:args.num_diff].tolist()))
    failures = []
    for c in range(len(CLASS_NAMES)):
        ids = np.where((labels == c) & ~correct)[0]
        failures += ids[np.argsort(-conf[ids], kind="stable")][:1].tolist()
    groups.append(("q4_failure", failures))

    rows, captions, visualizations = [], [], []
    for kind, ids in groups:
        if not ids:
            continue
        images, texts = [], []
        for i in ids:
            points = torch.from_numpy(gather_points(data[i:i + 1], idx[i:i + 1])[0]).float()
            gif = "gifs/{}_{:03d}_gt-{}_q4-{}.gif".format(kind, i, CLASS_NAMES[labels[i]], CLASS_NAMES[preds[i]])
            viz_cls(points, os.path.join(out_dir, gif), args.device)
            images.append(render_view(points, cls_colors(points), args.device))
            q1 = " Q1:{}".format(CLASS_NAMES[base_pred[i]]) if base_pred is not None else ""
            texts.append("#{} gt:{}{} Q4:{}".format(i, CLASS_NAMES[labels[i]], q1, CLASS_NAMES[preds[i]]))
            visualizations.append({"kind": kind, "object_id": int(i), "gt": CLASS_NAMES[labels[i]], "q4_pred": CLASS_NAMES[preds[i]],
                                   "q4_prob": float(conf[i]), "q1_pred": CLASS_NAMES[base_pred[i]] if base_pred is not None else None, "gif": gif})
        rows.append(images)
        captions.append(texts)
    if rows:
        save_montage(rows, captions, os.path.join(out_dir, "overview.png"))

    per_class = {n: {"num_objects": int((labels == c).sum()), "num_misclassified": int(((labels == c) & ~correct).sum()),
                     "accuracy": float(correct[labels == c].mean()) if (labels == c).any() else None} for c, n in enumerate(CLASS_NAMES)}
    metrics = {"task": "cls", "arch": ARCH, "accuracy": float(correct.mean()), "num_objects": len(data), "num_correct": int(correct.sum()),
               "class_names": CLASS_NAMES, "per_class": per_class, "confusion_matrix_rows_gt_cols_pred": confusion,
               "visualizations": visualizations,
               "selection": "random = numpy default_rng([seed, 1]) choice (same rule and ids as Q1); q4_fixes_q1 / q4_breaks_q1 = most "
                            "confident Q4 predictions among disagreements; q4_failure = most confident Q4 error per ground-truth class",
               "camera_and_color": "utils.render_view (elev 10, azim 30) and utils.viz_cls 360-degree ring, CLS_COLOR, same as Q1",
               "settings": eval_settings(args, meta, data.shape[1])}
    write_json(os.path.join(out_dir, "metrics.json"), metrics)
    write_json(os.path.join(out_dir, "comparison.json"), report)

    lines = ["# Q4 classification ({})".format(name), "",
             "Test accuracy: **{:.4f}** ({}/{}), checkpoint `{}` (epoch {}, val acc {}), {} points per object.".format(
                 metrics["accuracy"], metrics["num_correct"], len(data), ckpt, meta["epoch"], meta["val_acc"], args.num_points)]
    if report.get("matched_objects"):
        lines += ["", "| model | test accuracy |", "|---|---|",
                  "| Q1 PointNet | {:.4f} |".format(report["baseline_accuracy_recomputed"]), "| Q4 Point Transformer | {:.4f} |".format(metrics["accuracy"]), "",
                  "Same {} test objects. Both correct {}, only Q4 correct {}, only Q1 correct {}, both wrong {}. Point indices: {}.".format(
                      len(data), report["both_correct"], report["q4_only_correct"], report["baseline_only_correct"], report["both_wrong"], report["point_indices"])]
    lines += comparison_lines(report)
    lines += ["", "| class | objects | misclassified | accuracy |", "|---|---|---|---|"]
    lines += ["| {} | {} | {} | {:.4f} |".format(n, v["num_objects"], v["num_misclassified"], v["accuracy"]) for n, v in per_class.items()]
    lines += ["", "| kind | object | gt | Q1 | Q4 (prob) | gif |", "|---|---|---|---|---|---|"]
    lines += ["| {kind} | {object_id} | {gt} | {q1} | {q4_pred} ({q4_prob:.3f}) | [{gif}]({gif}) |".format(q1=v["q1_pred"] or "n/a", **v) for v in visualizations]
    with open(os.path.join(out_dir, "summary.md"), "w") as f:
        f.write("\n".join(lines) + "\n")
    print("Q4 classification test accuracy: {:.4f} ({}/{}) -> {}".format(metrics["accuracy"], metrics["num_correct"], len(data), out_dir))
    print("  comparison: " + ("Q1 {:.4f} on the same objects".format(report["baseline_accuracy_recomputed"]) if report.get("matched_objects") else report.get("reason", "")))
    return metrics


def seg_scores(pred, gt, num_classes):
    correct = pred == gt
    return {"accuracy": float(correct.mean()), "mean_object_accuracy": float(correct.mean(axis=1).mean()),
            "num_points_evaluated": int(correct.size),
            "per_part_id_accuracy": {str(c): (float(correct[gt == c].mean()) if (gt == c).any() else None) for c in range(num_classes)},
            "confusion_matrix_rows_gt_cols_pred": np.bincount((gt * num_classes + pred).ravel(), minlength=num_classes ** 2).reshape(num_classes, num_classes)}


def run_seg(args):
    name = run_name("seg", args.num_points, args.tag)
    ckpt = args.seg_checkpoint or os.path.join(args.checkpoint_root, name, "best_model.pt")
    out_dir = os.path.join(args.output_dir, name)
    model, meta = load_q4_model(ckpt, "seg", args.device)
    num_classes = meta["model_config"]["num_classes"]
    create_dir(os.path.join(out_dir, "gifs"))
    data = np.load(os.path.join(args.main_dir, "seg", "data_test.npy"))
    labels = np.load(os.path.join(args.main_dir, "seg", "label_test.npy")).astype(np.int64)
    order = point_orders(len(data), data.shape[1], args.num_points, args.seed)
    idx = order[:, :args.num_points]
    pred = predict(model, data, args.device, args.batch_size, idx).argmax(dim=-1).numpy()
    gt = gather_points(labels, idx)
    per_object = (pred == gt).mean(axis=1)
    scores = seg_scores(pred, gt, num_classes)
    np.savez_compressed(os.path.join(out_dir, "predictions.npz"), point_indices=idx.astype(np.uint16), pred=pred.astype(np.uint8),
                        gt=gt.astype(np.uint8), object_accuracy=per_object)

    baseline, report = load_baseline("seg", args, meta)
    base_pred = base_full = None
    if baseline is not None:
        npz = np.load(os.path.join(report["baseline_dir"], "predictions.npz"))
        bidx = npz["point_indices"].astype(np.int64)
        if bidx.shape[0] == len(data) and bidx.shape[1] >= args.num_points and np.array_equal(bidx[:, :args.num_points], idx):
            base_pred = npz["pred"][:, :args.num_points].astype(np.int64)
            base_obj = (base_pred == gt).mean(axis=1)
            subset = args.num_points < bidx.shape[1]
            report.update(
                matched_points=True,
                point_indices="verified: Q4 indices equal the first {} of the {} stored Q2 indices per object".format(args.num_points, bidx.shape[1]),
                gt_consistent=bool(np.array_equal(npz["gt"][:, :args.num_points], gt)),
                baseline_accuracy_on_q4_points=float((base_pred == gt).mean()), q4_accuracy=scores["accuracy"],
                baseline_mean_object_accuracy_on_q4_points=float(base_obj.mean()),
                baseline_input_points=int(bidx.shape[1]), q4_input_points=args.num_points,
                matched_note=("Q2 predictions restricted to the Q4 points; Q2 saw all {} points as input, Q4 only {}".format(bidx.shape[1], args.num_points)
                              if subset else "identical point sets and inputs"),
                objects_q4_better=int((per_object > base_obj).sum()), objects_q4_worse=int((per_object < base_obj).sum()),
                objects_equal=int((per_object == base_obj).sum()),
                points_both_correct=int(((pred == gt) & (base_pred == gt)).sum()), points_q4_only=int(((pred == gt) & (base_pred != gt)).sum()),
                points_baseline_only=int(((pred != gt) & (base_pred == gt)).sum()), points_both_wrong=int(((pred != gt) & (base_pred != gt)).sum()))
            if bidx.shape[1] == data.shape[1]:
                base_full = (bidx, npz["pred"].astype(np.int64))
        else:
            report.update(available=False, reason="comparison unavailable: stored Q2 point_indices do not match the Q4 point selection "
                                                  "(different seed or num_points); no matched comparison is claimed")

    chunked = None
    if args.seg_full_chunked and args.num_points < data.shape[1]:
        full = np.empty_like(labels)
        num_chunks = -(-data.shape[1] // args.num_points)
        for c in range(num_chunks):
            chunk = order[:, c * args.num_points:(c + 1) * args.num_points]
            np.put_along_axis(full, chunk, predict(model, data, args.device, args.batch_size, chunk).argmax(dim=-1).numpy(), axis=1)
        chunked = seg_scores(full, labels, num_classes)
        chunked["protocol"] = ("chunked full-object evaluation: each object's {}-point permutation is split into {} disjoint chunks of at most {} "
                               "points, each chunk is a separate model input, and the per-chunk predictions are combined. This is NOT equivalent "
                               "to a single {}-point input; it is reported separately from the subset result".format(
                                   data.shape[1], num_chunks, args.num_points, data.shape[1]))
        if base_full is not None:
            bidx, bpred = base_full
            base_on_full = np.empty_like(labels)
            np.put_along_axis(base_on_full, bidx, bpred, axis=1)
            chunked["baseline_full_accuracy"] = float((base_on_full == labels).mean())
        np.savez_compressed(os.path.join(out_dir, "predictions_chunked_full.npz"), pred=full.astype(np.uint8))

    if base_pred is not None:
        diff = per_object - (base_pred == gt).mean(axis=1)
        ids = [v["object_id"] for v in baseline.get("visualizations", [])]
        groups = [("q2_selection", ids)]
        better = [i for i in np.argsort(-diff, kind="stable") if diff[i] > 0 and i not in ids][:args.num_diff]
        worse = [i for i in np.argsort(diff, kind="stable") if diff[i] < 0 and i not in ids][:args.num_diff]
        groups += [("q4_better", [int(i) for i in better]), ("q4_worse", [int(i) for i in worse])]
    else:
        worst = np.argsort(per_object, kind="stable")[:args.seg_num_worst]
        rest = np.setdiff1d(np.arange(len(data)), worst)
        groups = [("random", np.sort(np.random.default_rng([args.seed, 2]).choice(rest, min(args.seg_num_random, len(rest)), replace=False)).tolist()),
                  ("poor", worst.tolist())]

    rows, captions, visualizations = [], [], []
    for kind, ids in groups:
        for i in ids:
            points = torch.from_numpy(gather_points(data[i:i + 1], idx[i:i + 1])[0]).float()
            entry = {"kind": kind, "object_id": int(i), "q4_accuracy": float(per_object[i]),
                     "gt_gif": "gifs/{}_{:03d}_gt.gif".format(kind, i), "q4_gif": "gifs/{}_{:03d}_q4.gif".format(kind, i)}
            viz_seg(points, torch.from_numpy(gt[i]), os.path.join(out_dir, entry["gt_gif"]), args.device, SEG_BACKGROUND)
            viz_seg(points, torch.from_numpy(pred[i]), os.path.join(out_dir, entry["q4_gif"]), args.device, SEG_BACKGROUND)
            images = [render_view(points, seg_colors(torch.from_numpy(gt[i])), args.device, SEG_BACKGROUND)]
            texts = ["#{} ({}) ground truth".format(i, kind)]
            if base_pred is not None:
                entry["q2_accuracy"] = float((base_pred[i] == gt[i]).mean())
                images.append(render_view(points, seg_colors(torch.from_numpy(base_pred[i])), args.device, SEG_BACKGROUND))
                texts.append("Q2 PointNet acc={:.3f}".format(entry["q2_accuracy"]))
            images.append(render_view(points, seg_colors(torch.from_numpy(pred[i])), args.device, SEG_BACKGROUND))
            texts.append("Q4 PT acc={:.3f}".format(per_object[i]))
            rows.append(images)
            captions.append(texts)
            visualizations.append(entry)
    if rows:
        save_montage(rows, captions, os.path.join(out_dir, "overview.png"))

    selection = {v["object_id"]: v["kind"] for v in visualizations}
    with open(os.path.join(out_dir, "per_object.csv"), "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["object_id", "q4_accuracy"] + (["q2_accuracy_same_points"] if base_pred is not None else []) + ["selected_as"])
        for i in range(len(data)):
            writer.writerow([i, "{:.6f}".format(per_object[i])] + (["{:.6f}".format((base_pred[i] == gt[i]).mean())] if base_pred is not None else [])
                            + [selection.get(i, "")])

    metrics = dict(task="seg", arch=ARCH, num_objects=len(data), **scores)
    metrics.update(part_id_colors={str(c): SEG_COLOR_NAMES[c] for c in range(num_classes)}, visualizations=visualizations,
                   evaluation="subset: one {}-point input per object; accuracy over those points{}".format(
                       args.num_points, "" if args.num_points == data.shape[1] else " only (not all {} points)".format(data.shape[1])),
                   selection=("q2_selection = the objects visualized in Q2; q4_better / q4_worse = largest per-object accuracy gain / loss vs Q2 on the same points"
                              if base_pred is not None else "poor = lowest Q4 per-object accuracies; random = default_rng([seed, 2]) over the rest (Q2 rule applied to Q4)"),
                   camera_and_color="utils.render_view / utils.viz_seg, same part colors and gray background as Q2",
                   chunked_full_object=chunked, settings=eval_settings(args, meta, data.shape[1]))
    write_json(os.path.join(out_dir, "metrics.json"), metrics)
    write_json(os.path.join(out_dir, "comparison.json"), report)

    lines = ["# Q4 segmentation ({})".format(name), "",
             "Test point accuracy: **{:.4f}** ({}), checkpoint `{}` (epoch {}, val acc {}).".format(
                 scores["accuracy"], metrics["evaluation"], ckpt, meta["epoch"], meta["val_acc"])]
    if report.get("matched_points"):
        lines += ["", "| model | point accuracy | mean object accuracy |", "|---|---|---|",
                  "| Q2 PointNet | {:.4f} | {:.4f} |".format(report["baseline_accuracy_on_q4_points"], report["baseline_mean_object_accuracy_on_q4_points"]),
                  "| Q4 Point Transformer | {:.4f} | {:.4f} |".format(scores["accuracy"], scores["mean_object_accuracy"]), "",
                  "Point indices: {}. {}. Objects better / worse / equal under Q4: {} / {} / {}.".format(
                      report["point_indices"], report["matched_note"], report["objects_q4_better"], report["objects_q4_worse"], report["objects_equal"])]
    lines += comparison_lines(report)
    if chunked:
        lines += ["", "Chunked full-object accuracy (separate protocol): {:.4f}{}. {}".format(
            chunked["accuracy"], "; Q2 single-input full accuracy {:.4f}".format(chunked["baseline_full_accuracy"]) if "baseline_full_accuracy" in chunked else "",
            chunked["protocol"])]
    lines += ["", "Colors: " + ", ".join("part {} = {}".format(c, SEG_COLOR_NAMES[c]) for c in range(num_classes)),
              "", "| kind | object | Q2 acc | Q4 acc | ground truth | Q4 prediction |", "|---|---|---|---|---|---|"]
    lines += ["| {kind} | {object_id} | {q2} | {q4_accuracy:.4f} | [{gt_gif}]({gt_gif}) | [{q4_gif}]({q4_gif}) |".format(
        q2="{:.4f}".format(v["q2_accuracy"]) if "q2_accuracy" in v else "n/a", **v) for v in visualizations]
    with open(os.path.join(out_dir, "summary.md"), "w") as f:
        f.write("\n".join(lines) + "\n")
    print("Q4 segmentation test point accuracy: {:.4f} ({}) -> {}".format(scores["accuracy"], metrics["evaluation"], out_dir))
    print("  comparison: " + ("Q2 {:.4f} on the same points".format(report["baseline_accuracy_on_q4_points"]) if report.get("matched_points") else report.get("reason", "")))
    if chunked:
        print("  chunked full-object accuracy (separate protocol): {:.4f}".format(chunked["accuracy"]))
    return metrics


def main(argv=None, prog=None):
    args = create_parser(prog).parse_args(argv)
    args.device = torch.device(args.device)
    tasks = ["cls", "seg"] if args.task == "all" else [args.task]
    for task in tasks:
        path = getattr(args, task + "_checkpoint") or os.path.join(args.checkpoint_root, run_name(task, args.num_points, args.tag), "best_model.pt")
        if not os.path.isfile(path):
            sys.exit("missing Q4 checkpoint {}. Inference does not train; run python main.py q4 train --task {} first.".format(path, task))
    for task in tasks:
        (run_cls if task == "cls" else run_seg)(args)


if __name__ == '__main__':
    main(prog="eval_q4.py")
