import os
import json
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from eval_cls import CLASS_NAMES
from utils import (create_dir, load_model, predict, point_orders, gather_points, rotation_matrix, eval_settings,
                   write_json, cls_colors, seg_colors, render_view, save_montage, SEG_BACKGROUND)

SERIES_COLOR = "#2a78d6"
SURFACE_COLOR = "#fcfcfb"
TEXT_COLOR = "#0b0b0b"
MUTED_COLOR = "#52514e"


def point_counts(num_points, fractions):
    counts = []
    for f in fractions:
        if not 0 < f <= 1:
            raise ValueError("--point_fractions must be in (0, 1], got {}".format(f))
        k = max(1, int(round(num_points * f)))
        if k not in counts:
            counts.append(k)
    return counts


def evaluate(model, task, data, labels, idx, rotation, args):
    logits = predict(model, data, args.device, args.batch_size, idx, rotation)
    if task == "cls":
        probs = torch.softmax(logits, dim=-1).numpy()
        pred = probs.argmax(1)
        return {"accuracy": float((pred == labels).mean()), "pred": pred, "prob": probs.max(1)}
    pred = logits.argmax(dim=-1).numpy()
    correct = pred == gather_points(labels, idx)
    return {"accuracy": float(correct.mean()), "pred": pred, "object_accuracy": correct.mean(axis=1)}


def plot_accuracy(results, experiment, task, baseline, args, path):
    x = [r["value"] for r in results]
    y = [100 * r["accuracy"] for r in results]
    fig, ax = plt.subplots(figsize=(4.8, 3.2), facecolor=SURFACE_COLOR)
    ax.set_facecolor(SURFACE_COLOR)
    ax.axhline(100 * baseline, color=MUTED_COLOR, linewidth=1, linestyle="--")
    ax.annotate("baseline {:.1f}%".format(100 * baseline), (1, 100 * baseline), xycoords=("axes fraction", "data"),
                xytext=(-4, 4), textcoords="offset points", ha="right", va="bottom", color=MUTED_COLOR, fontsize=8)
    ax.plot(x, y, color=SERIES_COLOR, linewidth=2, marker="o", markersize=8, markeredgecolor=SURFACE_COLOR, markeredgewidth=2)
    if experiment == "num_points":
        ax.set_xscale("log", base=2)
        ax.minorticks_off()
        ax.set_xlabel("points per object", color=TEXT_COLOR)
    else:
        ax.set_xlabel("rotation about {} axis (degrees)".format(args.rot_axis), color=TEXT_COLOR)
    ax.set_xticks(x)
    ax.set_xticklabels(["{:g}".format(v) for v in x])
    ax.set_ylabel("test accuracy (%)", color=TEXT_COLOR)
    ax.set_title("{} - {}".format("classification" if task == "cls" else "segmentation",
                                  "rotation" if experiment == "rotation" else "point count"), color=TEXT_COLOR, fontsize=10)
    ax.grid(axis="y", color="#e6e5e1", linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED_COLOR)
    ax.tick_params(colors=MUTED_COLOR, labelsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=120, facecolor=SURFACE_COLOR)
    plt.close(fig)


def run_q3(args, task):
    out_dir = os.path.join(args.output_dir, "q3", task)
    create_dir(out_dir)
    checkpoint = args.cls_checkpoint if task == "cls" else args.seg_checkpoint
    model, meta = load_model(checkpoint, task, args.device)
    data = np.load(os.path.join(args.main_dir, task, "data_test.npy"))
    labels = np.load(os.path.join(args.main_dir, task, "label_test.npy")).astype(np.int64)
    order = point_orders(len(data), data.shape[1], args.num_points, args.seed)
    base_idx = order[:, :args.num_points]
    baseline = evaluate(model, task, data, labels, base_idx, None, args)

    reference_path = os.path.join(args.output_dir, "q1_cls" if task == "cls" else "q2_seg", "metrics.json")
    reference = None
    if os.path.isfile(reference_path):
        with open(reference_path) as f:
            reference = json.load(f)["accuracy"]
        if abs(reference - baseline["accuracy"]) > 1e-9:
            print("Q3 {}: baseline {:.6f} differs from {} ({:.6f}); were Q1/Q2 run with other settings?".format(
                task, baseline["accuracy"], reference_path, reference))

    viz_ids = np.sort(np.random.default_rng([args.seed, 3]).choice(len(data), min(args.q3_num_viz, len(data)), replace=False))
    experiments = {
        "rotation": [("{:g}deg".format(a), a, base_idx, rotation_matrix(args.rot_axis, a)) for a in args.rot_angles],
        "num_points": [("{} pts".format(k), k, order[:, :k], None) for k in point_counts(args.num_points, args.point_fractions)],
    }
    settings = eval_settings(args, checkpoint, meta)
    settings.update(rotation_axis=args.rot_axis, rotation_angles_deg=args.rot_angles, point_fractions=args.point_fractions,
                    rotation_convention="points are row vectors p and are replaced by R p, with R the right-handed rotation about "
                                        "the +{} axis; the data is Y-up (objects stand along +Y). Rotation is applied after "
                                        "point selection, so point identity and labels are unchanged".format(args.rot_axis),
                    num_points_convention="the first k indices of the same per-object permutation used for the baseline "
                                          "(nested subsets, sampled without replacement); seg labels use the same indices",
                    visualized_object_ids=viz_ids)

    summary_lines = ["# Q3 robustness - {}".format("classification" if task == "cls" else "segmentation"), "",
                     "Baseline (no rotation, {} points): **{:.4f}**, checkpoint `{}` (epoch {}).".format(
                         args.num_points, baseline["accuracy"], checkpoint, meta.get("epoch"))]
    for experiment, conditions in experiments.items():
        results = []
        rows = [[] for _ in range(len(viz_ids) * (1 if task == "cls" else 2))]
        captions = [[] for _ in rows]
        for name, value, idx, rotation in conditions:
            r = evaluate(model, task, data, labels, idx, rotation, args)
            entry = {"condition": name, "value": value, "num_points": int(idx.shape[1]), "accuracy": r["accuracy"],
                     "delta_pp": 100 * (r["accuracy"] - baseline["accuracy"]), "visualized": []}
            for j, i in enumerate(viz_ids):
                points = torch.from_numpy(gather_points(data[i:i + 1], idx[i:i + 1])[0]).float()
                if rotation is not None:
                    points = points @ rotation.T
                if task == "cls":
                    pred = CLASS_NAMES[r["pred"][i]]
                    entry["visualized"].append({"object_id": int(i), "gt": CLASS_NAMES[labels[i]], "pred": pred, "pred_prob": float(r["prob"][i])})
                    rows[j].append(render_view(points, cls_colors(points), args.device))
                    captions[j].append("#{} {}: {} ({:.2f})".format(i, name, pred, r["prob"][i]))
                else:
                    gt = torch.from_numpy(gather_points(labels[i:i + 1], idx[i:i + 1])[0])
                    acc = float(r["object_accuracy"][i])
                    entry["visualized"].append({"object_id": int(i), "accuracy": acc})
                    rows[2 * j].append(render_view(points, seg_colors(gt), args.device, SEG_BACKGROUND))
                    rows[2 * j + 1].append(render_view(points, seg_colors(torch.from_numpy(r["pred"][i])), args.device, SEG_BACKGROUND))
                    captions[2 * j].append("#{} {} ground truth".format(i, name))
                    captions[2 * j + 1].append("#{} {} pred acc={:.3f}".format(i, name, acc))
            results.append(entry)
            print("Q3 {} {} {}: accuracy {:.4f} ({:+.2f} pp)".format(task, experiment, name, entry["accuracy"], entry["delta_pp"]))

        if rows and rows[0]:
            save_montage(rows, captions, os.path.join(out_dir, experiment + "_examples.png"))
        plot_accuracy(results, experiment, task, baseline["accuracy"], args, os.path.join(out_dir, experiment + "_accuracy.png"))
        write_json(os.path.join(out_dir, experiment + ".json"), {
            "task": task, "experiment": experiment,
            "metric": "object accuracy" if task == "cls" else "point accuracy",
            "baseline_accuracy": baseline["accuracy"],
            "q1_q2_reported_accuracy": reference,
            "conditions": results,
            "settings": settings,
        })
        summary_lines += ["", "## {}".format("Rotation about the {} axis".format(args.rot_axis) if experiment == "rotation" else "Points per object"), "",
                          "| condition | points | accuracy | change (pp) |", "|---|---|---|---|"]
        summary_lines += ["| {condition} | {num_points} | {accuracy:.4f} | {delta_pp:+.2f} |".format(**e) for e in results]
        summary_lines += ["", "![{0}]({0}_accuracy.png)".format(experiment), "", "![{0} examples]({0}_examples.png)".format(experiment)]
    with open(os.path.join(out_dir, "summary.md"), "w") as f:
        f.write("\n".join(summary_lines) + "\n")
    print("Q3 {} outputs -> {}".format(task, out_dir))
