import os
import csv
import numpy as np
import argparse

import torch
from models import cls_model
from utils import (create_dir, load_weights, load_model, predict, point_orders, gather_points, eval_settings,
                   write_json, viz_cls, cls_colors, render_view, save_montage)

CLASS_NAMES = ["chair", "vase", "lamp"]
CLASS_NAME_EVIDENCE = ("README lists chairs, vases and lamps in that order; the cls label-0 objects are exactly the "
                       "chair-only seg dataset (4489 train / 617 test objects, identical point sets); "
                       "renders of label-1 objects show vases/planters and label-2 objects show lamps")

def create_parser():
    """Creates a parser for command-line arguments.
    """
    parser = argparse.ArgumentParser()

    parser.add_argument('--num_cls_class', type=int, default=3, help='The number of classes')
    parser.add_argument('--num_points', type=int, default=10000, help='The number of points per object to be included in the input data')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--seed', type=int, default=0)

    # Directories and checkpoint/sample iterations
    parser.add_argument('--load_checkpoint', type=str, default='best_model')
    parser.add_argument('--i', type=int, default=0, help="index of the object to visualize")

    parser.add_argument('--test_data', type=str, default='./data/cls/data_test.npy')
    parser.add_argument('--test_label', type=str, default='./data/cls/label_test.npy')
    parser.add_argument('--output_dir', type=str, default='./output')

    parser.add_argument('--exp_name', type=str, default="exp", help='The name of the experiment')

    return parser


def run_q1(args):
    out_dir = os.path.join(args.output_dir, "q1_cls")
    create_dir(os.path.join(out_dir, "gifs"))
    model, meta = load_model(args.cls_checkpoint, "cls", args.device)
    if meta.get("num_classes", len(CLASS_NAMES)) != len(CLASS_NAMES):
        raise ValueError("Q1 expects a {}-class checkpoint".format(len(CLASS_NAMES)))

    data = np.load(os.path.join(args.main_dir, "cls", "data_test.npy"))
    labels = np.load(os.path.join(args.main_dir, "cls", "label_test.npy")).astype(np.int64)
    idx = point_orders(len(data), data.shape[1], args.num_points, args.seed)[:, :args.num_points]
    probs = torch.softmax(predict(model, data, args.device, args.batch_size, idx), dim=-1).numpy()
    preds = probs.argmax(1)
    correct = preds == labels
    confusion = np.zeros((len(CLASS_NAMES), len(CLASS_NAMES)), dtype=np.int64)
    np.add.at(confusion, (labels, preds), 1)

    with open(os.path.join(out_dir, "predictions.csv"), "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["object_id", "gt_id", "gt_name", "pred_id", "pred_name", "correct"] + ["prob_" + n for n in CLASS_NAMES])
        for i in range(len(data)):
            writer.writerow([i, labels[i], CLASS_NAMES[labels[i]], preds[i], CLASS_NAMES[preds[i]], int(correct[i])]
                            + ["{:.6f}".format(p) for p in probs[i]])

    rng = np.random.default_rng([args.seed, 1])
    random_ids = np.sort(rng.choice(len(data), min(args.cls_num_random, len(data)), replace=False))
    failures = {}
    for c, name in enumerate(CLASS_NAMES):
        wrong = np.where((labels == c) & ~correct)[0]
        failures[name] = wrong[np.argsort(-probs[wrong, preds[wrong]], kind="stable")][:args.cls_num_failures].tolist()

    visualizations, rows = [], {"random": ([], []), "failure": ([], [])}
    for kind, i in [("random", i) for i in random_ids] + [("failure", i) for n in CLASS_NAMES for i in failures[n]]:
        points = torch.from_numpy(gather_points(data[i:i + 1], idx[i:i + 1])[0]).float()
        gt, pred = CLASS_NAMES[labels[i]], CLASS_NAMES[preds[i]]
        gif = "gifs/{}_{:03d}_gt-{}_pred-{}.gif".format(kind, i, gt, pred)
        viz_cls(points, os.path.join(out_dir, gif), args.device)
        rows[kind][0].append(render_view(points, cls_colors(points), args.device))
        rows[kind][1].append("{} #{}: {} -> {} ({:.2f})".format(kind, i, gt, pred, probs[i, preds[i]]))
        visualizations.append({"kind": kind, "object_id": int(i), "gt": gt, "pred": pred,
                               "pred_prob": float(probs[i, preds[i]]), "gt_prob": float(probs[i, labels[i]]), "gif": gif})
    filled = [rows[k] for k in ("random", "failure") if rows[k][0]]
    if filled:
        save_montage([r[0] for r in filled], [r[1] for r in filled], os.path.join(out_dir, "overview.png"))

    per_class = {}
    for c, name in enumerate(CLASS_NAMES):
        mask = labels == c
        per_class[name] = {"num_objects": int(mask.sum()), "num_misclassified": int((mask & ~correct).sum()),
                           "accuracy": float(correct[mask].mean()) if mask.any() else None}
    no_failure = [n for n in CLASS_NAMES if not failures[n]]
    metrics = {
        "task": "cls",
        "accuracy": float(correct.mean()),
        "num_objects": len(data),
        "num_correct": int(correct.sum()),
        "class_names": CLASS_NAMES,
        "class_name_evidence": CLASS_NAME_EVIDENCE,
        "per_class": per_class,
        "confusion_matrix_rows_gt_cols_pred": confusion,
        "random_selection": "numpy default_rng([seed, 1]).choice without replacement over all test objects",
        "failure_selection": "per ground-truth class, misclassified objects sorted by predicted-class probability (most confident first)",
        "classes_without_misclassification": no_failure,
        "visualizations": visualizations,
        "settings": eval_settings(args, args.cls_checkpoint, meta),
    }
    write_json(os.path.join(out_dir, "metrics.json"), metrics)

    lines = ["# Q1 classification results", "",
             "Test accuracy: **{:.4f}** ({}/{} objects), checkpoint `{}` (epoch {}).".format(
                 metrics["accuracy"], metrics["num_correct"], len(data), args.cls_checkpoint, meta.get("epoch")), "",
             "| class | objects | misclassified | accuracy |", "|---|---|---|---|"]
    lines += ["| {} | {} | {} | {:.4f} |".format(n, v["num_objects"], v["num_misclassified"], v["accuracy"]) for n, v in per_class.items()]
    lines += ["", "| kind | object | ground truth | prediction (prob) | gif |", "|---|---|---|---|---|"]
    lines += ["| {kind} | {object_id} | {gt} | {pred} ({pred_prob:.3f}) | [{gif}]({gif}) |".format(**v) for v in visualizations]
    lines += [""] + ["No misclassified test objects for class '{}'.".format(n) for n in no_failure]
    with open(os.path.join(out_dir, "summary.md"), "w") as f:
        f.write("\n".join(lines) + "\n")
    print("Q1 classification test accuracy: {:.4f} ({}/{}) -> {}".format(metrics["accuracy"], metrics["num_correct"], len(data), out_dir))
    for n in no_failure:
        print("Q1: no misclassified test objects for class '{}'".format(n))
    return metrics


if __name__ == '__main__':
    parser = create_parser()
    args = parser.parse_args()
    args.device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')
    np.random.seed(args.seed)

    create_dir(args.output_dir)

    # ------ TO DO: Initialize Model for Classification Task ------
    model = cls_model(num_classes=args.num_cls_class).to(args.device)

    # Load Model Checkpoint
    model_path = './checkpoints/cls/{}.pt'.format(args.load_checkpoint)
    load_weights(model, model_path, args.device)
    model.eval()
    print ("successfully loaded checkpoint from {}".format(model_path))


    # Sample Points per Object
    ind = np.random.choice(10000,args.num_points, replace=False)
    test_data = torch.from_numpy((np.load(args.test_data))[:,ind,:])
    test_label = torch.from_numpy(np.load(args.test_label)).to(torch.long)

    # ------ TO DO: Make Prediction ------
    pred_label = predict(model, test_data.numpy(), args.device, args.batch_size).argmax(dim=-1)

    # Compute Accuracy
    test_accuracy = pred_label.eq(test_label.data).cpu().sum().item() / (test_label.size()[0])
    print ("test accuracy: {}".format(test_accuracy))

