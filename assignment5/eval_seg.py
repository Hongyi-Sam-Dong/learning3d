import os
import csv
import numpy as np
import argparse

import torch
from models import seg_model
from data_loader import get_data_loader
from utils import (create_dir, viz_seg, load_weights, load_model, predict, point_orders, gather_points, eval_settings,
                   write_json, seg_colors, render_view, save_montage, SEG_COLORS, SEG_COLOR_NAMES, SEG_BACKGROUND)


def create_parser():
    """Creates a parser for command-line arguments.
    """
    parser = argparse.ArgumentParser()

    parser.add_argument('--num_seg_class', type=int, default=6, help='The number of segmentation classes')
    parser.add_argument('--num_points', type=int, default=10000, help='The number of points per object to be included in the input data')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--seed', type=int, default=0)

    # Directories and checkpoint/sample iterations
    parser.add_argument('--load_checkpoint', type=str, default='best_model')
    parser.add_argument('--i', type=int, default=0, help="index of the object to visualize")

    parser.add_argument('--test_data', type=str, default='./data/seg/data_test.npy')
    parser.add_argument('--test_label', type=str, default='./data/seg/label_test.npy')
    parser.add_argument('--output_dir', type=str, default='./output')

    parser.add_argument('--exp_name', type=str, default="exp", help='The name of the experiment')

    return parser


def run_q2(args):
    out_dir = os.path.join(args.output_dir, "q2_seg")
    create_dir(os.path.join(out_dir, "gifs"))
    model, meta = load_model(args.seg_checkpoint, "seg", args.device)
    num_classes = meta.get("num_classes", len(SEG_COLORS))
    if num_classes > len(SEG_COLORS):
        raise ValueError("visualization supports at most {} part ids".format(len(SEG_COLORS)))

    data = np.load(os.path.join(args.main_dir, "seg", "data_test.npy"))
    labels = np.load(os.path.join(args.main_dir, "seg", "label_test.npy")).astype(np.int64)
    idx = point_orders(len(data), data.shape[1], args.num_points, args.seed)[:, :args.num_points]
    pred = predict(model, data, args.device, args.batch_size, idx).argmax(dim=-1).numpy()
    gt = gather_points(labels, idx)
    correct = pred == gt
    per_object = correct.mean(axis=1)
    confusion = np.bincount((gt * num_classes + pred).ravel(), minlength=num_classes * num_classes).reshape(num_classes, num_classes)

    worst = np.argsort(per_object, kind="stable")[:args.seg_num_worst]
    rest = np.setdiff1d(np.arange(len(data)), worst)
    random_ids = np.sort(np.random.default_rng([args.seed, 2]).choice(rest, min(args.seg_num_random, len(rest)), replace=False))
    selection = {int(i): "random" for i in random_ids}
    selection.update({int(i): "poor" for i in worst})

    np.savez_compressed(os.path.join(out_dir, "predictions.npz"), point_indices=idx.astype(np.uint16),
                        pred=pred.astype(np.uint8), gt=gt.astype(np.uint8), object_accuracy=per_object)
    with open(os.path.join(out_dir, "per_object.csv"), "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["object_id", "accuracy", "selected_as"])
        for i in range(len(data)):
            writer.writerow([i, "{:.6f}".format(per_object[i]), selection.get(i, "")])

    visualizations, rows, captions = [], [], []
    for kind, ids in (("random", random_ids), ("poor", worst)):
        for i in ids:
            points = torch.from_numpy(gather_points(data[i:i + 1], idx[i:i + 1])[0]).float()
            gt_labels, pred_labels = torch.from_numpy(gt[i]), torch.from_numpy(pred[i])
            entry = {"kind": kind, "object_id": int(i), "accuracy": float(per_object[i]),
                     "gt_gif": "gifs/{}_{:03d}_gt.gif".format(kind, i), "pred_gif": "gifs/{}_{:03d}_pred.gif".format(kind, i)}
            viz_seg(points, gt_labels, os.path.join(out_dir, entry["gt_gif"]), args.device, SEG_BACKGROUND)
            viz_seg(points, pred_labels, os.path.join(out_dir, entry["pred_gif"]), args.device, SEG_BACKGROUND)
            rows.append([render_view(points, seg_colors(gt_labels), args.device, SEG_BACKGROUND),
                         render_view(points, seg_colors(pred_labels), args.device, SEG_BACKGROUND)])
            captions.append(["#{} ({}) ground truth".format(i, kind), "#{} prediction acc={:.3f}".format(i, per_object[i])])
            visualizations.append(entry)
    if rows:
        save_montage(rows, captions, os.path.join(out_dir, "overview.png"))

    part_accuracy = {str(c): (float(correct[gt == c].mean()) if (gt == c).any() else None) for c in range(num_classes)}
    metrics = {
        "task": "seg",
        "accuracy": float(correct.mean()),
        "num_objects": len(data),
        "num_points_evaluated": int(correct.size),
        "mean_object_accuracy": float(per_object.mean()),
        "per_part_id_accuracy": part_accuracy,
        "confusion_matrix_rows_gt_cols_pred": confusion,
        "part_id_colors": {str(c): SEG_COLOR_NAMES[c] for c in range(num_classes)},
        "part_id_names": "not available in the repository or data; part ids are reported numerically",
        "selection": "poor = the {} lowest per-object accuracies (stable sort); random = numpy default_rng([seed, 2]).choice "
                     "without replacement over the remaining objects".format(args.seg_num_worst),
        "visualizations": visualizations,
        "settings": eval_settings(args, args.seg_checkpoint, meta),
    }
    write_json(os.path.join(out_dir, "metrics.json"), metrics)

    lines = ["# Q2 segmentation results", "",
             "Test point accuracy: **{:.4f}** over {} objects x {} points, checkpoint `{}` (epoch {}).".format(
                 metrics["accuracy"], len(data), args.num_points, args.seg_checkpoint, meta.get("epoch")), "",
             "Colors (same for ground truth and prediction): " + ", ".join("part {} = {}".format(c, SEG_COLOR_NAMES[c]) for c in range(num_classes)), "",
             "| kind | object | accuracy | ground truth | prediction |", "|---|---|---|---|---|"]
    lines += ["| {kind} | {object_id} | {accuracy:.4f} | [{gt_gif}]({gt_gif}) | [{pred_gif}]({pred_gif}) |".format(**v) for v in visualizations]
    with open(os.path.join(out_dir, "summary.md"), "w") as f:
        f.write("\n".join(lines) + "\n")
    print("Q2 segmentation test accuracy: {:.4f} -> {}".format(metrics["accuracy"], out_dir))
    return metrics


if __name__ == '__main__':
    parser = create_parser()
    args = parser.parse_args()
    args.device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')
    np.random.seed(args.seed)

    create_dir(args.output_dir)

    # ------ TO DO: Initialize Model for Segmentation Task  ------
    model = seg_model(num_seg_classes=args.num_seg_class).to(args.device)

    # Load Model Checkpoint
    model_path = './checkpoints/seg/{}.pt'.format(args.load_checkpoint)
    load_weights(model, model_path, args.device)
    model.eval()
    print ("successfully loaded checkpoint from {}".format(model_path))


    # Sample Points per Object
    ind = np.random.choice(10000,args.num_points, replace=False)
    test_data = torch.from_numpy((np.load(args.test_data))[:,ind,:])
    test_label = torch.from_numpy((np.load(args.test_label))[:,ind])

    # ------ TO DO: Make Prediction ------
    pred_label = predict(model, test_data.numpy(), args.device, args.batch_size).argmax(dim=-1)

    test_accuracy = pred_label.eq(test_label.data).cpu().sum().item() / (test_label.reshape((-1,1)).size()[0])
    print ("test accuracy: {}".format(test_accuracy))

    # Visualize Segmentation Result (Pred VS Ground Truth)
    viz_seg(test_data[args.i], test_label[args.i], "{}/gt_{}.gif".format(args.output_dir, args.exp_name), args.device)
    viz_seg(test_data[args.i], pred_label[args.i], "{}/pred_{}.gif".format(args.output_dir, args.exp_name), args.device)
