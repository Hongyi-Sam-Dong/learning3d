import argparse
import os
import sys

import torch


def create_parser(prog=None):
    parser = argparse.ArgumentParser(
        prog=prog,
        description="Evaluate trained checkpoints (never trains). --stage eval runs Q1 for cls and Q2 for seg; "
                    "--stage q3 runs the rotation and point-count robustness experiments; --stage all runs eval then q3. "
                    "--task selects cls, seg, or all (both).")
    parser.add_argument('--task', choices=["cls", "seg", "all"], default="all")
    parser.add_argument('--stage', choices=["eval", "q3", "all"], default="eval")
    parser.add_argument('--cls_checkpoint', type=str, default='./checkpoints/cls/best_model.pt')
    parser.add_argument('--seg_checkpoint', type=str, default='./checkpoints/seg/best_model.pt')
    parser.add_argument('--main_dir', type=str, default='./data/')
    parser.add_argument('--output_dir', type=str, default='./output')
    parser.add_argument('--num_points', type=int, default=10000, help='Baseline points per object for every stage')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--seed', type=int, default=0, help='Seeds point selection and example selection')
    parser.add_argument('--device', type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument('--cls_num_random', type=int, default=5, help='Q1: random test objects to visualize')
    parser.add_argument('--cls_num_failures', type=int, default=1, help='Q1: misclassified objects to visualize per ground-truth class')
    parser.add_argument('--seg_num_random', type=int, default=3, help='Q2: random test objects to visualize')
    parser.add_argument('--seg_num_worst', type=int, default=2, help='Q2: lowest-accuracy objects to visualize')
    parser.add_argument('--rot_axis', choices=["x", "y", "z"], default="x", help='Q3 rotation axis (data is Y-up; x tilts objects forward/back)')
    parser.add_argument('--rot_angles', type=float, nargs='+', default=[0, 15, 30, 60, 90])
    parser.add_argument('--point_fractions', type=float, nargs='+', default=[1.0, 0.5, 0.25, 0.125], help='Q3 fractions of --num_points')
    parser.add_argument('--q3_num_viz', type=int, default=3, help='Q3: fixed test objects shown under every condition')
    return parser


def main(argv=None, prog=None):
    args = create_parser(prog).parse_args(argv)
    args.device = torch.device(args.device)
    tasks = ["cls", "seg"] if args.task == "all" else [args.task]
    missing = [p for p in (args.cls_checkpoint if t == "cls" else args.seg_checkpoint for t in tasks) if not os.path.isfile(p)]
    if missing:
        sys.exit("missing checkpoint(s): {}. Inference does not train; run python main.py train --task {} first.".format(
            ", ".join(missing), args.task))

    from eval_cls import run_q1
    from eval_seg import run_q2
    from robustness import run_q3
    if args.stage in ("eval", "all"):
        for task in tasks:
            run_q1(args) if task == "cls" else run_q2(args)
    if args.stage in ("q3", "all"):
        for task in tasks:
            run_q3(args, task)


if __name__ == '__main__':
    main()
