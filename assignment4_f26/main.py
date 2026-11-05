import argparse
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT_ROOT = ROOT / "outputs" / "reproduce"
DATA_CONFIG = "configs/scannetpp/iphone/data.json"
VGGT_EVAL_FLAGS = [
    "--eval-root", "data/scannetpp/prepared_eval",
    "--num-views", "8",
    "--eval-num-frames", "32",
    "--image-size", "224",
    "--seed", "825",
]

STAGES = {
    "q1": "Q1 unprojection/canonicalization viewers and check_data.py on 4 training samples",
    "q2.1": "Q2.1 Mini-LVSM overfit on one fixed batch for 2000 steps",
    "q2.2-train": "Q2.2 Mini-LVSM baseline training, 100 epochs (configs/lvsm.json)",
    "q2.2-eval": "Q2.2 validation of lvsm/best.pt, S=1,2,4 x correct/shuffled/none",
    "q2.3-train": "Q2.3 Mini-LVSM MSE + 0.1 x L1 training, 100 epochs (configs/lvsm_mse_l1.json)",
    "q2.3-eval": "Q2.3 validation of lvsm_mse_l1/best.pt, S=1,2,4 x correct/shuffled/none",
    "q3.1": "Q3.1 Mini-VGGT single-clip overfit, 40 epochs x 25 steps (configs/vggt.json)",
    "q3.2-train": "Q3.2 Mini-VGGT joint training, 1000 epochs (configs/vggt.json)",
    "q3.2-eval": "Q3.2 full val + test evaluation of vggt_train/best.pt",
    "q3.3-train": "Q3.3 Mini-VGGT depth-only training, 1000 epochs (configs/vggt_depth_only.json)",
    "q3.3-eval": "Q3.3 full val comparison of vggt_train/last.pt vs vggt_depth_only/last.pt",
}


class Step:
    def __init__(self, command, fresh=(), needs=(), after=None):
        self.command = [str(part) for part in command]
        self.fresh = list(fresh)
        self.needs = list(needs)
        self.after = after


def python(*args):
    return [sys.executable, *args]


def lvsm_eval(name, out, ckpt):
    return Step(
        python(
            "-m", "models.lvsm.eval",
            "--checkpoint", ckpt / name / "best.pt",
            "--split", "val",
            "--num-source-views", "1", "2", "4",
            "--conditioning", "correct", "shuffled", "none",
            "--output", out / name / "eval_val",
        ),
        fresh=[out / name / "eval_val"],
        needs=[ckpt / name / "best.pt"],
    )


def vggt_eval(checkpoint, split, report, visualization=None):
    command = python(
        "-m", "models.vggt.eval",
        "--checkpoint", checkpoint,
        "--split", split,
        *VGGT_EVAL_FLAGS,
        "--report", report,
    )
    fresh = [report]
    if visualization is not None:
        command += ["--visualization", visualization]
        fresh.append(visualization)
    return Step(command, fresh=fresh, needs=[checkpoint])


def synergy_summary(directory):
    def write():
        rows = {}
        for run in ("joint_last", "depth_only_last"):
            report = json.loads((directory / f"{run}_val.json").read_text())
            rows[run] = {
                "checkpoint": report["checkpoint"],
                "depth_abs_rel": report["depth_abs_rel"],
                "depth_delta1": report["depth_delta1"],
            }
        (directory / "summary.json").write_text(json.dumps(rows, indent=2) + "\n")
        print(f"\n{'run':<18}{'depth_abs_rel':>15}{'depth_delta1':>15}")
        for run, row in rows.items():
            print(f"{run:<18}{row['depth_abs_rel']:>15.4f}{row['depth_delta1']:>15.4f}")
        print(f"Saved {display(directory / 'summary.json')}")

    return write


def plan(stage, out, ckpt):
    if stage == "q1":
        viewers = [
            Step(
                python(
                    "scripts/visualize_data.py",
                    "--stage", name,
                    "--num-scenes", "5",
                    "--max-points", "10000",
                    "--output", out / name,
                ),
                fresh=[out / name],
            )
            for name in ("unprojection", "canonicalization")
        ]
        check = Step(
            python(
                "scripts/check_data.py",
                "--config", DATA_CONFIG,
                "--split", "train",
                "--max-samples", "4",
                "--report", out / "check_data" / "train.json",
            ),
            fresh=[out / "check_data"],
        )
        return [*viewers, check]
    if stage == "q2.1":
        return [Step(
            python(
                "-m", "models.lvsm.train",
                "--config", "configs/lvsm.json",
                "--overfit", "2000",
                "--output-dir", out / "lvsm_overfit",
            ),
            fresh=[out / "lvsm_overfit"],
        )]
    if stage in ("q2.2-train", "q2.3-train"):
        config, name = {
            "q2.2-train": ("configs/lvsm.json", "lvsm"),
            "q2.3-train": ("configs/lvsm_mse_l1.json", "lvsm_mse_l1"),
        }[stage]
        return [Step(
            python(
                "-m", "models.lvsm.train",
                "--config", config,
                "--epochs", "100",
                "--output-dir", out / name,
            ),
            fresh=[out / name],
        )]
    if stage == "q2.2-eval":
        return [lvsm_eval("lvsm", out, ckpt)]
    if stage == "q2.3-eval":
        return [lvsm_eval("lvsm_mse_l1", out, ckpt)]
    if stage == "q3.1":
        return [Step(
            python(
                "-m", "models.vggt.train",
                "--config", "configs/vggt.json",
                "--overfit",
                "--overfit-scenes", "1",
                "--epochs", "40",
                "--steps-per-epoch", "25",
                "--output", out / "vggt_overfit",
            ),
            fresh=[out / "vggt_overfit"],
        )]
    if stage in ("q3.2-train", "q3.3-train"):
        config, name = {
            "q3.2-train": ("configs/vggt.json", "vggt_train"),
            "q3.3-train": ("configs/vggt_depth_only.json", "vggt_depth_only"),
        }[stage]
        return [Step(
            python(
                "-m", "models.vggt.train",
                "--config", config,
                "--epochs", "1000",
                "--output", out / name,
            ),
            fresh=[out / name],
        )]
    if stage == "q3.2-eval":
        checkpoint = ckpt / "vggt_train" / "best.pt"
        return [
            vggt_eval(
                checkpoint, "val",
                out / "vggt_train" / "eval_val" / "metrics.json",
                out / "vggt_train" / "eval_val" / "index.html",
            ),
            vggt_eval(
                checkpoint, "test",
                out / "vggt_train" / "eval_test" / "metrics.json",
            ),
        ]
    if stage == "q3.3-eval":
        directory = out / "vggt_synergy_eval"
        joint = vggt_eval(
            ckpt / "vggt_train" / "last.pt", "val", directory / "joint_last_val.json"
        )
        depth_only = vggt_eval(
            ckpt / "vggt_depth_only" / "last.pt", "val",
            directory / "depth_only_last_val.json",
        )
        depth_only.fresh.append(directory / "summary.json")
        depth_only.after = synergy_summary(directory)
        return [joint, depth_only]
    raise ValueError(stage)


def display(path):
    path = Path(path)
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def occupied(path):
    if path.is_dir():
        return any(path.iterdir())
    return path.exists()


def preflight(steps, check_needs):
    problems = []
    for step in steps:
        for path in step.fresh:
            if occupied(path):
                problems.append(
                    f"{display(path)} already exists; refusing to overwrite or append "
                    "to it. Choose another --output-root or remove it yourself."
                )
        if check_needs:
            for path in step.needs:
                if not path.is_file():
                    problems.append(
                        f"missing checkpoint {display(path)}; run the matching -train "
                        "stage first or pass --checkpoint-root."
                    )
    return list(dict.fromkeys(problems))


def run(steps, dry_run):
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(
        part for part in (str(ROOT), env.get("PYTHONPATH")) if part
    )
    for step in steps:
        print("+ " + shlex.join(step.command), flush=True)
        if dry_run:
            continue
        result = subprocess.run(step.command, cwd=ROOT, env=env)
        if result.returncode != 0:
            print(
                f"main.py: command failed with exit code {result.returncode}",
                file=sys.stderr,
            )
            sys.exit(result.returncode)
        if step.after is not None:
            step.after()


def stage_listing():
    lines = ["stages:"]
    lines += [f"  {name:<12}{text}" for name, text in STAGES.items()]
    lines.append(f"  {'all':<12}every stage above, in order")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        prog="main.py",
        description="Run the 16-825 Assignment 4 experiments from the project root.",
        epilog=stage_listing(),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--stage", choices=[*STAGES, "all"], help="stage to run")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help="where new results are written (default: outputs/reproduce)",
    )
    parser.add_argument(
        "--checkpoint-root",
        type=Path,
        help="where -eval stages read checkpoints (default: --output-root)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="print the commands and checks without running anything",
    )
    args = parser.parse_args()
    if args.stage is None:
        parser.print_help()
        return
    out = args.output_root.resolve()
    ckpt = (args.checkpoint_root or args.output_root).resolve()
    if args.stage == "all" and ckpt != out:
        parser.error("--stage all trains its own checkpoints; omit --checkpoint-root")
    stages = list(STAGES) if args.stage == "all" else [args.stage]
    steps = [step for stage in stages for step in plan(stage, out, ckpt)]
    problems = preflight(steps, check_needs=args.stage != "all")
    print(f"project root: {ROOT}")
    print(f"output root: {out}")
    if any(step.needs for step in steps):
        print(f"checkpoint root: {ckpt}")
    if problems:
        for problem in problems:
            print(f"main.py: error: {problem}", file=sys.stderr)
        sys.exit(2)
    run(steps, args.dry_run)


if __name__ == "__main__":
    main()
