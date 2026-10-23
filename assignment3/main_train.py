"""
Unified training launcher for Assignment 3.

Each part is dispatched to its original script, run from that script's own
directory (the scripts use relative data/output paths). Any extra arguments are
forwarded unchanged:
    - Part A (Hydra):  config overrides, e.g.  training.num_epochs=10
    - Part B (argparse): script flags,  e.g.  --num_itrs 500

Examples:
    python main_train.py --part 3
    python main_train.py --part 4.2 training.num_epochs=100
    python main_train.py --part 6 --num_itrs 1000
    python main_train.py --part 7.2 --gaussians_per_splat 2048
"""
import os
import sys
import argparse
import subprocess

ROOT = os.path.dirname(os.path.abspath(__file__))

# part -> (working directory, script + fixed args, description)
PARTS = {
    "2":   ("nerf", ["volume_rendering_main.py", "--config-name=train_box"],
            "Part A Q2: optimizing a basic implicit volume (box)"),
    "3":   ("nerf", ["volume_rendering_main.py", "--config-name=nerf_lego"],
            "Part A Q3: NeRF on lego"),
    "4.1": ("nerf", ["volume_rendering_main.py", "--config-name=nerf_materials"],
            "Part A Q4.1: NeRF on materials (view dependence set in configs/nerf_materials.yaml)"),
    "4.2": ("nerf", ["volume_rendering_main.py", "--config-name=nerf_materials_hierarchical"],
            "Part A Q4.2: NeRF on materials with hierarchical sampling"),
    "4.3": ("nerf", ["volume_rendering_main.py", "--config-name=volsdf_surface"],
            "Part A Q4.3: VolSDF surface on lego"),
    "6":   ("GS", ["train.py"],
            "Part B Q6: training isotropic 3D Gaussians on the truck scene"),
    "7.2": ("GS", ["train_harder_scene.py"],
            "Part B Q7.2: training 3D Gaussians on the harder materials scene"),
}


def launch(parts, description, argv=None):

    parser = argparse.ArgumentParser(
        description=description,
        epilog="Parts:\n" + "\n".join(f"  {k:<5} {v[2]}" for k, v in parts.items())
               + "\n\nAny other arguments are forwarded to the underlying script.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--part", required=True, choices=list(parts), help="Assignment part to run.")
    parser.add_argument("--dry_run", action="store_true", help="Print the command without running it.")
    args, forwarded = parser.parse_known_args(argv)

    subdir, script_args, desc = parts[args.part]
    cwd = os.path.join(ROOT, subdir)
    cmd = [sys.executable] + script_args + forwarded

    print(f"[main] Part {args.part}: {desc}")
    print(f"[main] cwd: {cwd}")
    print(f"[main] cmd: {' '.join(cmd)}", flush=True)
    if args.dry_run:
        return 0

    return subprocess.call(cmd, cwd=cwd)


if __name__ == "__main__":

    sys.exit(launch(PARTS, "Assignment 3 training launcher."))
