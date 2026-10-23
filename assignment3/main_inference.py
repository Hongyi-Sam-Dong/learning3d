"""
Unified inference / rendering launcher for Assignment 3.

Only parts with a standalone rendering entry point are listed. NeRF (Q3/Q4)
renders only from inside its training loop, and Q6/Q7.2 do not save trained
gaussians, so those parts produce their renders through main_train.py.

Extra arguments are forwarded unchanged to the underlying script.

Examples:
    python main_inference.py --part 1
    python main_inference.py --part 5
    python main_inference.py --part 7.1 --img_dim 128
"""
import sys

from main_train import launch

# part -> (working directory, script + fixed args, description)
PARTS = {
    "1":   ("nerf", ["volume_rendering_main.py", "--config-name=box"],
            "Part A Q1: volume rendering of an SDF box"),
    "5":   ("GS", ["render.py"],
            "Part B Q5: rendering pre-trained 3D Gaussians with fixed (DC) colours"),
    "7.1": ("GS", ["render.py", "--use_sh"],
            "Part B Q7.1: rendering pre-trained 3D Gaussians with spherical harmonics"),
}

if __name__ == "__main__":

    sys.exit(launch(PARTS, "Assignment 3 inference / rendering launcher."))
