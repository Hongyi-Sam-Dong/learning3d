import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent

STEPS = [
    ("1.1 360-degree cow", ["starter.q1_360"]),
    ("1.2 dolly zoom", ["starter.dolly_zoom", "--num_frames", "50"]),
    ("2.1 tetrahedron", ["starter.q2_1_tetrahydron"]),
    ("2.2 cube", ["starter.q2_2_cube"]),
    ("3 re-textured cow", ["starter.q3_texture"]),
    ("4 camera transforms", ["starter.camera_transforms"]),
    ("5.1 rgb-d point clouds", ["starter.q5_part1", "--num_frames", "12"]),
    ("5.2 parametric torus and mobius strip", ["starter.q5_part2"]),
    ("5.3 implicit torus and ellipsoid", ["starter.q5_part3"]),
    ("6 smiley", ["starter.q6"]),
    ("7 mesh sampling", ["starter.q7"]),
]


def main():
    (ROOT / "output").mkdir(exist_ok=True)
    for label, command in STEPS:
        print(f"[{label}] python -m {' '.join(command)}", flush=True)
        result = subprocess.run([sys.executable, "-m", *command], cwd=ROOT)
        if result.returncode != 0:
            sys.exit(f"failed: {label}")


if __name__ == "__main__":
    main()
