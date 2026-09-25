"""
1.1 360-degree renders.

Usage:
    python -m starter.q1_360
"""
import argparse

import pytorch3d
import torch

from starter.gif_utils import render_mesh_360, save_gif
from starter.utils import load_cow_mesh


def cow_mesh(cow_path="data/cow.obj", color=[0.7, 0.7, 1.0]):
    vertices, faces = load_cow_mesh(cow_path)
    vertices = vertices.unsqueeze(0)  # (N_v, 3) -> (1, N_v, 3)
    faces = faces.unsqueeze(0)  # (N_f, 3) -> (1, N_f, 3)
    textures = torch.ones_like(vertices) * torch.tensor(color)
    return pytorch3d.structures.Meshes(
        verts=vertices,
        faces=faces,
        textures=pytorch3d.renderer.TexturesVertex(textures),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cow_path", type=str, default="data/cow.obj")
    parser.add_argument("--output_path", type=str, default="output/cow_360.gif")
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--num_frames", type=int, default=36)
    args = parser.parse_args()

    frames = render_mesh_360(
        cow_mesh(args.cow_path),
        image_size=args.image_size,
        num_frames=args.num_frames,
    )
    save_gif(frames, args.output_path)
