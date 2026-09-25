"""
Sample code to render a cow.

Usage:
    python -m starter.render_mesh --image_size 256 --output_path images/cow_render.jpg
"""
import argparse

import matplotlib.pyplot as plt
import pytorch3d
from starter.gif_utils import render_mesh_360, save_gif
import torch

from starter.utils import get_device, get_mesh_renderer, load_cow_mesh


def render_cow(
    cow_path="data/cow.obj", image_size=256, color=[0.7, 0.7, 1], device=None,
):
    # The device tells us whether we are rendering with GPU or CPU. The rendering will
    # be *much* faster if you have a CUDA-enabled NVIDIA GPU. However, your code will
    # still run fine on a CPU.
    # The default is to run on CPU, so if you do not have a GPU, you do not need to
    # worry about specifying the device in all of these functions.
    if device is None:
        device = get_device()

    # Get the renderer.
    renderer = get_mesh_renderer(image_size=image_size)

    # Get the vertices, faces, and textures.
    vertices, faces = load_cow_mesh(cow_path)
    vertices = vertices.unsqueeze(0)  # (N_v, 3) -> (1, N_v, 3)
    faces = faces.unsqueeze(0)  # (N_f, 3) -> (1, N_f, 3)

    color1 = torch.tensor([1.0, 0.0, 0.0]) 
    color2 = torch.tensor([0.0, 1.0, 1.0]) 
    z = vertices[..., 2]  # (1, N_v)
    z_min = z.min()
    z_max = z.max()
    alpha = (z - z_min) / (z_max - z_min)
    alpha = alpha.unsqueeze(-1) 
    textures = alpha * color2 + (1 - alpha) * color1
    mesh = pytorch3d.structures.Meshes(
        verts=vertices,
        faces=faces,
        textures=pytorch3d.renderer.TexturesVertex(textures),
    )
    return mesh


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cow_path", type=str, default="data/cow.obj")
    parser.add_argument("--output_path", type=str, default="output/cow_texture.gif")
    parser.add_argument("--image_size", type=int, default=256)
    args = parser.parse_args()
    mesh = render_cow(
        image_size=args.image_size
    )
    frames = render_mesh_360(
        mesh,
        image_size=args.image_size,
        num_frames=30,
        dist=3.0,
        elev=20.0,
        device=None,
        light_location=[[2, 2, -3]]
    )

    save_gif(
        frames,
        args.output_path,
        fps=15,
    )