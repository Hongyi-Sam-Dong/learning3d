import argparse

import matplotlib.pyplot as plt
import pytorch3d
from starter.gif_utils import render_mesh_360, save_gif
import torch

from starter.utils import get_device, get_mesh_renderer, load_cow_mesh


def render_tetrahydron(
    image_size=256, color=[0.7, 0.7, 1], device=None,
):
    if device is None:
        device = get_device()

    # Get the renderer.
    renderer = get_mesh_renderer(image_size=image_size)

    # Get the vertices, faces, and textures.
    vertices = torch.tensor([
        [-1.0, -1.0, 0.0],
        [ 1.0, -1.0, 0.0],
        [ 0.0,  1.0, 0.0],
        [ 0.0,  0.0, 1.5],
    ])

    vertices = vertices.unsqueeze(0)  # (N_v, 3) -> (1, N_v, 3)
    faces = torch.tensor([
        [0, 2, 1],
        [0, 1, 3],
        [1, 2, 3],
        [2, 0, 3],
    ])
    faces = faces.unsqueeze(0)  # (N_f, 3) -> (1, N_f, 3)
    textures = torch.ones_like(vertices)  # (1, N_v, 3)
    textures = textures * torch.tensor(color)  # (1, N_v, 3)
    mesh = pytorch3d.structures.Meshes(
        verts=vertices,
        faces=faces,
        textures=pytorch3d.renderer.TexturesVertex(textures),
    )
    return mesh.to(device)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output_path",
        type=str,
        default="output/tetrahedron.gif",
    )
    parser.add_argument("--image_size", type=int, default=256)
    args = parser.parse_args()

    mesh = render_tetrahydron(
        image_size=args.image_size
    )
    frames = render_mesh_360(
        mesh,
        image_size=args.image_size,
        num_frames=50,
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