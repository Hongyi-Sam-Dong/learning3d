import argparse

import mcubes
import numpy as np
import pytorch3d
import torch

from starter.gif_utils import render_mesh_360, save_gif
from starter.utils import get_device


def implicit_to_mesh(voxels, min_value, max_value, device):
    # Marching cubes extracts the F(x,y,z) = 0 surface
    vertices, faces = mcubes.marching_cubes(
        voxels.detach().cpu().numpy(),
        isovalue=0,
    )

    vertices = torch.tensor(vertices, dtype=torch.float32)
    faces = torch.tensor(faces.astype(np.int64), dtype=torch.int64)

    voxel_size = voxels.shape[0]

    vertices = (
        vertices / (voxel_size - 1)
    ) * (max_value - min_value) + min_value

    # Simple per-vertex coloring
    textures = (vertices - vertices.min(dim=0).values) / (
        vertices.max(dim=0).values
        - vertices.min(dim=0).values
        + 1e-8
    )

    mesh = pytorch3d.structures.Meshes(
        verts=[vertices],
        faces=[faces],
        textures=pytorch3d.renderer.TexturesVertex(
            verts_features=[textures]
        ),
    ).to(device)

    return mesh


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--voxel_size", type=int, default=64)
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--num_frames", type=int, default=36)
    parser.add_argument("--fps", type=int, default=15)

    args = parser.parse_args()

    device = get_device()

    # A

    major_R = 1.0
    minor_r = 0.4

    min_value = -1.6
    max_value = 1.6

    axis = torch.linspace(
        min_value,
        max_value,
        args.voxel_size,
    )

    X, Y, Z = torch.meshgrid(
        axis,
        axis,
        axis,
        indexing="ij",
    )

    # F(x,y,z) = 0 defines the torus surface
    torus_voxels = (
        (
            X**2
            + Y**2
            + Z**2
            + major_R**2
            - minor_r**2
        ) ** 2
        - 4 * major_R**2 * (X**2 + Y**2)
    )

    torus_mesh = implicit_to_mesh(
        torus_voxels,
        min_value,
        max_value,
        device,
    )

    torus_frames = render_mesh_360(
        torus_mesh,
        image_size=args.image_size,
        num_frames=args.num_frames,
        dist=4.0,
        elev=25.0,
        device=device,
        light_location=[[2, 2, -3]],
    )

    save_gif(
        torus_frames,
        "output/implicit_torus.gif",
        fps=args.fps,
    )

    # B

    a = 1.2
    b = 0.8
    c = 0.6

    min_value = -1.5
    max_value = 1.5

    axis = torch.linspace(
        min_value,
        max_value,
        args.voxel_size,
    )

    X, Y, Z = torch.meshgrid(
        axis,
        axis,
        axis,
        indexing="ij",
    )

    # F(x,y,z) = 0 defines the ellipsoid surface
    ellipsoid_voxels = (
        X**2 / a**2
        + Y**2 / b**2
        + Z**2 / c**2
        - 1
    )

    ellipsoid_mesh = implicit_to_mesh(
        ellipsoid_voxels,
        min_value,
        max_value,
        device,
    )

    ellipsoid_frames = render_mesh_360(
        ellipsoid_mesh,
        image_size=args.image_size,
        num_frames=args.num_frames,
        dist=4.0,
        elev=25.0,
        device=device,
        light_location=[[2, 2, -3]],
    )

    save_gif(
        ellipsoid_frames,
        "output/implicit_ellipsoid.gif",
        fps=args.fps,
    )