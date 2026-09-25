import argparse

import imageio
import numpy as np
import pytorch3d
import torch

from starter.utils import get_device, get_points_renderer


def save_gif(frames, path, fps=15):
    images = [(np.clip(frame, 0, 1) * 255).astype(np.uint8) for frame in frames]
    imageio.mimsave(path, images, duration=1000 // fps, loop=0)
    print(f"wrote {path} ({len(images)} frames)")


def render_pointcloud_360(
    point_cloud,
    image_size=256,
    num_frames=36,
    dist=8.0,
    elev=15.0,
    radius=0.01,
    device=None,
):
    if device is None:
        device = get_device()

    renderer = get_points_renderer(
        image_size=image_size,
        device=device,
        radius=radius,
        background_color=(1, 1, 1),
    )

    frames = []
    azims = torch.linspace(0, 360, num_frames + 1)[:-1]

    for azim in azims:
        R, T = pytorch3d.renderer.look_at_view_transform(
            dist=dist,
            elev=elev,
            azim=azim,
        )
        cameras = pytorch3d.renderer.FoVPerspectiveCameras(
            R=R,
            T=T,
            device=device,
        )

        rend = renderer(point_cloud, cameras=cameras)
        frame = rend[0, ..., :3].detach().cpu().numpy()
        frames.append(frame)

    return frames


def make_sphere(center, radius, num_u=60, num_v=60, color=(1.0, 1.0, 0.0)):
    """
    Returns:
        points: (N, 3)
        colors: (N, 3)
    """
    u = torch.linspace(0, 2 * np.pi, num_u)
    v = torch.linspace(0, np.pi, num_v)
    U, V = torch.meshgrid(u, v, indexing="ij")

    x = radius * torch.sin(V) * torch.cos(U)
    y = radius * torch.sin(V) * torch.sin(U)
    z = radius * torch.cos(V)

    points = torch.stack([x.flatten(), y.flatten(), z.flatten()], dim=1)
    points = points + torch.tensor(center, dtype=torch.float32)

    colors = torch.ones_like(points) * torch.tensor(color, dtype=torch.float32)
    return points, colors


def make_tube_arc(
    center=(0.0, -1.2, 1.7),
    arc_radius=0.9,
    tube_radius=0.12,
    angle_start_deg=25,
    angle_end_deg=155,
    num_u=120,
    num_v=24,
    color=(1.0, 0.0, 0.0),
):
    """
    Make a smile-shaped tube arc in 3D.

    center: center of the underlying smile arc circle
    arc_radius: radius of the smile arc
    tube_radius: thickness of the mouth tube
    """
    u = torch.linspace(
        np.deg2rad(angle_start_deg),
        np.deg2rad(angle_end_deg),
        num_u,
    )
    v = torch.linspace(0, 2 * np.pi, num_v)

    U, V = torch.meshgrid(u, v, indexing="ij")

    # Centerline of the smile arc, lying roughly in the x-y plane
    cx = center[0] + arc_radius * torch.cos(U)
    cy = center[1] + arc_radius * torch.sin(U)
    cz = torch.ones_like(U) * center[2]

    # Normal in the x-y plane
    nx = torch.cos(U)
    ny = torch.sin(U)
    nz = torch.zeros_like(U)

    # Binormal along z
    bx = torch.zeros_like(U)
    by = torch.zeros_like(U)
    bz = torch.ones_like(U)

    x = cx + tube_radius * (torch.cos(V) * nx + torch.sin(V) * bx)
    y = cy + tube_radius * (torch.cos(V) * ny + torch.sin(V) * by)
    z = cz + tube_radius * (torch.cos(V) * nz + torch.sin(V) * bz)

    points = torch.stack([x.flatten(), y.flatten(), z.flatten()], dim=1)
    colors = torch.ones_like(points) * torch.tensor(color, dtype=torch.float32)

    return points, colors


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--num_frames", type=int, default=36)
    parser.add_argument("--fps", type=int, default=15)
    parser.add_argument("--output_path", type=str, default="output/q6_smiley.gif")
    args = parser.parse_args()

    device = get_device()

    # Head
    head_points, head_colors = make_sphere(
        center=(0.0, 0.0, 0.0),
        radius=2.0,
        num_u=80,
        num_v=80,
        color=(1.0, 0.9, 0.1),
    )

    # Left eye
    left_eye_points, left_eye_colors = make_sphere(
        center=(-0.7, 0.7, 1.65),
        radius=0.22,
        num_u=40,
        num_v=40,
        color=(0.0, 0.0, 0.0),
    )

    # Right eye
    right_eye_points, right_eye_colors = make_sphere(
        center=(0.7, 0.7, 1.65),
        radius=0.22,
        num_u=40,
        num_v=40,
        color=(0.0, 0.0, 0.0),
    )

    # Mouth
    mouth_points, mouth_colors = make_tube_arc(
        center=(0.0, -0.15, 1.88),
        arc_radius=0.95,
        tube_radius=0.10,
        angle_start_deg=205,
        angle_end_deg=335,
        num_u=140,
        num_v=24,
        color=(1.0, 0.2, 0.2),
    )

    # Concatenate everything into one scene point cloud
    all_points = torch.cat(
        [head_points, left_eye_points, right_eye_points, mouth_points],
        dim=0,
    )

    all_colors = torch.cat(
        [head_colors, left_eye_colors, right_eye_colors, mouth_colors],
        dim=0,
    )

    smiley_pc = pytorch3d.structures.Pointclouds(
        points=[all_points],
        features=[all_colors],
    ).to(device)
    frames = render_pointcloud_360(
        smiley_pc,
        image_size=args.image_size,
        num_frames=args.num_frames,
        dist=8.0,
        elev=15.0,
        radius=0.01,
        device=device,
    )

    save_gif(
        frames,
        args.output_path,
        fps=args.fps,
    )