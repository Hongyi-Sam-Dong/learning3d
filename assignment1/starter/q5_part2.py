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
    dist=4.0,
    elev=20.0,
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
        R_cam, T_cam = pytorch3d.renderer.look_at_view_transform(
            dist=dist,
            elev=elev,
            azim=azim,
        )
        cameras = pytorch3d.renderer.FoVPerspectiveCameras(
            R=R_cam,
            T=T_cam,
            device=device,
        )
        rend = renderer(point_cloud, cameras=cameras)
        frames.append(rend[0, ..., :3].detach().cpu().numpy())

    return frames


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--num_frames", type=int, default=36)
    parser.add_argument("--num_samples", type=int, default=100)
    parser.add_argument("--fps", type=int, default=15)
    parser.add_argument("--output_dir", type=str, default="output")
    args = parser.parse_args()

    device = get_device()

    # A
    R = 1.0   # major radius
    r = 0.4   # minor radius

    theta = torch.linspace(0, 2 * np.pi, args.num_samples)
    phi = torch.linspace(0, 2 * np.pi, args.num_samples)
    Theta, Phi = torch.meshgrid(theta, phi, indexing="ij")

    x = (R + r * torch.cos(Theta)) * torch.cos(Phi)
    y = r * torch.sin(Theta)
    z = (R + r * torch.cos(Theta)) * torch.sin(Phi)

    torus_points = torch.stack(
        [x.flatten(), y.flatten(), z.flatten()],
        dim=1,
    )
    torus_color = (torus_points - torus_points.min()) / (
        torus_points.max() - torus_points.min()
    )

    torus_pc = pytorch3d.structures.Pointclouds(
        points=[torus_points],
        features=[torus_color],
    ).to(device)

    torus_frames = render_pointcloud_360(
        torus_pc,
        image_size=args.image_size,
        num_frames=args.num_frames,
        dist=4.0,
        elev=20.0,
        radius=0.01,
        device=device,
    )

    save_gif(
        torus_frames,
        f"{args.output_dir}/torus.gif",
        fps=args.fps,
    )

    # B
    u = torch.linspace(0, 2 * np.pi, args.num_samples)
    v = torch.linspace(-0.4, 0.4, args.num_samples)
    U, V = torch.meshgrid(u, v, indexing="ij")

    mobius_x = (1 + (V / 2) * torch.cos(U / 2)) * torch.cos(U)
    mobius_y = (1 + (V / 2) * torch.cos(U / 2)) * torch.sin(U)
    mobius_z = (V / 2) * torch.sin(U / 2)

    mobius_points = torch.stack(
        [mobius_x.flatten(), mobius_y.flatten(), mobius_z.flatten()],
        dim=1,
    )
    mobius_color = (mobius_points - mobius_points.min()) / (
        mobius_points.max() - mobius_points.min()
    )

    mobius_pc = pytorch3d.structures.Pointclouds(
        points=[mobius_points],
        features=[mobius_color],
    ).to(device)

    mobius_frames = render_pointcloud_360(
        mobius_pc,
        image_size=args.image_size,
        num_frames=args.num_frames,
        dist=4.0,
        elev=20.0,
        radius=0.01,
        device=device,
    )

    save_gif(
        mobius_frames,
        f"{args.output_dir}/mobius.gif",
        fps=args.fps,
    )