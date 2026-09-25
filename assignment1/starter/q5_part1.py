import argparse
import pickle

import imageio
import numpy as np
import pytorch3d
import torch

from starter.render_generic import load_rgbd_data
from starter.utils import get_device, get_points_renderer, unproject_depth_image


def save_gif(frames, path, fps=15):
    images = [(np.clip(f, 0, 1) * 255).astype(np.uint8) for f in frames]
    imageio.mimsave(path, images, duration=1000 // fps, loop=0)
    print(f"wrote {path} ({len(images)} frames)")


def render_pointcloud_360(
    point_cloud,
    image_size=256,
    num_frames=36,
    dist=6.0,
    elev=10.0,
    device=None,
):
    if device is None:
        device = get_device()

    renderer = get_points_renderer(
        image_size=image_size,
        device=device,
        background_color=(1, 1, 1),
    )

    azims = torch.linspace(0, 360, num_frames + 1)[:-1]
    frames = []

    for azim in azims:
        R, T = pytorch3d.renderer.look_at_view_transform(
            dist=dist,
            elev=elev,
            azim=azim,
            up=((0, -1, 0),),
        )
        cameras = pytorch3d.renderer.FoVPerspectiveCameras(
            R=R,
            T=T,
            device=device,
        )
        rend = renderer(point_cloud, cameras=cameras)
        frames.append(rend[0, ..., :3].detach().cpu().numpy())

    return frames


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--num_frames", type=int, default=36)
    parser.add_argument("--fps", type=int, default=15)
    parser.add_argument("--point_stride", type=int, default=1)
    parser.add_argument("--output_dir", type=str, default="output")
    args = parser.parse_args()

    device = get_device()
    data = load_rgbd_data()

    # ---------- Point Cloud 1 ----------
    image1 = torch.tensor(data["rgb1"], dtype=torch.float32)
    mask1 = torch.tensor(data["mask1"], dtype=torch.float32)
    depth1 = torch.tensor(data["depth1"], dtype=torch.float32)
    camera1 = data["cameras1"].to(device)

    points1, rgba1 = unproject_depth_image(
        image=image1,
        mask=mask1,
        depth=depth1,
        camera=camera1,
    )

    points1 = points1[::args.point_stride]
    rgba1 = rgba1[::args.point_stride]

    point_cloud1 = pytorch3d.structures.Pointclouds(
        points=[points1],
        features=[rgba1],
    ).to(device)

    frames1 = render_pointcloud_360(
        point_cloud1,
        image_size=args.image_size,
        num_frames=args.num_frames,
        dist=6.0,
        elev=10.0,
        device=device,
    )

    save_gif(
        frames1,
        f"{args.output_dir}/plant_view1.gif",
        fps=args.fps,
    )

    # ---------- Point Cloud 2 ----------
    image2 = torch.tensor(data["rgb2"], dtype=torch.float32)
    mask2 = torch.tensor(data["mask2"], dtype=torch.float32)
    depth2 = torch.tensor(data["depth2"], dtype=torch.float32)
    camera2 = data["cameras2"].to(device)

    points2, rgba2 = unproject_depth_image(
        image=image2,
        mask=mask2,
        depth=depth2,
        camera=camera2,
    )

    points2 = points2[::args.point_stride]
    rgba2 = rgba2[::args.point_stride]

    point_cloud2 = pytorch3d.structures.Pointclouds(
        points=[points2],
        features=[rgba2],
    ).to(device)

    frames2 = render_pointcloud_360(
        point_cloud2,
        image_size=args.image_size,
        num_frames=args.num_frames,
        dist=6.0,
        elev=10.0,
        device=device,
    )

    save_gif(
        frames2,
        f"{args.output_dir}/plant_view2.gif",
        fps=args.fps,
    )

    # ---------- Merged Point Cloud ----------
    points12 = torch.cat([points1, points2], dim=0)
    rgba12 = torch.cat([rgba1, rgba2], dim=0)

    point_cloud12 = pytorch3d.structures.Pointclouds(
        points=[points12],
        features=[rgba12],
    ).to(device)

    frames12 = render_pointcloud_360(
        point_cloud12,
        image_size=args.image_size,
        num_frames=args.num_frames,
        dist=6.0,
        elev=10.0,
        device=device,
    )

    save_gif(
        frames12,
        f"{args.output_dir}/plant_merged.gif",
        fps=args.fps,
    )