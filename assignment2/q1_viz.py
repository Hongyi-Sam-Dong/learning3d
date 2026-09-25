"""
Q1 visualization: fit a voxel grid / point cloud / mesh to the first training
sample (via fit_data.train_model) and render the optimized result next to the
ground truth as 360-degree turntable GIFs plus a static side-by-side PNG.

Usage:
    python q1_viz.py --type vox   --device cpu
    python q1_viz.py --type point --device cpu
    python q1_viz.py --type mesh  --device cpu
"""
import argparse
import os

import imageio
import mcubes
import numpy as np
import torch
from pytorch3d.renderer import (
    AlphaCompositor,
    FoVPerspectiveCameras,
    HardPhongShader,
    MeshRasterizer,
    MeshRenderer,
    PointLights,
    PointsRasterizationSettings,
    PointsRasterizer,
    PointsRenderer,
    RasterizationSettings,
    TexturesVertex,
    look_at_view_transform,
)
from pytorch3d.structures import Meshes, Pointclouds

import fit_data

SRC_COLOR = (0.3, 0.5, 0.9)  # optimized: blue
TGT_COLOR = (0.9, 0.6, 0.3)  # ground truth: orange


def get_args_parser():
    parser = argparse.ArgumentParser('Q1 Visualization', parents=[fit_data.get_args_parser()])
    parser.add_argument('--output_dir', default='outputs/q1', type=str)
    parser.add_argument('--image_size', default=256, type=int)
    parser.add_argument('--num_views', default=36, type=int)
    parser.add_argument('--dist', default=1.4, type=float)
    parser.add_argument('--elev', default=20.0, type=float)
    parser.add_argument('--azim', default=45.0, type=float, help='azimuth of the static PNG view')
    parser.add_argument('--vox_threshold', default=0.5, type=float,
                        help='occupancy threshold on sigmoid(logits) for the optimized voxels')
    parser.add_argument('--point_radius', default=0.01, type=float)
    return parser


def voxels_to_mesh(voxels, threshold, device):
    """Marching cubes on a (1, D, D, D) occupancy grid indexed (z, y, x), in [-0.5, 0.5]^3."""
    grid = voxels.detach().squeeze(0).cpu().numpy()
    D = grid.shape[0]
    # pad so surfaces touching the grid border are closed
    grid = np.pad(grid, 1, mode='constant', constant_values=0)
    verts, faces = mcubes.marching_cubes(grid, threshold)
    if len(faces) == 0:
        return None
    verts = verts - 1  # undo padding
    verts = verts[:, ::-1] / D - 0.5  # (z, y, x) index -> (x, y, z) world, matches utils_vox.Ref2Mem
    verts = torch.tensor(verts.copy(), dtype=torch.float32, device=device)
    faces = torch.tensor(faces.astype(np.int64), device=device)
    return Meshes(verts=[verts], faces=[faces])


def colored_mesh(mesh, color):
    verts = mesh.verts_packed().detach()
    faces = mesh.faces_packed()
    textures = TexturesVertex(verts_features=(torch.ones_like(verts) * torch.tensor(color, device=verts.device))[None])
    return Meshes(verts=[verts], faces=[faces], textures=textures)


def render_views(obj, azims, args):
    """Render a Meshes or Pointclouds object from each azimuth; returns list of HxWx3 uint8 frames."""
    device = args.device
    if isinstance(obj, Meshes):
        renderer = MeshRenderer(
            rasterizer=MeshRasterizer(raster_settings=RasterizationSettings(
                image_size=args.image_size, blur_radius=0.0, faces_per_pixel=1)),
            shader=HardPhongShader(device=device),
        )
    else:
        renderer = PointsRenderer(
            rasterizer=PointsRasterizer(raster_settings=PointsRasterizationSettings(
                image_size=args.image_size, radius=args.point_radius)),
            compositor=AlphaCompositor(background_color=(1, 1, 1)),
        )

    frames = []
    for azim in azims:
        R, T = look_at_view_transform(dist=args.dist, elev=args.elev, azim=azim)
        cameras = FoVPerspectiveCameras(R=R, T=T, device=device)
        if obj is None:  # e.g. empty voxel grid after thresholding
            frames.append(np.full((args.image_size, args.image_size, 3), 255, dtype=np.uint8))
            continue
        if isinstance(obj, Meshes):
            lights = PointLights(location=cameras.get_camera_center(), device=device)
            image = renderer(obj, cameras=cameras, lights=lights)
        else:
            image = renderer(obj, cameras=cameras)
        image = image[0, ..., :3].clamp(0, 1).cpu().numpy()
        frames.append((image * 255).astype(np.uint8))
    return frames


def to_renderable(src, tgt, args):
    """Convert fit_data.train_model outputs into renderable (optimized, ground truth) objects."""
    device = args.device
    if args.type == 'vox':
        src_mesh = voxels_to_mesh(torch.sigmoid(src), args.vox_threshold, device)
        tgt_mesh = voxels_to_mesh(tgt, 0.5, device)
        return (colored_mesh(src_mesh, SRC_COLOR) if src_mesh is not None else None,
                colored_mesh(tgt_mesh, TGT_COLOR))
    if args.type == 'point':
        def pc(points, color):
            points = points.detach()[0]
            return Pointclouds(points=[points], features=[torch.ones_like(points) * torch.tensor(color, device=device)])
        return pc(src, SRC_COLOR), pc(tgt, TGT_COLOR)
    return colored_mesh(src, SRC_COLOR), colored_mesh(tgt, TGT_COLOR)


def main(args):
    os.makedirs(args.output_dir, exist_ok=True)

    src, tgt = fit_data.train_model(args)
    src_obj, tgt_obj = to_renderable(src, tgt, args)

    azims = np.linspace(-180, 180, args.num_views, endpoint=False)
    src_frames = render_views(src_obj, azims, args)
    tgt_frames = render_views(tgt_obj, azims, args)
    side_by_side = [np.concatenate([s, t], axis=1) for s, t in zip(src_frames, tgt_frames)]

    prefix = os.path.join(args.output_dir, args.type)
    duration_ms = 1000 * 3.6 / args.num_views  # ~3.6 s per full turn
    imageio.mimsave(f'{prefix}_optimized.gif', src_frames, duration=duration_ms, loop=0)
    imageio.mimsave(f'{prefix}_gt.gif', tgt_frames, duration=duration_ms, loop=0)
    imageio.mimsave(f'{prefix}_comparison.gif', side_by_side, duration=duration_ms, loop=0)

    static = np.concatenate(
        [render_views(src_obj, [args.azim], args)[0], render_views(tgt_obj, [args.azim], args)[0]], axis=1)
    imageio.imwrite(f'{prefix}_comparison.png', static)

    print(f'Saved {prefix}_{{optimized,gt,comparison}}.gif and {prefix}_comparison.png '
          '(left: optimized, right: ground truth)')


if __name__ == '__main__':
    args = get_args_parser().parse_args()
    main(args)
