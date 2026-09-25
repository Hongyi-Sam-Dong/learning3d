"""
Shared helpers for rendering 360-degree turntable gifs.

Used by most parts of the assignment, since nearly every submission asks for a
turntable view of some geometry.
"""
import imageio
import numpy as np
import torch
from pytorch3d.renderer import FoVPerspectiveCameras, PointLights, look_at_view_transform

from starter.utils import get_device, get_mesh_renderer


def save_gif(frames, path, fps=15):
    """
    Saves a list of (H, W, 3) float images in [0, 1] as an animated gif.

    The renderer returns floats, but gifs need uint8, so the conversion happens
    here rather than at every call site.
    """
    images = [(np.clip(f, 0, 1) * 255).astype(np.uint8) for f in frames]
    imageio.mimsave(path, images, duration=1000 // fps, loop=0)
    print(f"wrote {path} ({len(images)} frames)")


def turntable_views(num_frames=36, dist=3.0, elev=0.0):
    """
    Returns a list of (R, T) camera extrinsics orbiting the origin.

    The final azimuth is dropped because linspace includes both 0 and 360,
    which would render the same view twice and make the loop stutter.
    """
    azims = torch.linspace(0, 360, num_frames + 1)[:-1]
    return [look_at_view_transform(dist=dist, elev=elev, azim=azim) for azim in azims]


def render_mesh_360(
    mesh,
    image_size=256,
    num_frames=36,
    dist=3.0,
    elev=0.0,
    device=None,
    light_location=[[0, 0, -3]],
):
    """
    Renders a mesh from num_frames viewpoints orbiting the origin.

    Returns a list of (H, W, 3) float images in [0, 1].
    """
    if device is None:
        device = get_device()
    mesh = mesh.to(device)
    renderer = get_mesh_renderer(image_size=image_size, device=device)
    lights = PointLights(location=light_location, device=device)

    frames = []
    for R, T in turntable_views(num_frames, dist, elev):
        cameras = FoVPerspectiveCameras(R=R, T=T, device=device)
        rend = renderer(mesh, cameras=cameras, lights=lights)
        frames.append(rend[0, ..., :3].detach().cpu().numpy())
    return frames
