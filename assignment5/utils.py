import os
import json
import math
import torch
import numpy as np
import pytorch3d
from pytorch3d.renderer import (
    AlphaCompositor,
    PointsRasterizationSettings,
    PointsRenderer,
    PointsRasterizer,
)
import imageio
from PIL import Image, ImageDraw, ImageFont

from models import cls_model, seg_model

SEG_COLORS = [[1.0,1.0,1.0], [1.0,0.0,1.0], [0.0,1.0,1.0],[1.0,1.0,0.0],[0.0,0.0,1.0], [1.0,0.0,0.0]]
SEG_COLOR_NAMES = ["white", "magenta", "cyan", "yellow", "blue", "red"]
SEG_BACKGROUND = (0.5, 0.5, 0.5)
CLS_COLOR = [0.16, 0.47, 0.84]


def checkpoint_config(args):
    return {k: (str(v) if isinstance(v, torch.device) else v) for k, v in vars(args).items()}


def save_checkpoint(epoch, model, args, best=False, name=None, extra=None):
    if name is None:
        name = 'best_model' if best else 'model_epoch_{}'.format(epoch)
    path = os.path.join(args.checkpoint_dir, name + '.pt')
    num_classes = args.num_cls_class if args.task == 'cls' else args.num_seg_class
    torch.save(dict(task=args.task, epoch=epoch, num_classes=num_classes, config=checkpoint_config(args),
                    model_state_dict=model.state_dict(), **(extra or {})), path)
    return path


def read_checkpoint(path, device):
    if not os.path.isfile(path):
        raise FileNotFoundError("checkpoint not found: {} (train first, e.g. python main.py train --task cls|seg)".format(path))
    ckpt = torch.load(path, map_location=device)
    if 'model_state_dict' in ckpt:
        return ckpt['model_state_dict'], {k: v for k, v in ckpt.items() if k != 'model_state_dict'}
    return ckpt, {}


def load_weights(model, path, device):
    state_dict, meta = read_checkpoint(path, device)
    model.load_state_dict(state_dict)
    return meta


def load_model(path, task, device):
    state_dict, meta = read_checkpoint(path, device)
    if meta.get('task', task) != task:
        raise ValueError("{} is a '{}' checkpoint, expected '{}'".format(path, meta['task'], task))
    num_classes = meta.get('num_classes', 3 if task == 'cls' else 6)
    model = cls_model(num_classes=num_classes) if task == 'cls' else seg_model(num_seg_classes=num_classes)
    model.load_state_dict(state_dict)
    return model.to(device).eval(), meta


def eval_settings(args, checkpoint, meta):
    return {
        "checkpoint": checkpoint,
        "checkpoint_epoch": meta.get("epoch"),
        "checkpoint_val_acc": meta.get("val_acc"),
        "checkpoint_config": meta.get("config"),
        "data_dir": args.main_dir,
        "num_points": args.num_points,
        "seed": args.seed,
        "batch_size": args.batch_size,
        "device": str(args.device),
        "point_sampling": "per-object random permutation of all available points from numpy default_rng(seed); "
                          "the first num_points indices are used, so smaller subsets are nested in larger ones",
    }


def write_json(path, obj):
    with open(path, 'w') as f:
        json.dump(obj, f, indent=2, default=lambda o: o.tolist())


def point_orders(num_objects, num_available, num_points, seed):
    if not 0 < num_points <= num_available:
        raise ValueError("--num_points must be in [1, {}], got {}".format(num_available, num_points))
    return np.argsort(np.random.default_rng(seed).random((num_objects, num_available)), axis=1)


def gather_points(array, idx):
    return np.take_along_axis(array, idx.reshape(idx.shape + (1,) * (array.ndim - 2)), axis=1)


def rotation_matrix(axis, degrees):
    t = math.radians(degrees)
    c, s = math.cos(t), math.sin(t)
    matrices = {
        'x': [[1, 0, 0], [0, c, -s], [0, s, c]],
        'y': [[c, 0, s], [0, 1, 0], [-s, 0, c]],
        'z': [[c, -s, 0], [s, c, 0], [0, 0, 1]],
    }
    return torch.tensor(matrices[axis], dtype=torch.float32)


def predict(model, data, device, batch_size, idx=None, rotation=None):
    model.eval()
    outputs = []
    with torch.no_grad():
        for start in range(0, len(data), batch_size):
            points = data[start:start + batch_size]
            if idx is not None:
                points = gather_points(points, idx[start:start + batch_size])
            points = torch.from_numpy(np.ascontiguousarray(points)).float().to(device)
            if rotation is not None:
                points = points @ rotation.to(device).T
            outputs.append(model(points).float().cpu())
    return torch.cat(outputs)


def save_checkpoint_dir_guard(path, overwrite):
    if os.path.exists(path) and not overwrite:
        raise SystemExit("{} already exists; pass --overwrite or choose another --checkpoint_dir".format(path))


def create_dir(directory):
    """
    Creates a directory if it does not already exist.
    """
    if not os.path.exists(directory):
        os.makedirs(directory)

def get_points_renderer(
    image_size=256, device=None, radius=0.01, background_color=(1, 1, 1)
):
    """
    Returns a Pytorch3D renderer for point clouds.

    Args:
        image_size (int): The rendered image size.
        device (torch.device): The torch device to use (CPU or GPU). If not specified,
            will automatically use GPU if available, otherwise CPU.
        radius (float): The radius of the rendered point in NDC.
        background_color (tuple): The background color of the rendered image.

    Returns:
        PointsRenderer.
    """
    if device is None:
        if torch.cuda.is_available():
            device = torch.device("cuda:0")
        else:
            device = torch.device("cpu")
    raster_settings = PointsRasterizationSettings(image_size=image_size, radius=radius,)
    renderer = PointsRenderer(
        rasterizer=PointsRasterizer(raster_settings=raster_settings),
        compositor=AlphaCompositor(background_color=background_color),
    )
    return renderer


def seg_colors(labels):
    return torch.tensor(SEG_COLORS)[labels.long().cpu()]


def cls_colors(verts):
    return torch.tensor(CLS_COLOR).repeat(verts.shape[0], 1)


def viz_points(verts, colors, path, device, background_color=(1, 1, 1)):
    image_size=256

    # Construct various camera viewpoints
    dist = 3
    elev = 0
    azim = [180 - 12*i for i in range(30)]
    R, T = pytorch3d.renderer.cameras.look_at_view_transform(dist=dist, elev=elev, azim=azim, device=device)
    c = pytorch3d.renderer.FoVPerspectiveCameras(R=R, T=T, fov=60, device=device)

    sample_verts = verts.unsqueeze(0).repeat(30,1,1).to(torch.float)
    sample_colors = colors.unsqueeze(0).repeat(30,1,1).to(torch.float)

    point_cloud = pytorch3d.structures.Pointclouds(points=sample_verts, features=sample_colors).to(device)

    renderer = get_points_renderer(image_size=image_size, background_color=background_color, device=device)
    rend = renderer(point_cloud, cameras=c).cpu().numpy() # (30, 256, 256, 3)
    rend = (rend.clip(0, 1) * 255).astype(np.uint8)

    imageio.mimsave(path, rend, fps=15)


def viz_seg (verts, labels, path, device, background_color=(1, 1, 1)):
    """
    visualize segmentation result
    output: a 360-degree gif
    """
    # Colorize points based on segmentation labels
    viz_points(verts, seg_colors(labels), path, device, background_color)


def viz_cls(verts, path, device):
    viz_points(verts, cls_colors(verts), path, device)


def render_view(verts, colors, device, background_color=(1, 1, 1), image_size=240, elev=10, azim=30):
    R, T = pytorch3d.renderer.cameras.look_at_view_transform(dist=3, elev=elev, azim=azim, device=device)
    c = pytorch3d.renderer.FoVPerspectiveCameras(R=R, T=T, fov=60, device=device)
    point_cloud = pytorch3d.structures.Pointclouds(points=[verts.float().to(device)], features=[colors.float().to(device)])
    rend = get_points_renderer(image_size=image_size, background_color=background_color, device=device)(point_cloud, cameras=c)
    return (rend[0, ..., :3].clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)


def save_montage(rows, captions, path, caption_height=20):
    font = ImageFont.load_default(size=13)
    h, w = rows[0][0].shape[:2]
    canvas = Image.new('RGB', (w * max(len(r) for r in rows), (h + caption_height) * len(rows)), 'white')
    draw = ImageDraw.Draw(canvas)
    for r, (images, texts) in enumerate(zip(rows, captions)):
        top = r * (h + caption_height)
        for c, (image, text) in enumerate(zip(images, texts)):
            canvas.paste(Image.fromarray(image), (c * w, top + caption_height))
            draw.text((c * w + 4, top + 3), text, fill=(11, 11, 11), font=font)
    canvas.save(path, optimize=True)
