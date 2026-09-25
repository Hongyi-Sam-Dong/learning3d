import argparse

import imageio
import numpy as np
import pytorch3d
import torch

from starter.utils import (
    get_device,
    get_mesh_renderer,
    get_points_renderer,
    load_cow_mesh,
)


def sample_points_from_mesh(vertices, faces, num_samples):
    """
    Uniformly sample points from the surface of a triangle mesh.

    Args:
        vertices: (Nv, 3)
        faces: (Nf, 3)
        num_samples: number of points to sample

    Returns:
        points: (num_samples, 3)
    """

    v0 = vertices[faces[:, 0]]
    v1 = vertices[faces[:, 1]]
    v2 = vertices[faces[:, 2]]

    edge1 = v1 - v0
    edge2 = v2 - v0

    cross = torch.cross(edge1, edge2, dim=1)
    face_areas = 0.5 * torch.linalg.norm(cross, dim=1)

    # Convert areas into probabilities
    face_probs = face_areas / face_areas.sum()

    sampled_face_indices = torch.multinomial(
        face_probs,
        num_samples=num_samples,
        replacement=True,
    )

    A = v0[sampled_face_indices]
    B = v1[sampled_face_indices]
    C = v2[sampled_face_indices]

    u = torch.rand(num_samples, 1, device=vertices.device)
    v = torch.rand(num_samples, 1, device=vertices.device)

    sqrt_u = torch.sqrt(u)

    alpha = 1 - sqrt_u
    beta = sqrt_u * (1 - v)
    gamma = sqrt_u * v

    points = alpha * A + beta * B + gamma * C

    return points


def make_point_cloud(points, color=(0.1, 0.4, 1.0)):
    colors = torch.ones_like(points) * torch.tensor(
        color,
        dtype=torch.float32,
        device=points.device,
    )

    return pytorch3d.structures.Pointclouds(
        points=[points],
        features=[colors],
    )


def add_label(image, text):
    """
    Very simple label using matplotlib-style image padding.
    """
    from PIL import Image, ImageDraw

    image = (np.clip(image, 0, 1) * 255).astype(np.uint8)

    pil_image = Image.fromarray(image)
    draw = ImageDraw.Draw(pil_image)

    draw.rectangle(
        [0, 0, pil_image.width, 24],
        fill="white",
    )

    draw.text(
        (8, 5),
        text,
        fill="black",
    )

    return np.asarray(pil_image)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--cow_path",
        type=str,
        default="data/cow.obj",
    )

    parser.add_argument(
        "--output_path",
        type=str,
        default="output/q7_sampling.gif",
    )

    parser.add_argument(
        "--image_size",
        type=int,
        default=256,
    )

    parser.add_argument(
        "--num_frames",
        type=int,
        default=36,
    )

    parser.add_argument(
        "--fps",
        type=int,
        default=15,
    )

    args = parser.parse_args()

    device = get_device()

    vertices, faces = load_cow_mesh(args.cow_path)

    vertices = vertices.to(device)
    faces = faces.to(device)

    cow_color = torch.ones_like(vertices) * torch.tensor(
        [0.7, 0.7, 1.0],
        device=device,
    )

    cow_mesh = pytorch3d.structures.Meshes(
        verts=[vertices],
        faces=[faces],
        textures=pytorch3d.renderer.TexturesVertex(
            verts_features=[cow_color]
        ),
    )
    sample_counts = [
        10,
        100,
        1000,
        10000,
    ]

    point_clouds = []

    for n in sample_counts:
        points = sample_points_from_mesh(
            vertices,
            faces,
            n,
        )

        pc = make_point_cloud(points)
        point_clouds.append(pc)
    mesh_renderer = get_mesh_renderer(
        image_size=args.image_size,
        device=device,
    )

    point_renderer = get_points_renderer(
        image_size=args.image_size,
        device=device,
        radius=0.01,
        background_color=(1, 1, 1),
    )

    lights = pytorch3d.renderer.PointLights(
        location=[[2, 2, -3]],
        device=device,
    )

    frames = []

    azims = torch.linspace(
        0,
        360,
        args.num_frames + 1,
    )[:-1]

    for azim in azims:

        R, T = pytorch3d.renderer.look_at_view_transform(
            dist=3.0,
            elev=20.0,
            azim=azim,
        )

        cameras = pytorch3d.renderer.FoVPerspectiveCameras(
            R=R,
            T=T,
            device=device,
        )

        cow_rend = mesh_renderer(
            cow_mesh,
            cameras=cameras,
            lights=lights,
        )

        cow_image = (
            cow_rend[0, ..., :3]
            .detach()
            .cpu()
            .numpy()
        )

        cow_image = add_label(
            cow_image,
            "Original Mesh",
        )

        panels = [cow_image]

        for n, point_cloud in zip(
            sample_counts,
            point_clouds,
        ):
            rend = point_renderer(
                point_cloud,
                cameras=cameras,
            )

            image = (
                rend[0, ..., :3]
                .detach()
                .cpu()
                .numpy()
            )

            image = add_label(
                image,
                f"{n} Points",
            )

            panels.append(image)

        # Combine all five images horizontally
        gallery = np.concatenate(
            panels,
            axis=1,
        )

        frames.append(gallery)

    imageio.mimsave(
        args.output_path,
        frames,
        duration=1000 // args.fps,
        loop=0,
    )

    print(
        f"wrote {args.output_path} "
        f"({len(frames)} frames)"
    )