import torch
import torch.nn.functional as F

from pytorch3d.ops import knn_points
from pytorch3d.loss import mesh_laplacian_smoothing


# define losses
def voxel_loss(voxel_src, voxel_tgt):
    # voxel_src: b x h x w x d
    # voxel_tgt: b x h x w x d
    # implement some loss for binary voxel grids

    loss = F.binary_cross_entropy_with_logits(
        voxel_src,
        voxel_tgt
    )

    return loss


def chamfer_loss(point_cloud_src, point_cloud_tgt):
    # point_cloud_src, point_cloud_tgt: b x n_points x 3
    # implement chamfer loss from scratch

    src_to_tgt = knn_points(
        point_cloud_src,
        point_cloud_tgt,
        K=1
    )

    tgt_to_src = knn_points(
        point_cloud_tgt,
        point_cloud_src,
        K=1
    )

    loss_src_to_tgt = src_to_tgt.dists.mean()
    loss_tgt_to_src = tgt_to_src.dists.mean()

    loss_chamfer = loss_src_to_tgt + loss_tgt_to_src

    return loss_chamfer


def smoothness_loss(mesh_src):
    # implement laplacian smoothening loss

    loss_laplacian = mesh_laplacian_smoothing(
        mesh_src,
        method="uniform"
    )

    return loss_laplacian