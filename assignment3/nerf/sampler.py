import math
from typing import List

import torch
from ray_utils import RayBundle
from pytorch3d.renderer.cameras import CamerasBase


# Sampler which implements stratified (uniform) point sampling along rays
class StratifiedRaysampler(torch.nn.Module):
    def __init__(
        self,
        cfg
    ):
        super().__init__()

        self.n_pts_per_ray = cfg.n_pts_per_ray
        self.min_depth = cfg.min_depth
        self.max_depth = cfg.max_depth

    def forward(
        self,
        ray_bundle,
    ):
        # TODO (Q1.4): Compute z values for self.n_pts_per_ray points uniformly sampled between [near, far]
        z_vals = torch.linspace(
            self.min_depth,
            self.max_depth,
            self.n_pts_per_ray,
            device=ray_bundle.origins.device,
        )

        z_vals = z_vals.view(1, -1, 1)

        # TODO (Q1.4): Sample points from z values
        sample_points = (
            ray_bundle.origins[:, None, :]
            + z_vals * ray_bundle.directions[:, None, :]
        )

        # Return
        return ray_bundle._replace(
            sample_points=sample_points,
            sample_lengths=z_vals * torch.ones_like(sample_points[..., :1]),
        )


def sample_pdf(bins, weights, n_samples, deterministic=False, eps=1e-5):
    """
    Q4.2: inverse-CDF sampling of a piecewise-constant PDF along each ray.

    bins:    (C, Nb+1) bin edges (sorted along dim 1)
    weights: (C, Nb)   non-negative weight of each bin
    returns: (C, n_samples) depths in [bins[:, 0], bins[:, -1]]
    """
    weights = weights + eps  # avoid all-zero rays / division by zero
    pdf = weights / torch.sum(weights, dim=-1, keepdim=True)
    cdf = torch.cumsum(pdf, dim=-1)
    cdf = torch.cat([torch.zeros_like(cdf[..., :1]), cdf], dim=-1)  # (C, Nb+1)

    if deterministic:
        u = torch.linspace(0.0, 1.0, n_samples, device=bins.device, dtype=bins.dtype)
        u = u.expand(*cdf.shape[:-1], n_samples)
    else:
        u = torch.rand(*cdf.shape[:-1], n_samples, device=bins.device, dtype=bins.dtype)
    u = u.contiguous()

    # Index of the first CDF edge strictly greater than u -> bin is [inds-1, inds]
    inds = torch.searchsorted(cdf.contiguous(), u, right=True)
    below = torch.clamp(inds - 1, min=0)
    above = torch.clamp(inds, max=cdf.shape[-1] - 1)

    cdf_below = torch.gather(cdf, -1, below)
    cdf_above = torch.gather(cdf, -1, above)
    bins_below = torch.gather(bins, -1, below)
    bins_above = torch.gather(bins, -1, above)

    denom = cdf_above - cdf_below
    denom = torch.where(denom < eps, torch.ones_like(denom), denom)
    t = (u - cdf_below) / denom

    return bins_below + t * (bins_above - bins_below)


# Q4.2: Sampler which draws extra points along rays according to coarse rendering weights
class ImportanceRaysampler(torch.nn.Module):
    def __init__(
        self,
        cfg
    ):
        super().__init__()

        self.n_pts_per_ray = cfg.n_pts_per_ray_fine

    def forward(
        self,
        ray_bundle,
        weights,
    ):
        # ray_bundle holds the coarse samples: sample_lengths (C, Nc, 1); weights (C, Nc, 1).
        # Coarse depths t_i are sample *locations*. Sample i is taken to represent the
        # interval between its neighbouring midpoints, so the Nc-1 midpoints are bin edges
        # for the Nc-2 interior samples. The two endpoint weights are dropped: the first
        # cell is only half-sized and the last one absorbs all remaining transmittance
        # because its delta is 1e10.
        z_coarse = ray_bundle.sample_lengths[..., 0]
        z_mid = 0.5 * (z_coarse[..., 1:] + z_coarse[..., :-1])  # (C, Nc-1)

        # Weights only shape the sampling distribution; no gradient flows through them.
        z_fine = sample_pdf(
            z_mid,
            weights[..., 0][..., 1:-1].detach(),
            self.n_pts_per_ray,
            deterministic=not self.training,
        ).detach()  # (C, Nf)

        z_vals, _ = torch.sort(torch.cat([z_coarse, z_fine], dim=-1), dim=-1)
        z_vals = z_vals[..., None]  # (C, Nc+Nf, 1)

        sample_points = (
            ray_bundle.origins[:, None, :]
            + z_vals * ray_bundle.directions[:, None, :]
        )

        # New bundle (RayBundle._replace mutates in place; keep the coarse bundle intact)
        return RayBundle(
            ray_bundle.origins,
            ray_bundle.directions,
            sample_points,
            z_vals,
        )


sampler_dict = {
    'stratified': StratifiedRaysampler,
    'importance': ImportanceRaysampler,
}