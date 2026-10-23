import torch

from typing import List, Optional, Tuple
from pytorch3d.renderer.cameras import CamerasBase


# Volume renderer which integrates color and density along rays
# according to the equations defined in [Mildenhall et al. 2020]
class VolumeRenderer(torch.nn.Module):
    def __init__(
        self,
        cfg
    ):
        super().__init__()

        self._chunk_size = cfg.chunk_size
        self._white_background = cfg.white_background if 'white_background' in cfg else False

    def _compute_weights(
        self,
        deltas,
        rays_density: torch.Tensor,
        eps: float = 1e-10
    ):
        # TODO (1.5): Compute transmittance using the equation described in the README
        alpha = 1.0 - torch.exp(-rays_density * deltas)
        transmittance = torch.cumprod(torch.cat([torch.ones_like(alpha[:, :1]), 1.0 - alpha + eps], dim=1), dim=1)[:, :-1]
        
        # TODO (1.5): Compute weight used for rendering from transmittance and alpha
        weights = transmittance * alpha
        return weights
    
    def _aggregate(
        self,
        weights: torch.Tensor,
        rays_feature: torch.Tensor
    ):
        # TODO (1.5): Aggregate (weighted sum of) features using weights
        return torch.sum(weights * rays_feature, dim=1)

    def forward(
        self,
        sampler,
        implicit_fn,
        ray_bundle,
    ):
        B = ray_bundle.shape[0]

        # Process the chunks of rays.
        chunk_outputs = []

        for chunk_start in range(0, B, self._chunk_size):
            cur_ray_bundle = ray_bundle[chunk_start:chunk_start+self._chunk_size]

            # Sample points along the ray
            cur_ray_bundle = sampler(cur_ray_bundle)
            n_pts = cur_ray_bundle.sample_shape[1]

            # Call implicit function with sample points
            implicit_output = implicit_fn(cur_ray_bundle)
            density = implicit_output['density']
            feature = implicit_output['feature']

            # Compute length of each ray segment
            depth_values = cur_ray_bundle.sample_lengths[..., 0]
            deltas = torch.cat(
                (
                    depth_values[..., 1:] - depth_values[..., :-1],
                    1e10 * torch.ones_like(depth_values[..., :1]),
                ),
                dim=-1,
            )[..., None]

            # Compute aggregation weights
            weights = self._compute_weights(
                deltas.view(-1, n_pts, 1),
                density.view(-1, n_pts, 1)
            ) 

            # TODO (1.5): Render (color) features using weights
            # TODO (1.5): Render depth map
            rays_feature = implicit_output["feature"].view(-1, n_pts, implicit_output["feature"].shape[-1])
            feature = self._aggregate(weights, rays_feature)
            depth = self._aggregate(weights, cur_ray_bundle.sample_lengths)

            # Return
            cur_out = {
                'feature': feature,
                'depth': depth,
            }

            chunk_outputs.append(cur_out)

        # Concatenate chunk outputs
        out = {
            k: torch.cat(
              [chunk_out[k] for chunk_out in chunk_outputs],
              dim=0
            ) for k in chunk_outputs[0].keys()
        }

        return out


# Q4.2: two-pass (coarse -> fine) volume renderer
class HierarchicalVolumeRenderer(VolumeRenderer):
    def _render_pass(
        self,
        implicit_fn,
        cur_ray_bundle,
    ):
        n_pts = cur_ray_bundle.sample_shape[1]
        implicit_output = implicit_fn(cur_ray_bundle)

        depth_values = cur_ray_bundle.sample_lengths[..., 0]
        deltas = torch.cat(
            (
                depth_values[..., 1:] - depth_values[..., :-1],
                1e10 * torch.ones_like(depth_values[..., :1]),
            ),
            dim=-1,
        )[..., None]

        weights = self._compute_weights(
            deltas.view(-1, n_pts, 1),
            implicit_output['density'].view(-1, n_pts, 1)
        )

        rays_feature = implicit_output['feature'].view(-1, n_pts, implicit_output['feature'].shape[-1])
        feature = self._aggregate(weights, rays_feature)
        depth = self._aggregate(weights, cur_ray_bundle.sample_lengths)

        return weights, feature, depth

    def forward(
        self,
        sampler,
        implicit_fn,
        ray_bundle,
        fine_sampler=None,
        fine_implicit_fn=None,
    ):
        B = ray_bundle.shape[0]

        # Process the chunks of rays.
        chunk_outputs = []

        for chunk_start in range(0, B, self._chunk_size):
            cur_ray_bundle = ray_bundle[chunk_start:chunk_start+self._chunk_size]

            # Coarse pass: uniform samples -> coarse network
            coarse_bundle = sampler(cur_ray_bundle)
            coarse_weights, coarse_feature, coarse_depth = self._render_pass(
                implicit_fn, coarse_bundle
            )

            # Fine pass: coarse + importance samples -> fine network
            fine_bundle = fine_sampler(coarse_bundle, coarse_weights)
            _, fine_feature, fine_depth = self._render_pass(
                fine_implicit_fn, fine_bundle
            )

            # 'feature' / 'depth' are the fine (final) outputs
            cur_out = {
                'feature': fine_feature,
                'depth': fine_depth,
                'coarse_feature': coarse_feature,
                'coarse_depth': coarse_depth,
            }

            chunk_outputs.append(cur_out)

        # Concatenate chunk outputs
        out = {
            k: torch.cat(
              [chunk_out[k] for chunk_out in chunk_outputs],
              dim=0
            ) for k in chunk_outputs[0].keys()
        }

        return out


def sdf_to_density(signed_distance, alpha, beta):
    # TODO (Q4.3): Convert signed distance to density with alpha, beta parameters
    # VolSDF: sigma = alpha * Psi_beta(-d), Psi_beta = CDF of a zero-mean Laplace with scale beta
    #   d >= 0 (outside): alpha * 0.5 * exp(-d / beta)
    #   d <  0 (inside):  alpha * (1 - 0.5 * exp(d / beta))
    # exp(-|d| / beta) <= 1 in both branches, so neither branch can overflow.
    e = torch.exp(-torch.abs(signed_distance) / beta)
    return alpha * torch.where(signed_distance >= 0, 0.5 * e, 1.0 - 0.5 * e)

class VolumeSDFRenderer(VolumeRenderer):
    def __init__(
        self,
        cfg
    ):
        super().__init__(cfg)

        self._chunk_size = cfg.chunk_size
        self._white_background = cfg.white_background if 'white_background' in cfg else False
        self.alpha = cfg.alpha
        self.beta = cfg.beta

        self.cfg = cfg

    def forward(
        self,
        sampler,
        implicit_fn,
        ray_bundle,
        light_dir=None
    ):
        B = ray_bundle.shape[0]

        # Process the chunks of rays.
        chunk_outputs = []

        for chunk_start in range(0, B, self._chunk_size):
            cur_ray_bundle = ray_bundle[chunk_start:chunk_start+self._chunk_size]

            # Sample points along the ray
            cur_ray_bundle = sampler(cur_ray_bundle)
            n_pts = cur_ray_bundle.sample_shape[1]

            # Call implicit function with sample points
            distance, color = implicit_fn.get_distance_color(cur_ray_bundle.sample_points)
            density = sdf_to_density(distance, self.alpha, self.beta) # TODO (Q4.3): convert SDF to density

            # Compute length of each ray segment
            depth_values = cur_ray_bundle.sample_lengths[..., 0]
            deltas = torch.cat(
                (
                    depth_values[..., 1:] - depth_values[..., :-1],
                    1e10 * torch.ones_like(depth_values[..., :1]),
                ),
                dim=-1,
            )[..., None]

            # Compute aggregation weights
            weights = self._compute_weights(
                deltas.view(-1, n_pts, 1),
                density.view(-1, n_pts, 1)
            ) 

            geometry_color = torch.zeros_like(color)

            # Compute color
            color = self._aggregate(
                weights,
                color.view(-1, n_pts, color.shape[-1])
            )
            depth = self._aggregate(weights, cur_ray_bundle.sample_lengths)

            # Return
            cur_out = {
                'feature': color,
                'depth': depth,
                "geometry": geometry_color
            }

            chunk_outputs.append(cur_out)

        # Concatenate chunk outputs
        out = {
            k: torch.cat(
              [chunk_out[k] for chunk_out in chunk_outputs],
              dim=0
            ) for k in chunk_outputs[0].keys()
        }

        return out


renderer_dict = {
    'volume': VolumeRenderer,
    'volume_sdf': VolumeSDFRenderer,
    'volume_hierarchical': HierarchicalVolumeRenderer,
}
