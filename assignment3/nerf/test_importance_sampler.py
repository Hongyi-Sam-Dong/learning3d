# Q4.2: synthetic check of the importance sampler.
# Run from the nerf/ directory: python test_importance_sampler.py
import torch
from omegaconf import OmegaConf

from ray_utils import RayBundle
from sampler import ImportanceRaysampler, sample_pdf


def test_sample_pdf_concentrates_in_heavy_bin():
    torch.manual_seed(0)
    C, Nb, Ns = 256, 10, 64
    bins = torch.linspace(1.0, 7.0, Nb + 1).expand(C, Nb + 1)
    weights = torch.full((C, Nb), 0.01)
    weights[:, 5] = 1.0  # dominant middle interval [bins[5], bins[6]]

    z = sample_pdf(bins, weights, Ns)
    in_bin = ((z >= bins[:, 5:6]) & (z <= bins[:, 6:7])).float().mean()
    # Expected fraction: 1.0 / (1.0 + 9 * 0.01) ~= 0.917
    assert in_bin > 0.85, in_bin
    assert (z >= 1.0).all() and (z <= 7.0).all()


def test_importance_sampler_merges_and_sorts():
    torch.manual_seed(0)
    C, Nc, Nf = 8, 64, 64
    cfg = OmegaConf.create({'n_pts_per_ray_fine': Nf})
    sampler = ImportanceRaysampler(cfg)

    z = torch.linspace(1.0, 7.0, Nc).view(1, Nc, 1).expand(C, Nc, 1)
    origins, directions = torch.zeros(C, 3), torch.tensor([[0.0, 0.0, 1.0]]).expand(C, 3)
    bundle = RayBundle(origins, directions, origins[:, None] + z * directions[:, None], z)

    weights = torch.zeros(C, Nc, 1)
    weights[:, 30] = 1.0
    out = sampler(bundle, weights)

    assert out.sample_lengths.shape == (C, Nc + Nf, 1)
    assert out.sample_points.shape == (C, Nc + Nf, 3)
    assert (out.sample_lengths[:, 1:] >= out.sample_lengths[:, :-1]).all()
    # Coarse bundle untouched
    assert bundle.sample_lengths.shape == (C, Nc, 1)
    # Fine samples land in sample 30's cell (between its neighbouring midpoints)
    lo, hi = 0.5 * (z[0, 29] + z[0, 30]), 0.5 * (z[0, 30] + z[0, 31])
    new = out.sample_lengths[..., 0]
    assert ((new >= lo - 1e-5) & (new <= hi + 1e-5)).sum(-1).min() >= Nf


if __name__ == '__main__':
    test_sample_pdf_concentrates_in_heavy_bin()
    test_importance_sampler_merges_and_sorts()
    print('importance sampler tests passed')
