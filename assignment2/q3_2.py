"""Q3.2: parametric network  g(image_feature, u, v) -> (x, y, z).

Encoder is the Part 2 one (ImageNet ResNet18 -> 512-d, or the precomputed feats with --load_feat).
The decoder is an MLP on [feature (512) | uv (2)] that maps each 2D parameter sample to one 3D
point. (u, v) live in an abstract parameter square [-1, 1]^2, i.e. they index the predicted surface
itself; they are NOT pixel coordinates of the input image. The network has no n_points-sized
layer, so any number of uv samples can be decoded with the same weights.

Supervision: N points sampled from the GT mesh vs the N decoded points, compared with Chamfer loss
(losses.chamfer_loss). There is no correspondence between the i-th uv sample and the i-th GT sample,
so a pointwise MSE would be meaningless.

Limitation: one square chart is topologically a sheet. A single continuous map from it has to be
folded/stretched to cover chairs with holes (backrest slats) or disconnected parts (legs), so such
regions come out with stray "bridging" points between parts.

F1 and rendering go through the Part 2 point-cloud path (q2_viz.evaluate_sample -> eval_model.evaluate
with --type point: no re-centering), so Q3.2 F1 is directly comparable with Part 2 point F1.

    python q3_2.py --mode train --device cuda
    python q3_2.py --mode eval --device cuda --load_checkpoint
"""
import argparse
import math
import os
import random
import time

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from pytorch3d.ops import sample_points_from_meshes
from torchvision import models as torchvision_models
from torchvision import transforms

import losses
import q1_viz
import q2_viz
import q3_1

THRESHOLDS = q2_viz.THRESHOLDS  # [0.01, 0.02, 0.03, 0.04, 0.05]
KEY_T = q2_viz.KEY_T  # 0.05


def get_args_parser():
    parser = argparse.ArgumentParser('Q3.2 parametric network')
    parser.add_argument('--mode', default='train', choices=['train', 'eval'])
    parser.add_argument('--device', default='cuda', type=str)
    parser.add_argument('--arch', default='resnet18', type=str)
    parser.add_argument('--lr', default=4e-4, type=float)
    parser.add_argument('--max_iter', default=100000, type=int)
    parser.add_argument('--batch_size', default=None, type=int,
                        help='default: 32 for train (train_model.py), 1 for eval (eval_model.py)')
    parser.add_argument('--num_workers', default=4, type=int)
    parser.add_argument('--save_freq', default=2000, type=int)
    parser.add_argument('--load_checkpoint', action='store_true')
    parser.add_argument('--load_feat', action='store_true')
    parser.add_argument('--n_points', default=1000, type=int,
                        help='uv samples decoded per object = GT mesh points sampled per object')
    parser.add_argument('--uv_sampling', default='random', choices=['random', 'grid'],
                        help='uv sampling during training')
    parser.add_argument('--eval_uv_sampling', default='grid', choices=['random', 'grid'],
                        help='uv sampling during eval/visualization (grid = deterministic)')
    parser.add_argument('--checkpoint', default='checkpoint_q3_2.pth', type=str)
    parser.add_argument('--output_dir', default='outputs/q3_2', type=str)
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--max_samples', default=None, type=int,
                        help='use only the first N samples of the split (smoke tests / overfitting)')
    # eval / visualization (defaults as in q2_viz.py)
    parser.add_argument('--num_examples', default=3, type=int)
    parser.add_argument('--vis_idxs', default=None, type=int, nargs='*',
                        help='test-set indices to visualize (default: the first --num_examples)')
    parser.add_argument('--image_size', default=256, type=int)
    parser.add_argument('--dist', default=1.4, type=float)
    parser.add_argument('--elev', default=20.0, type=float)
    parser.add_argument('--azim', default=45.0, type=float)
    parser.add_argument('--point_radius', default=0.01, type=float)
    return parser


# ---------------------------------------------------------------------------- uv domain

def uv_grid(n, device='cpu'):
    """Exactly n deterministic, near-regular uv samples in [-1, 1]^2 (abstract chart, not pixels).

    Builds the smallest side x side lattice with side^2 >= n and keeps n of its nodes at evenly spaced
    flat indices, so the side^2 - n dropped nodes are spread out instead of cut off as one strip.
    """
    side = math.ceil(math.sqrt(n))
    lin = torch.linspace(-1, 1, side, device=device)
    vv, uu = torch.meshgrid(lin, lin, indexing='ij')
    lattice = torch.stack([uu, vv], dim=-1).reshape(-1, 2)  # side^2 x 2
    keep = torch.linspace(0, side * side - 1, n, device=device).round().long()
    return lattice[keep]


def sample_uv(batch_size, n, mode, device='cpu'):
    """B x n x 2 abstract surface parameters: uniform in [-1, 1]^2 per object, or the shared grid."""
    if mode == 'random':
        return torch.rand(batch_size, n, 2, device=device) * 2 - 1
    return uv_grid(n, device)[None].expand(batch_size, -1, -1)


# ---------------------------------------------------------------------------- model

class ParametricSingleView3D(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.load_feat = args.load_feat
        if not args.load_feat:  # same encoder as model.SingleViewto3D / q3_1
            vision_model = torchvision_models.__dict__[args.arch](pretrained=True)
            self.encoder = nn.Sequential(*(list(vision_model.children())[:-1]))
            self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        # [feature (512) | uv (2)] -> xyz, one point per uv sample (independent of n_points)
        self.decoder = nn.Sequential(
            nn.Linear(512 + 2, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 3),
        )

    def encode(self, images):
        """B x H x W x 3 RGB in [0, 1] (or B x 512 precomputed feats) -> B x 512."""
        if self.load_feat:
            return images
        images_normalize = self.normalize(images.permute(0, 3, 1, 2))
        return self.encoder(images_normalize).flatten(1)

    def decode(self, feats, uv):
        """feats: B x 512, uv: B x N x 2 -> points: B x N x 3."""
        feats = feats[:, None, :].expand(-1, uv.shape[1], -1)  # B x N x 512
        return self.decoder(torch.cat([feats, uv], dim=-1))  # B x N x 514 -> B x N x 3

    def forward(self, images, uv):
        return self.decode(self.encode(images), uv)


# ---------------------------------------------------------------------------- checkpoint

def load_checkpoint(model, optimizer, args):
    """q3_1.load_checkpoint (load_feat check, map_location) with a clear error for foreign checkpoints."""
    try:
        return q3_1.load_checkpoint(model, optimizer, args)
    except (RuntimeError, KeyError, ValueError) as e:
        raise SystemExit(f'{args.checkpoint} is not a compatible Q3.2 checkpoint:\n{e}')


# ---------------------------------------------------------------------------- train

def train(args):
    loader = q3_1.make_loader('train', args, shuffle=True)
    model = ParametricSingleView3D(args).to(args.device)
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    start_iter = 0
    if args.load_checkpoint:
        start_iter = load_checkpoint(model, optimizer, args) + 1  # 'step' = last completed step

    print(f'Starting training ({len(loader.dataset)} train objects, {args.n_points} points / object, '
          f'uv sampling: {args.uv_sampling})')
    start_time = time.time()
    train_loader = iter(loader)
    step = start_iter - 1
    for step in range(start_iter, args.max_iter):
        iter_start_time = time.time()
        try:
            feed_dict = next(train_loader)
        except StopIteration:  # restart after one epoch
            train_loader = iter(loader)
            feed_dict = next(train_loader)
        images = q3_1.model_input(feed_dict, args)
        gt_points = sample_points_from_meshes(feed_dict['mesh'].to(args.device), args.n_points)  # B x N x 3
        read_time = time.time() - iter_start_time

        uv = sample_uv(images.shape[0], args.n_points, args.uv_sampling, args.device)  # B x N x 2
        pred_points = model(images, uv)  # B x N x 3
        loss = losses.chamfer_loss(pred_points, gt_points)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % args.save_freq == 0 and step > 0:
            q3_1.save_checkpoint(model, optimizer, step, args)
        print('[%4d/%4d]; ttime: %.0f (%.2f, %.2f); loss: %.5f'
              % (step, args.max_iter, time.time() - start_time, read_time,
                 time.time() - iter_start_time, loss.item()))

    if step >= start_iter:
        q3_1.save_checkpoint(model, optimizer, step, args)
    print('Done!')


# ---------------------------------------------------------------------------- eval

def draw_row(axes, row):
    for ax, img, label in zip(axes, [row['rgb'], row['pred'], row['gt']],
                              ['Input RGB', 'Parametric prediction', 'Ground truth mesh']):
        ax.imshow(img)
        ax.set_title(label, fontsize=10)
        ax.axis('off')


def save_examples(rows, args):
    for i, row in enumerate(rows):
        fig, axes = plt.subplots(1, 3, figsize=(9, 3.4))
        draw_row(axes, row)
        fig.suptitle(q2_viz.panel_title(i + 1, row['f1']) + f'  (test idx {row["idx"]})')
        fig.tight_layout()
        fig.savefig(os.path.join(args.output_dir, f'example_{i}.png'), dpi=120, bbox_inches='tight')
        plt.close(fig)
    if not rows:
        return
    fig, axes = plt.subplots(len(rows), 3, figsize=(9, 3.2 * len(rows)), squeeze=False)
    for i, row in enumerate(rows):
        draw_row(axes[i], row)
        axes[i][0].text(-0.08, 0.5, q2_viz.panel_title(i + 1, row['f1']), transform=axes[i][0].transAxes,
                        rotation=90, va='center', ha='right', fontsize=10)
    fig.suptitle('Q3.2 parametric: Input RGB | Prediction | GT mesh')
    fig.tight_layout()
    fig.savefig(os.path.join(args.output_dir, 'examples.png'), dpi=120, bbox_inches='tight')
    plt.close(fig)


def save_f1_plot(avg_f1, args):
    fig, ax = plt.subplots()
    ax.plot(THRESHOLDS, avg_f1.numpy(), marker='o')
    ax.set_xlabel('Threshold')
    ax.set_ylabel('F1-score')
    ax.set_title('Evaluation parametric (Q3.2)')
    ax.grid(alpha=0.3)
    fig.savefig(os.path.join(args.output_dir, 'f1.png'), bbox_inches='tight')
    plt.close(fig)


def write_metrics(avg_p, avg_r, avg_f1, n_eval, step, args):
    k = THRESHOLDS.index(KEY_T)
    lines = [
        f'checkpoint: {os.path.abspath(args.checkpoint)} (step {step})',
        f'load_feat: {args.load_feat}',
        f'n_points (predicted = GT samples): {args.n_points}',
        f'eval uv sampling: {args.eval_uv_sampling}',
        f'samples evaluated: {n_eval}' + (' (partial, --max_samples)' if args.max_samples else ' (full test split)'),
        f'seed: {args.seed}',
        '', f'{"threshold":>9} {"precision":>10} {"recall":>10} {"F1":>10}',
    ]
    lines += [f'{t:>9.2f} {avg_p[i]:>10.3f} {avg_r[i]:>10.3f} {avg_f1[i]:>10.3f}' for i, t in enumerate(THRESHOLDS)]
    lines += ['', f'*** F1@{KEY_T}: {avg_f1[k]:.3f} ***']
    text = '\n'.join(lines) + '\n'
    with open(os.path.join(args.output_dir, 'metrics.txt'), 'w') as f:
        f.write(text)
    print('\n' + text)


def evaluate(args):
    if not args.load_checkpoint:
        raise SystemExit('Refusing to evaluate an untrained model: pass --load_checkpoint.')
    os.makedirs(args.output_dir, exist_ok=True)
    model = ParametricSingleView3D(args).to(args.device)
    step = load_checkpoint(model, None, args)
    model.eval()
    args.type = 'point'  # q2_viz / eval_model helpers: use the Part 2 point-cloud F1 + rendering path

    loader = q3_1.make_loader('test', args, shuffle=False)
    vis_idxs = args.vis_idxs if args.vis_idxs is not None else list(range(args.num_examples))
    vis_rows, all_p, all_r, all_f1 = {}, [], [], []
    n_total = len(loader) * args.batch_size

    with torch.no_grad():
        for step_i, feed_dict in enumerate(loader):
            images = q3_1.model_input(feed_dict, args)
            mesh_gt = feed_dict['mesh']
            uv = sample_uv(images.shape[0], args.n_points, args.eval_uv_sampling, args.device)
            pred_points = model(images, uv)  # B x N x 3
            for b in range(pred_points.shape[0]):
                idx = step_i * args.batch_size + b
                # eval_model.evaluate samples n_points GT points and runs compute_sampling_metrics
                metrics, _ = q2_viz.evaluate_sample(pred_points[b:b + 1], mesh_gt[b], args)
                all_p.append(torch.tensor([metrics['Precision@%f' % t].item() for t in THRESHOLDS]))
                all_r.append(torch.tensor([metrics['Recall@%f' % t].item() for t in THRESHOLDS]))
                all_f1.append(torch.tensor([metrics['F1@%f' % t].item() for t in THRESHOLDS]))
                if idx in vis_idxs:
                    vis_rows[idx] = {
                        'idx': idx,
                        'f1': all_f1[-1][THRESHOLDS.index(KEY_T)].item(),
                        'rgb': feed_dict['images'][b].cpu().numpy().clip(0, 1),
                        'pred': q2_viz.render(q2_viz.pred_renderable(pred_points[b:b + 1], args), args),
                        'gt': q2_viz.render(q1_viz.colored_mesh(
                            q2_viz.gt_mesh_for_render(mesh_gt[b], args), q2_viz.GT_COLOR), args),
                    }
            n_done = len(all_f1)
            if n_done % 50 == 0 or n_done == n_total:
                f1_05 = torch.stack(all_f1)[:, THRESHOLDS.index(KEY_T)].mean()
                print(f'[{n_done:4d}/{n_total:4d}] running avg F1@{KEY_T}: {f1_05:.3f}')

    avg_p, avg_r, avg_f1 = (torch.stack(x).mean(0) for x in (all_p, all_r, all_f1))
    save_examples([vis_rows[i] for i in vis_idxs if i in vis_rows], args)
    save_f1_plot(avg_f1, args)
    write_metrics(avg_p, avg_r, avg_f1, len(all_f1), step, args)
    print(f'Saved outputs to {args.output_dir}/')


def main(args):
    if args.batch_size is None:
        args.batch_size = 32 if args.mode == 'train' else 1
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    train(args) if args.mode == 'train' else evaluate(args)


if __name__ == '__main__':
    main(get_args_parser().parse_args())
