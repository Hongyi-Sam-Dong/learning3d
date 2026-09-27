"""Q3.1: implicit occupancy network  f(image_feature, x, y, z) -> occupancy logit.

Encoder is the Part 2 one (ImageNet ResNet18 -> 512-d, or the precomputed feats with --load_feat).
The decoder is an MLP on [feature (512) | xyz (3)] that scores each query point separately. No voxel
grid is decoded and then indexed.

Coordinate convention (matches the GT voxels built by utils_vox.get_occupancy):
    GT voxels:  B x 1 x 32 x 32 x 32, indexed [b, 0, z, y, x]  (mem x/y/z = voxel index along W/H/D)
    query grid: grid[z, y, x] = (x_n, y_n, z_n), where c_n = linspace(-1, 1, 32)[c_idx]
    flattening both in C order gives flat index  z * 32*32 + y * 32 + x  for voxels and queries alike.
The query feature order is (x, y, z), and the grid axes are (z, y, x) to match the voxel tensor.
For F1 and rendering, predicted grids go through the exact Part 2 vox path (q2_viz.evaluate_sample ->
eval_model.evaluate: marching cubes, utils_vox.Mem2Ref, -pi rotation about y, re-centering), so
Q3.1 F1 is directly comparable with Part 2 voxel F1.

    python q3_1.py --mode train --device cuda
    python q3_1.py --mode eval --device cuda --load_checkpoint
"""
import argparse
import os
import random
import time

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch3d.datasets.r2n2.utils import collate_batched_R2N2
from torchvision import models as torchvision_models
from torchvision import transforms

import dataset_location
import q1_viz
import q2_viz
from r2n2_custom import R2N2

VOX_RES = 32
THRESHOLDS = q2_viz.THRESHOLDS  # [0.01, 0.02, 0.03, 0.04, 0.05]
KEY_T = q2_viz.KEY_T  # 0.05


def get_args_parser():
    parser = argparse.ArgumentParser('Q3.1 implicit occupancy network')
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
    parser.add_argument('--train_query_points', default=4096, type=int,
                        help='query points sampled per object per iteration (32768 = full grid)')
    parser.add_argument('--query_chunk_size', default=8192, type=int,
                        help='query points per forward pass during full-grid inference')
    parser.add_argument('--checkpoint', default='checkpoint_q3_1.pth', type=str)
    parser.add_argument('--output_dir', default='outputs/q3_1', type=str)
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--max_samples', default=None, type=int,
                        help='use only the first N samples of the split (smoke tests / overfitting)')
    parser.add_argument('--vox_threshold', default=0.5, type=float,
                        help='occupancy threshold on sigmoid(logits)')
    # eval / visualization (defaults as in eval_model.py / q2_viz.py)
    parser.add_argument('--n_points', default=1000, type=int, help='surface samples for F1')
    parser.add_argument('--num_examples', default=3, type=int)
    parser.add_argument('--vis_idxs', default=None, type=int, nargs='*',
                        help='test-set indices to visualize (default: the first --num_examples)')
    parser.add_argument('--image_size', default=256, type=int)
    parser.add_argument('--dist', default=1.4, type=float)
    parser.add_argument('--elev', default=20.0, type=float)
    parser.add_argument('--azim', default=45.0, type=float)
    parser.add_argument('--point_radius', default=0.01, type=float,
                        help='read by q1_viz.render_views (also on its empty-prediction path)')
    return parser


# ---------------------------------------------------------------------------- coordinates

def make_query_grid(res=VOX_RES, device='cpu'):
    """(res^3, 3) xyz in (-1, 1)^3, row z*res*res + y*res + x, same as voxels[..., z, y, x].flatten()."""
    lin = torch.linspace(-1, 1, res, device=device)
    zz, yy, xx = torch.meshgrid(lin, lin, lin, indexing='ij')  # each (res, res, res), indexed [z, y, x]
    return torch.stack([xx, yy, zz], dim=-1).reshape(-1, 3)


# ---------------------------------------------------------------------------- model

class ImplicitSingleView3D(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.load_feat = args.load_feat
        if not args.load_feat:  # same encoder as model.SingleViewto3D
            vision_model = torchvision_models.__dict__[args.arch](pretrained=True)
            self.encoder = nn.Sequential(*(list(vision_model.children())[:-1]))
            self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        # [feature (512) | xyz (3)] -> occupancy logit (no sigmoid)
        self.decoder = nn.Sequential(
            nn.Linear(512 + 3, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )
        self.register_buffer('grid', make_query_grid(), persistent=False)  # 32768 x 3

    def encode(self, images):
        """B x H x W x 3 RGB in [0, 1] (or B x 512 precomputed feats) -> B x 512."""
        if self.load_feat:
            return images
        images_normalize = self.normalize(images.permute(0, 3, 1, 2))
        return self.encoder(images_normalize).flatten(1)

    def query(self, feats, xyz):
        """feats: B x 512, xyz: B x N x 3 -> logits: B x N x 1."""
        feats = feats[:, None, :].expand(-1, xyz.shape[1], -1)
        return self.decoder(torch.cat([feats, xyz], dim=-1))

    def query_grid(self, feats, chunk_size=None):
        """Full 32^3 grid, queried in chunks along N -> logits B x 1 x 32 x 32 x 32."""
        B = feats.shape[0]
        chunk_size = chunk_size or self.grid.shape[0]
        logits = torch.cat([self.query(feats, chunk[None].expand(B, -1, -1))
                            for chunk in self.grid.split(chunk_size)], dim=1)  # B x 32768 x 1
        return logits.reshape(B, 1, VOX_RES, VOX_RES, VOX_RES)

    def forward(self, images, xyz=None, chunk_size=None):
        feats = self.encode(images)
        return self.query_grid(feats, chunk_size) if xyz is None else self.query(feats, xyz)


# ---------------------------------------------------------------------------- data

def make_loader(split, args, shuffle):
    dataset = R2N2(split, dataset_location.SHAPENET_PATH, dataset_location.R2N2_PATH,
                   dataset_location.SPLITS_PATH, return_voxels=True, return_feats=args.load_feat)
    if args.max_samples:
        dataset = torch.utils.data.Subset(dataset, range(min(args.max_samples, len(dataset))))
    return torch.utils.data.DataLoader(
        dataset, batch_size=args.batch_size, num_workers=args.num_workers,
        collate_fn=collate_batched_R2N2, pin_memory=True, drop_last=True, shuffle=shuffle)


def model_input(feed_dict, args):
    """Exactly the view the model sees: RGB B x H x W x 3, or that view's B x 512 feature."""
    if args.load_feat:
        return torch.stack(feed_dict['feats']).to(args.device)
    return feed_dict['images'].squeeze(1).to(args.device)


def sample_queries(grid, voxels, n):
    """Per-object random subset of grid points (without replacement) and their GT occupancy.

    grid: 32768 x 3, voxels: B x 1 x 32 x 32 x 32 -> xyz B x n x 3, occ B x n x 1.
    """
    B, N = voxels.shape[0], grid.shape[0]
    occ = voxels.reshape(B, N)  # same C-order flattening as grid
    if n >= N:
        return grid[None].expand(B, -1, -1), occ[..., None]
    idx = torch.rand(B, N, device=voxels.device).argsort(dim=1)[:, :n]  # B x n
    return grid[idx], occ.gather(1, idx)[..., None]


# ---------------------------------------------------------------------------- checkpoint

def save_checkpoint(model, optimizer, step, args):
    torch.save({'step': step,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'args': vars(args)}, args.checkpoint)
    print(f'Saved {args.checkpoint} (step {step})')


def load_checkpoint(model, optimizer, args):
    if not os.path.isfile(args.checkpoint):
        raise SystemExit(f'Checkpoint not found: {os.path.abspath(args.checkpoint)}')
    checkpoint = torch.load(args.checkpoint, map_location=args.device)
    ckpt_load_feat = checkpoint.get('args', {}).get('load_feat')
    if ckpt_load_feat is not None and ckpt_load_feat != args.load_feat:
        raise SystemExit(f'Checkpoint was trained with load_feat={ckpt_load_feat}; '
                         f'pass {"--load_feat" if ckpt_load_feat else "no --load_feat"}.')
    model.load_state_dict(checkpoint['model_state_dict'])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    print(f'Loaded {args.checkpoint} (step {checkpoint["step"]})')
    return checkpoint['step']


# ---------------------------------------------------------------------------- train

def train(args):
    loader = make_loader('train', args, shuffle=True)
    model = ImplicitSingleView3D(args).to(args.device)
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    start_iter = 0
    if args.load_checkpoint:
        start_iter = load_checkpoint(model, optimizer, args) + 1  # 'step' = last completed step

    print(f'Starting training ({len(loader.dataset)} train objects, '
          f'{min(args.train_query_points, VOX_RES ** 3)} query points / object / iter)')
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
        images = model_input(feed_dict, args)
        voxels = feed_dict['voxels'].float().to(args.device)  # B x 1 x 32 x 32 x 32
        read_time = time.time() - iter_start_time

        xyz, occ = sample_queries(model.grid, voxels, args.train_query_points)
        logits = model(images, xyz)  # B x n x 1
        loss = F.binary_cross_entropy_with_logits(logits, occ)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % args.save_freq == 0 and step > 0:
            save_checkpoint(model, optimizer, step, args)
        print('[%4d/%4d]; ttime: %.0f (%.2f, %.2f); loss: %.4f'
              % (step, args.max_iter, time.time() - start_time, read_time,
                 time.time() - iter_start_time, loss.item()))

    if step >= start_iter:
        save_checkpoint(model, optimizer, step, args)
    print('Done!')


# ---------------------------------------------------------------------------- eval

def draw_row(axes, row):
    for ax, img, label in zip(axes, [row['rgb'], row['pred'], row['gt']],
                              ['Input RGB', 'Implicit prediction', 'Ground truth mesh']):
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
    fig.suptitle('Q3.1 implicit: Input RGB | Prediction | GT mesh')
    fig.tight_layout()
    fig.savefig(os.path.join(args.output_dir, 'examples.png'), dpi=120, bbox_inches='tight')
    plt.close(fig)


def save_f1_plot(avg_f1, args):
    fig, ax = plt.subplots()
    ax.plot(THRESHOLDS, avg_f1.numpy(), marker='o')
    ax.set_xlabel('Threshold')
    ax.set_ylabel('F1-score')
    ax.set_title('Evaluation implicit (Q3.1)')
    ax.grid(alpha=0.3)
    fig.savefig(os.path.join(args.output_dir, 'f1.png'), bbox_inches='tight')
    plt.close(fig)


def write_metrics(avg_p, avg_r, avg_f1, n_eval, n_empty, step, args):
    k = THRESHOLDS.index(KEY_T)
    lines = [
        f'checkpoint: {os.path.abspath(args.checkpoint)} (step {step})',
        f'load_feat: {args.load_feat}',
        f'n_points: {args.n_points}',
        f'samples evaluated: {n_eval}' + (' (partial, --max_samples)' if args.max_samples else ' (full test split)'),
        f'seed: {args.seed}',
        f'vox_threshold (on sigmoid(logits)): {args.vox_threshold}',
        f'samples with empty predicted surface (scored 0): {n_empty}',
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
    model = ImplicitSingleView3D(args).to(args.device)
    step = load_checkpoint(model, None, args)
    model.eval()
    args.type = 'vox'  # q2_viz / eval_model helpers: use the Part 2 voxel F1 + rendering path

    loader = make_loader('test', args, shuffle=False)
    vis_idxs = args.vis_idxs if args.vis_idxs is not None else list(range(args.num_examples))
    vis_rows, all_p, all_r, all_f1 = {}, [], [], []
    n_empty = 0
    n_total = len(loader) * args.batch_size

    with torch.no_grad():
        for step_i, feed_dict in enumerate(loader):
            images = model_input(feed_dict, args)
            mesh_gt = feed_dict['mesh']
            logits = model(images, chunk_size=args.query_chunk_size)  # B x 1 x 32 x 32 x 32
            for b in range(logits.shape[0]):
                idx = step_i * args.batch_size + b
                # sigmoid, threshold, marching cubes, Mem2Ref, rotation, re-centering: all Part 2 code
                metrics, empty = q2_viz.evaluate_sample(logits[b:b + 1], mesh_gt[b], args)
                n_empty += int(empty)
                all_p.append(torch.tensor([metrics['Precision@%f' % t].item() for t in THRESHOLDS]))
                all_r.append(torch.tensor([metrics['Recall@%f' % t].item() for t in THRESHOLDS]))
                all_f1.append(torch.tensor([metrics['F1@%f' % t].item() for t in THRESHOLDS]))
                if idx in vis_idxs:
                    vis_rows[idx] = {
                        'idx': idx,
                        'f1': all_f1[-1][THRESHOLDS.index(KEY_T)].item(),
                        'rgb': feed_dict['images'][b].cpu().numpy().clip(0, 1),
                        'pred': q2_viz.render(q2_viz.pred_renderable(logits[b:b + 1], args), args),
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
    write_metrics(avg_p, avg_r, avg_f1, len(all_f1), n_empty, step, args)
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
