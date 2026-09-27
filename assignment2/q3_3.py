"""Q3.3: extended dataset (chair + car + plane) for single-view voxel reconstruction.

The model is the Part 2 voxel model, unchanged (model.SingleViewto3D with type='vox'):
    RGB B x 137 x 137 x 3 -> ImageNet ResNet18 (fc removed) -> B x 512
        (or the precomputed B x 512 ResNet18 feature of the same view with --load_feat)
    -> Linear(512, 256*4^3) -> B x 256 x 4 x 4 x 4 -> 3 x ConvTranspose3d -> B x 1 x 32 x 32 x 32 raw logits
Loss is losses.voxel_loss (BCE with logits) against the GT occupancy B x 1 x 32 x 32 x 32.
Only the training data changes: split_3c.json of the full dataset instead of the chair-only split.

Modes:
    train    train on all three classes          -> checkpoint_q3_3_vox.pth
    eval     overall + per-class F1 of that model, qualitative examples per class
    compare  chair-only Part 2 model (checkpoint_vox.pth) vs the three-class model on the SAME
             chair test samples: same object, same input view, same GT points, same thresholds

Evaluation follows the Part 2 voxel convention (eval_model.evaluate, as wrapped by
q2_viz.evaluate_sample): sigmoid(logits), marching cubes at --vox_threshold, utils_vox.Mem2Ref,
-pi rotation about y, re-centering of both point sets, eval_model.compute_sampling_metrics
(F1 on a 0-100 scale at 0.01..0.05). The only difference from eval_model.evaluate is that the
predicted and GT point samples are drawn under fixed per-sample seeds, so both models in compare
mode are scored against the identical GT point set.

    python q3_3.py --mode train --device cuda
    python q3_3.py --mode train --device cuda --load_checkpoint        # resume
    python q3_3.py --mode eval --device cuda --load_checkpoint
    python q3_3.py --mode compare --device cuda
"""
import argparse
import csv
import functools
import json
import math
import os
import random
import time

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import mcubes
import numpy as np
import torch
from pytorch3d.datasets.r2n2.utils import collate_batched_R2N2
from pytorch3d.ops import sample_points_from_meshes
from pytorch3d.structures import Meshes
from pytorch3d.transforms import Rotate, axis_angle_to_matrix

import eval_model
import losses
import q1_viz
import q2_viz
import utils_vox
from model import SingleViewto3D
from r2n2_custom import R2N2

HERE = os.path.dirname(os.path.abspath(__file__))
THRESHOLDS = q2_viz.THRESHOLDS  # [0.01, 0.02, 0.03, 0.04, 0.05]
KEY_T = q2_viz.KEY_T  # 0.05
K05 = THRESHOLDS.index(KEY_T)
CLASS_NAMES = {'03001627': 'chair', '02958343': 'car', '02691156': 'plane'}
CHAIR = '03001627'
# Checkpoints this script must never write to.
PROTECTED_CHECKPOINTS = {'checkpoint_vox.pth', 'checkpoint_point.pth', 'checkpoint_mesh.pth',
                         'checkpoint_q3_1.pth', 'checkpoint_q3_2.pth'}


def get_args_parser():
    parser = argparse.ArgumentParser('Q3.3 extended dataset (voxels)')
    parser.add_argument('--mode', default='train', choices=['train', 'eval', 'compare'])
    parser.add_argument('--device', default='cuda', type=str)
    parser.add_argument('--arch', default='resnet18', type=str)
    parser.add_argument('--lr', default=4e-4, type=float)
    parser.add_argument('--max_iter', default=10001, type=int,
                        help='default 10001 = steps 0..10000, the same number of updates as the '
                             'chair-only checkpoint_vox.pth (saved at step 10000)')
    parser.add_argument('--batch_size', default=None, type=int,
                        help='default: 32 for train (train_model.py), 1 for eval/compare (eval_model.py)')
    parser.add_argument('--num_workers', default=4, type=int)
    parser.add_argument('--save_freq', default=2000, type=int)
    parser.add_argument('--load_checkpoint', action='store_true')
    parser.add_argument('--load_feat', action='store_true')
    parser.add_argument('--checkpoint', default='checkpoint_q3_3_vox.pth', type=str,
                        help='three-class checkpoint (written by train, read by eval/compare)')
    parser.add_argument('--single_class_checkpoint', default='checkpoint_vox.pth', type=str,
                        help='chair-only Part 2 voxel checkpoint (compare mode, read only)')
    parser.add_argument('--dataset_root', default=os.path.join(HERE, 'r2n2_shapenet_dataset_full'), type=str)
    parser.add_argument('--chair_only_split', default=os.path.join(HERE, 'r2n2_shapenet_dataset', 'split_03001627.json'),
                        type=str, help='Part 2 split, only read to report train/test overlap in compare mode')
    parser.add_argument('--output_dir', default='outputs/q3_3', type=str)
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--max_samples', default=None, type=int,
                        help='use only the first N objects of EACH class (smoke tests)')
    parser.add_argument('--vox_threshold', default=0.5, type=float, help='occupancy threshold on sigmoid(logits)')
    # eval / visualization (defaults as in eval_model.py / q2_viz.py)
    parser.add_argument('--n_points', default=1000, type=int, help='surface samples for F1')
    parser.add_argument('--num_examples', default=1, type=int, help='eval: examples per class')
    parser.add_argument('--num_compare_examples', default=3, type=int, help='compare: chair examples')
    parser.add_argument('--vis_idxs', default=None, type=int, nargs='*',
                        help='compare: positions within the evaluated chair list to visualize')
    parser.add_argument('--image_size', default=256, type=int)
    parser.add_argument('--dist', default=1.4, type=float)
    parser.add_argument('--elev', default=20.0, type=float)
    parser.add_argument('--azim', default=45.0, type=float)
    parser.add_argument('--point_radius', default=0.01, type=float,
                        help='read by q1_viz.render_views (also on its empty-prediction path)')
    return parser


# ---------------------------------------------------------------------------- data

def dataset_paths(args):
    """Same layout as dataset_location.py with use_full_dataset=True."""
    root = args.dataset_root
    return os.path.join(root, 'shapenet'), os.path.join(root, 'r2n2'), os.path.join(root, 'split_3c.json')


def load_split(args, split):
    shapenet, r2n2, split_file = dataset_paths(args)
    for p in (shapenet, r2n2, split_file):
        if not os.path.exists(p):
            raise SystemExit(f'Full dataset not found: {p} is missing (see --dataset_root).')
    return R2N2(split, shapenet, r2n2, split_file, return_voxels=True, return_feats=args.load_feat)


def class_indices(dataset, max_per_class=None):
    """{synset: [dataset indices]} for the three classes, optionally capped per class."""
    out = {}
    for synset in CLASS_NAMES:
        if synset not in dataset.synset_start_idxs:
            continue
        start, n = dataset.synset_start_idxs[synset], dataset.synset_num_models[synset]
        out[synset] = list(range(start, start + (n if max_per_class is None else min(n, max_per_class))))
    return out


def print_dataset_summary(args):
    _, _, split_file = dataset_paths(args)
    with open(split_file) as f:
        splits = json.load(f)
    rows = []
    for synset in sorted(set(splits['train']) | set(splits['test'])):
        name = CLASS_NAMES.get(synset, synset)
        n_tr, n_te = len(splits['train'].get(synset, {})), len(splits['test'].get(synset, {}))
        rows.append((name, synset, n_tr, n_te))
    print(f'split file: {split_file}')
    print(f'{"class":>6} {"synset":>9} {"train models":>13} {"test models":>12}')
    for r in rows:
        print(f'{r[0]:>6} {r[1]:>9} {r[2]:>13} {r[3]:>12}')
    print(f'{"total":>6} {"":>9} {sum(r[2] for r in rows):>13} {sum(r[3] for r in rows):>12}')
    missing = [CLASS_NAMES[s] for s in CLASS_NAMES if s not in splits['train'] or s not in splits['test']]
    if missing:
        raise SystemExit(f'split_3c.json is missing classes: {missing}')
    return splits


def view_seed(seed, dataset_idx):
    return seed * 1_000_003 + dataset_idx


class FixedViewDataset(torch.utils.data.Dataset):
    """Test-time wrapper: the random input view r2n2_custom.R2N2 picks for item i depends only on
    (seed, i), independent of worker count / order, so every model and every run sees the same view.

    R2N2.__getitem__ makes exactly one draw from Python's `random` (randint over the item's views);
    replaying that draw recovers which view was chosen.
    """

    def __init__(self, base, indices, seed):
        self.base, self.indices, self.seed = base, list(indices), seed

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, j):
        i = self.indices[j]
        state = random.getstate()
        random.seed(view_seed(self.seed, i))
        item = self.base[i]
        random.seed(view_seed(self.seed, i))
        views = self.base.views_per_model_list[i]
        item['view_idx'] = views[random.randint(0, len(views) - 1)]
        random.setstate(state)
        item['dataset_idx'] = i
        return item


def make_loader(dataset, args, shuffle):
    return torch.utils.data.DataLoader(
        dataset, batch_size=args.batch_size, num_workers=args.num_workers,
        collate_fn=collate_batched_R2N2, pin_memory=True, drop_last=shuffle, shuffle=shuffle)


def model_input(feed_dict, load_feat, device):
    """Exactly the view the model sees: RGB B x H x W x 3, or that view's B x 512 ResNet18 feature."""
    if load_feat:
        return torch.stack(feed_dict['feats']).to(device)
    return feed_dict['images'].to(device)


# ---------------------------------------------------------------------------- model / checkpoint

def model_args(args, load_feat):
    """Namespace for model.SingleViewto3D: Part 2 voxel model with the given --load_feat."""
    margs = argparse.Namespace(**vars(args))
    margs.type, margs.load_feat = 'vox', load_feat
    return margs


def read_checkpoint(path, device):
    if not os.path.isfile(path):
        raise SystemExit(f'Checkpoint not found: {os.path.abspath(path)}')
    return torch.load(path, map_location=device, weights_only=True)


def checkpoint_load_feat(ckpt):
    """--load_feat the checkpoint was trained with (stored args, else presence of encoder weights)."""
    stored = (ckpt.get('args') or {}).get('load_feat')
    inferred = not any(k.startswith('encoder.') for k in ckpt['model_state_dict'])
    if stored is not None and stored != inferred:
        raise SystemExit(f'Inconsistent checkpoint: args.load_feat={stored} but encoder weights '
                         f'{"absent" if inferred else "present"}.')
    return inferred


def build_model(args, load_feat, ckpt=None):
    margs = model_args(args, load_feat)
    model = SingleViewto3D(margs).to(args.device)
    if ckpt is not None:
        q2_viz.check_compatibility(model, ckpt['model_state_dict'], margs)
        model.load_state_dict(ckpt['model_state_dict'])
    return model, margs


def save_checkpoint(model, optimizer, step, args):
    torch.save({'step': step,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'args': vars(args)}, args.checkpoint)
    print(f'Saved {args.checkpoint} (step {step})')


# ---------------------------------------------------------------------------- train

def train(args):
    if os.path.basename(args.checkpoint) in PROTECTED_CHECKPOINTS:
        raise SystemExit(f'Refusing to write {args.checkpoint}: it belongs to another part of the assignment.')
    print_dataset_summary(args)
    dataset = load_split(args, 'train')
    if args.max_samples:
        dataset = torch.utils.data.Subset(
            dataset, [i for idxs in class_indices(dataset, args.max_samples).values() for i in idxs])
    loader = make_loader(dataset, args, shuffle=True)

    model, margs = build_model(args, args.load_feat)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    start_iter = 0
    if args.load_checkpoint:
        ckpt = read_checkpoint(args.checkpoint, args.device)
        if checkpoint_load_feat(ckpt) != args.load_feat:
            raise SystemExit(f'{args.checkpoint} was trained with load_feat={checkpoint_load_feat(ckpt)}; '
                             f'pass {"--load_feat" if checkpoint_load_feat(ckpt) else "no --load_feat"}.')
        q2_viz.check_compatibility(model, ckpt['model_state_dict'], margs)
        model.load_state_dict(ckpt['model_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        start_iter = ckpt['step'] + 1  # 'step' = last completed step
        print(f'Loaded {args.checkpoint} (step {ckpt["step"]}), resuming at step {start_iter}')
        del ckpt
    model.train()

    print(f'Starting training ({len(dataset)} train objects, batch {args.batch_size}, '
          f'load_feat={args.load_feat}, steps {start_iter}..{args.max_iter - 1})')
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
        images = model_input(feed_dict, args.load_feat, args.device)
        voxels = feed_dict['voxels'].float().to(args.device)  # B x 1 x 32 x 32 x 32
        read_time = time.time() - iter_start_time

        logits = model(images, margs)  # B x 1 x 32 x 32 x 32, raw logits
        loss = losses.voxel_loss(logits, voxels)

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


# ---------------------------------------------------------------------------- metrics

def sample_seed(seed, dataset_idx, what):
    return (seed * 1_000_003 + dataset_idx) * 2 + {'gt': 0, 'pred': 1}[what]


def vox_pred_points(logits, args):
    """Predicted surface points in the eval frame: eval_model.evaluate's vox branch, with
    q2_viz.evaluate_sample's --vox_threshold shift. Returns 1 x n_points x 3, or None if the
    thresholded grid has no surface (scored 0, as in q2_viz)."""
    probs = torch.sigmoid(logits.detach()).cpu()
    t = args.vox_threshold
    if not ((probs > t).any() and (probs < t).any()):
        return None
    voxels_src = probs - t + 0.5
    H, W, D = voxels_src.shape[2:]
    vertices_src, faces_src = mcubes.marching_cubes(voxels_src.squeeze().numpy(), isovalue=0.5)
    vertices_src = torch.tensor(vertices_src).float()
    faces_src = torch.tensor(faces_src.astype(int))
    mesh_src = Meshes([vertices_src], [faces_src])
    pred_points = sample_points_from_meshes(mesh_src, args.n_points)
    pred_points = utils_vox.Mem2Ref(pred_points, H, W, D)
    axis_angle = torch.as_tensor(np.array([[0.0, -math.pi, 0.0]]))
    pred_points = Rotate(axis_angle_to_matrix(axis_angle)).transform_points(pred_points)
    return pred_points - pred_points.mean(1, keepdim=True)


def gt_points_eval(mesh_gt, args):
    gt_points = sample_points_from_meshes(mesh_gt, args.n_points)
    return gt_points - gt_points.mean(1, keepdim=True)


def score(pred_points, gt_points):
    """(precision, recall, F1) tensors over THRESHOLDS, 0-100 scale."""
    if pred_points is None:
        z = torch.zeros(len(THRESHOLDS))
        return z, z.clone(), z.clone()
    m = eval_model.compute_sampling_metrics(pred_points, gt_points, THRESHOLDS)
    return tuple(torch.tensor([m[f'{k}@{t:f}'].item() for t in THRESHOLDS]) for k in ('Precision', 'Recall', 'F1'))


def seeded(seed, fn, *a):
    torch.manual_seed(seed)
    return fn(*a)


# ---------------------------------------------------------------------------- rendering

def render(obj, args):
    """q2_viz.render (same camera, lights, shader, colors) with naive rasterization (bin_size=0).

    Car/plane GT meshes have up to ~220k faces, which overflow the coarse-to-fine rasterizer's bins
    and silently drop triangles; naive rasterization draws every face. Rendering only, not metrics.
    """
    default = q1_viz.RasterizationSettings
    q1_viz.RasterizationSettings = functools.partial(default, bin_size=0)
    try:
        return q2_viz.render(obj, args)
    finally:
        q1_viz.RasterizationSettings = default


def render_pred(logits, args):
    return render(q2_viz.pred_renderable(logits, args), args)


def render_gt(mesh_gt, seed, args):
    torch.manual_seed(seed)  # gt_mesh_for_render re-centers with random surface samples
    return render(q1_viz.colored_mesh(q2_viz.gt_mesh_for_render(mesh_gt, args), q2_viz.GT_COLOR), args)


def show_row(axes, images, titles):
    for ax, img, title in zip(axes, images, titles):
        ax.imshow(img)
        ax.set_title(title, fontsize=9)
        ax.axis('off')


def sample_label(row):
    return f'{row["model_id"][:12]} view {row["view_idx"]}'


# ---------------------------------------------------------------------------- eval (three-class model)

def mean_table(rows_p, rows_r, rows_f):
    return tuple(torch.stack(x).mean(0) for x in (rows_p, rows_r, rows_f))


def evaluate(args):
    if not args.load_checkpoint:
        raise SystemExit('Refusing to evaluate an untrained model: pass --load_checkpoint.')
    os.makedirs(args.output_dir, exist_ok=True)
    print_dataset_summary(args)
    ckpt = read_checkpoint(args.checkpoint, args.device)
    if checkpoint_load_feat(ckpt) != args.load_feat:
        raise SystemExit(f'{args.checkpoint} was trained with load_feat={checkpoint_load_feat(ckpt)}; '
                         f'pass {"--load_feat" if checkpoint_load_feat(ckpt) else "no --load_feat"}.')
    model, margs = build_model(args, args.load_feat, ckpt)
    ckpt_step = ckpt['step']
    del ckpt
    model.eval()
    print(f'Loaded {args.checkpoint} (step {ckpt_step})')

    base = load_split(args, 'test')
    per_class = class_indices(base, args.max_samples)
    order = [i for idxs in per_class.values() for i in idxs]
    vis_set = {i for idxs in per_class.values() for i in idxs[:args.num_examples]}
    loader = make_loader(FixedViewDataset(base, order, args.seed), args, shuffle=False)

    records, vis = [], {}
    with torch.no_grad():
        for feed_dict in loader:
            logits = model(model_input(feed_dict, args.load_feat, args.device), margs)
            for b in range(logits.shape[0]):
                i = feed_dict['dataset_idx'][b]
                mesh_gt = feed_dict['mesh'][b]
                gt_pts = seeded(sample_seed(args.seed, i, 'gt'), gt_points_eval, mesh_gt, args)
                pred_pts = seeded(sample_seed(args.seed, i, 'pred'), vox_pred_points, logits[b:b + 1], args)
                p, r, f = score(pred_pts, gt_pts)
                rec = {'dataset_idx': i, 'synset': feed_dict['synset_id'][b], 'model_id': feed_dict['model_id'][b],
                       'view_idx': feed_dict['view_idx'][b], 'empty': pred_pts is None, 'p': p, 'r': r, 'f1': f}
                records.append(rec)
                if i in vis_set:
                    vis[i] = dict(rec, rgb=feed_dict['images'][b].numpy().clip(0, 1),
                                  pred=render_pred(logits[b:b + 1], margs),
                                  gt=render_gt(mesh_gt, sample_seed(args.seed, i, 'gt'), margs))
            if len(records) % 100 == 0 or len(records) == len(order):
                f05 = torch.stack([x['f1'] for x in records])[:, K05].mean()
                print(f'[{len(records):4d}/{len(order):4d}] running avg F1@{KEY_T}: {f05:.3f}')
            del logits

    # ---- qualitative
    rows = []
    for synset, idxs in per_class.items():
        for k, i in enumerate(idxs[:args.num_examples]):
            row = vis[i]
            rows.append(row)
            fig, axes = plt.subplots(1, 3, figsize=(9, 3.4))
            show_row(axes, [row['rgb'], row['pred'], row['gt']],
                     ['Input RGB', f'Predicted voxels (F1@{KEY_T} = {row["f1"][K05]:.1f})', 'Ground truth mesh'])
            fig.suptitle(f'Three-class model: {CLASS_NAMES[synset]} ({sample_label(row)})')
            fig.tight_layout()
            fig.savefig(os.path.join(args.output_dir, f'{CLASS_NAMES[synset]}_example_{k}.png'),
                        dpi=120, bbox_inches='tight')
            plt.close(fig)
    if rows:
        fig, axes = plt.subplots(len(rows), 3, figsize=(9, 3.2 * len(rows)), squeeze=False)
        for ax_row, row in zip(axes, rows):
            show_row(ax_row, [row['rgb'], row['pred'], row['gt']],
                     ['Input RGB', f'Predicted voxels (F1@{KEY_T} = {row["f1"][K05]:.1f})', 'Ground truth mesh'])
            ax_row[0].text(-0.08, 0.5, CLASS_NAMES[row['synset']], transform=ax_row[0].transAxes,
                           rotation=90, va='center', ha='right', fontsize=11)
        fig.suptitle('Q3.3 three-class voxel model: Input RGB | Prediction | GT mesh')
        fig.tight_layout()
        fig.savefig(os.path.join(args.output_dir, 'examples.png'), dpi=120, bbox_inches='tight')
        plt.close(fig)

    # ---- quantitative
    groups = {'overall': records}
    groups.update({CLASS_NAMES[s]: [x for x in records if x['synset'] == s] for s in per_class})
    tables = {g: mean_table([x['p'] for x in rs], [x['r'] for x in rs], [x['f1'] for x in rs])
              for g, rs in groups.items() if rs}

    fig, ax = plt.subplots()
    for g, (_, _, f) in tables.items():
        ax.plot(THRESHOLDS, f.numpy(), marker='o', label=f'{g} (n={len(groups[g])})',
                linewidth=2.5 if g == 'overall' else 1.5)
    ax.set_xlabel('Threshold')
    ax.set_ylabel('F1-score')
    ax.set_title('Evaluation vox, three-class model (Q3.3)')
    ax.grid(alpha=0.3)
    ax.legend()
    fig.savefig(os.path.join(args.output_dir, 'f1.png'), bbox_inches='tight')
    plt.close(fig)

    lines = [
        'Q3.3 three-class voxel model, evaluated on the split_3c.json test split',
        f'checkpoint: {os.path.abspath(args.checkpoint)} (step {ckpt_step})',
        f'split file: {dataset_paths(args)[2]}',
        f'load_feat: {args.load_feat}',
        f'vox_threshold (on sigmoid(logits)): {args.vox_threshold}',
        f'n_points: {args.n_points}',
        f'seed: {args.seed} (input view and point samples are fixed per sample by the seed)',
        f'samples evaluated: {len(records)}' + (f' (partial, --max_samples {args.max_samples} per class)'
                                                 if args.max_samples else ' (full test split)'),
    ]
    for g, (p, r, f) in tables.items():
        n_empty = sum(x['empty'] for x in groups[g])
        lines += ['', f'== {g}: {len(groups[g])} samples, {n_empty} with empty predicted surface (scored 0)',
                  f'{"threshold":>9} {"precision":>10} {"recall":>10} {"F1":>10}']
        lines += [f'{t:>9.2f} {p[k]:>10.3f} {r[k]:>10.3f} {f[k]:>10.3f}' for k, t in enumerate(THRESHOLDS)]
    lines += ['', f'*** overall F1@{KEY_T}: {tables["overall"][2][K05]:.3f} ***']
    lines += [f'*** {CLASS_NAMES[s]} F1@{KEY_T}: {tables[CLASS_NAMES[s]][2][K05]:.3f} ***' for s in per_class]
    lines += [f'mean of per-class F1@{KEY_T}: '
              f'{np.mean([tables[CLASS_NAMES[s]][2][K05].item() for s in per_class]):.3f}']
    write_text(os.path.join(args.output_dir, 'metrics.txt'), lines)
    write_csv(os.path.join(args.output_dir, 'per_sample.csv'), records,
              extra=lambda x: {'class': CLASS_NAMES[x['synset']]}, keys=[('three_class', None)])
    print(f'Saved outputs to {args.output_dir}/')


# ---------------------------------------------------------------------------- compare (chair-only vs three-class)

def pairwise_iou_mean(occ):
    """Mean IoU over all pairs i < j of the N x V boolean grids (how alike the shapes are)."""
    x = occ.float()
    inter = x @ x.T
    s = x.sum(1)
    union = s[:, None] + s[None, :] - inter
    iou = torch.where(union > 0, inter / union.clamp(min=1), torch.ones_like(inter))
    n = x.shape[0]
    if n < 2:
        return float('nan')
    return ((iou.sum() - iou.diagonal().sum()) / (n * (n - 1))).item()


def compare(args):
    os.makedirs(args.output_dir, exist_ok=True)
    splits = print_dataset_summary(args)

    models = {}
    for key, path, label in (('single', args.single_class_checkpoint, 'chair-only'),
                             ('three', args.checkpoint, 'three-class')):
        ckpt = read_checkpoint(path, args.device)
        lf = checkpoint_load_feat(ckpt)
        model, margs = build_model(args, lf, ckpt)
        model.eval()
        models[key] = {'model': model, 'margs': margs, 'load_feat': lf, 'path': path, 'label': label,
                       'step': ckpt['step'], 'train_args': ckpt.get('args')}
        print(f'Loaded {label} model {path} (step {ckpt["step"]}, load_feat={lf})')
        del ckpt
    load_feat = models['single']['load_feat']
    if models['three']['load_feat'] != load_feat:
        raise SystemExit('Unfair comparison: the chair-only model was trained with load_feat='
                         f'{load_feat} but the three-class model with load_feat={models["three"]["load_feat"]}. '
                         'Retrain the three-class model with the same setting.')
    if args.load_feat != load_feat:
        raise SystemExit(f'Both checkpoints were trained with load_feat={load_feat}; '
                         f'pass {"--load_feat" if load_feat else "no --load_feat"}.')

    base = load_split(args, 'test')
    chair_idxs = class_indices(base, args.max_samples)[CHAIR]
    loader = make_loader(FixedViewDataset(base, chair_idxs, args.seed), args, shuffle=False)
    vis_pos = args.vis_idxs if args.vis_idxs is not None else list(range(args.num_compare_examples))
    vis_set = {chair_idxs[k] for k in vis_pos if k < len(chair_idxs)}

    records, vis = [], {}
    occ = {'single': [], 'three': [], 'gt': []}
    with torch.no_grad():
        for feed_dict in loader:
            # ONE input tensor (same object, same view) is fed to both models.
            inputs = model_input(feed_dict, load_feat, args.device)
            logits = {k: m['model'](inputs, m['margs']) for k, m in models.items()}
            gt_vox = feed_dict['voxels'].float()
            for b in range(inputs.shape[0]):
                i = feed_dict['dataset_idx'][b]
                mesh_gt = feed_dict['mesh'][b]
                gt_pts = seeded(sample_seed(args.seed, i, 'gt'), gt_points_eval, mesh_gt, args)  # shared
                rec = {'dataset_idx': i, 'synset': feed_dict['synset_id'][b], 'model_id': feed_dict['model_id'][b],
                       'view_idx': feed_dict['view_idx'][b]}
                gt_occ = (gt_vox[b] > 0.5).flatten()
                occ['gt'].append(gt_occ)
                for k in models:
                    lg = logits[k][b:b + 1]
                    pred_pts = seeded(sample_seed(args.seed, i, 'pred'), vox_pred_points, lg, args)
                    p, r, f = score(pred_pts, gt_pts)
                    pred_occ = (torch.sigmoid(lg) > args.vox_threshold).flatten().cpu()
                    occ[k].append(pred_occ)
                    union = (pred_occ | gt_occ).sum().item()
                    rec[k] = {'p': p, 'r': r, 'f1': f, 'empty': pred_pts is None,
                              'iou': (pred_occ & gt_occ).sum().item() / union if union else 1.0,
                              'n_occ': pred_occ.sum().item()}
                rec['gt_n_occ'] = gt_occ.sum().item()
                records.append(rec)
                if i in vis_set:
                    vis[i] = dict(rec, rgb=feed_dict['images'][b].numpy().clip(0, 1),
                                  gt=render_gt(mesh_gt, sample_seed(args.seed, i, 'gt'), models['single']['margs']),
                                  **{f'img_{k}': render_pred(logits[k][b:b + 1], m['margs']) for k, m in models.items()})
            if len(records) % 100 == 0 or len(records) == len(chair_idxs):
                f = {k: torch.stack([x[k]['f1'] for x in records])[:, K05].mean().item() for k in models}
                print(f'[{len(records):4d}/{len(chair_idxs):4d}] chair F1@{KEY_T}: chair-only {f["single"]:.3f}, '
                      f'three-class {f["three"]:.3f}')
            del logits, inputs

    # ---- qualitative
    rows = [vis[chair_idxs[k]] for k in vis_pos if k < len(chair_idxs)]
    titles = lambda row: ['Input RGB',
                          f'Chair-only prediction\nF1@{KEY_T} = {row["single"]["f1"][K05]:.1f}',
                          f'Three-class prediction\nF1@{KEY_T} = {row["three"]["f1"][K05]:.1f}',
                          'Ground truth mesh']
    for n, row in enumerate(rows):
        fig, axes = plt.subplots(1, 4, figsize=(12, 3.6))
        show_row(axes, [row['rgb'], row['img_single'], row['img_three'], row['gt']], titles(row))
        fig.suptitle(f'Chair test sample ({sample_label(row)}), same input view for both models')
        fig.tight_layout()
        fig.savefig(os.path.join(args.output_dir, f'chair_comparison_{n}.png'), dpi=120, bbox_inches='tight')
        plt.close(fig)
    if rows:
        fig, axes = plt.subplots(len(rows), 4, figsize=(12, 3.4 * len(rows)), squeeze=False)
        for ax_row, row in zip(axes, rows):
            show_row(ax_row, [row['rgb'], row['img_single'], row['img_three'], row['gt']], titles(row))
        fig.suptitle('Chair test samples: Input RGB | Chair-only | Three-class | GT mesh')
        fig.tight_layout()
        fig.savefig(os.path.join(args.output_dir, 'chair_comparison.png'), dpi=120, bbox_inches='tight')
        plt.close(fig)

    # ---- quantitative
    tables = {k: mean_table([x[k]['p'] for x in records], [x[k]['r'] for x in records],
                            [x[k]['f1'] for x in records]) for k in models}
    fig, ax = plt.subplots()
    for k, m in models.items():
        ax.plot(THRESHOLDS, tables[k][2].numpy(), marker='o', label=m['label'])
    ax.set_xlabel('Threshold')
    ax.set_ylabel('F1-score')
    ax.set_title(f'Chair test F1, n={len(records)}')
    ax.grid(alpha=0.3)
    ax.legend()
    fig.savefig(os.path.join(args.output_dir, 'comparison_f1.png'), bbox_inches='tight')
    plt.close(fig)

    f05 = {k: torch.stack([x[k]['f1'][K05] for x in records]) for k in models}
    diff = f05['three'] - f05['single']
    iou = {k: np.array([x[k]['iou'] for x in records]) for k in models}
    n_occ = {k: np.array([x[k]['n_occ'] for x in records]) for k in models}
    gt_n_occ = np.array([x['gt_n_occ'] for x in records])
    pair_iou = {k: pairwise_iou_mean(torch.stack(v).to(args.device)) for k, v in occ.items()}

    # overlap of the evaluated chairs with each model's training split
    eval_ids = {x['model_id'] for x in records}
    overlap_3c = len(eval_ids & set(splits['train'].get(CHAIR, {})))
    overlap_1c = 'n/a (chair-only split file not found)'
    same_test_split = 'n/a'
    if os.path.isfile(args.chair_only_split):
        with open(args.chair_only_split) as fh:
            split_1c = json.load(fh)
        overlap_1c = len(eval_ids & set(split_1c['train'].get(CHAIR, {})))
        same_test_split = set(split_1c['test'].get(CHAIR, {})) == set(splits['test'][CHAIR])

    def cfg(m):
        ta = m['train_args']
        if not ta:
            return 'no training args stored in checkpoint (train_model.py format)'
        return ', '.join(f'{k}={ta.get(k)}' for k in ('lr', 'batch_size', 'max_iter', 'arch', 'load_feat', 'seed'))

    lines = [
        'Q3.3 comparison on chair test samples: chair-only training vs chair+car+plane training',
        '',
        'checkpoints',
        f'  chair-only : {os.path.abspath(models["single"]["path"])} (step {models["single"]["step"]}; {cfg(models["single"])})',
        f'  three-class: {os.path.abspath(models["three"]["path"])} (step {models["three"]["step"]}; {cfg(models["three"])})',
        '',
        'evaluation configuration (identical for both models)',
        f'  test samples: split_3c.json chair ({CHAIR}) test split, {dataset_paths(args)[2]}',
        f'  chair samples evaluated: {len(records)}' + (f' (partial, --max_samples {args.max_samples})'
                                                        if args.max_samples else ' (all chair test objects)'),
        f'  input: one view per object, chosen by seed {args.seed}; the SAME input tensor is fed to both models',
        f'  load_feat: {load_feat}',
        f'  architecture: model.SingleViewto3D type=vox (ResNet18 {args.arch} -> 512 -> Linear -> 256x4^3 -> '
        f'3x ConvTranspose3d -> 1x32^3 logits)',
        f'  vox_threshold (on sigmoid(logits)): {args.vox_threshold}',
        f'  F1: eval_model.compute_sampling_metrics, 0-100 scale, thresholds {THRESHOLDS}',
        f'  n_points: {args.n_points} pred / {args.n_points} GT surface samples; GT samples shared by both models',
        '  transforms: marching cubes -> utils_vox.Mem2Ref -> -pi rotation about y -> re-center (pred and GT)',
        '  empty predicted surface -> precision = recall = F1 = 0 (as in q2_viz)',
        '',
        'data overlap (evaluated chair objects that appear in a training split)',
        f'  in split_3c.json chair train split: {overlap_3c}',
        f'  in chair-only split_03001627.json train split: {overlap_1c}',
        f'  chair test split identical in both split files: {same_test_split}',
        '',
        f'{"threshold":>9} | {"chair-only P":>12} {"R":>7} {"F1":>7} | {"three-class P":>13} {"R":>7} {"F1":>7} | {"F1 diff":>8}',
    ]
    s, t3 = tables['single'], tables['three']
    for k, th in enumerate(THRESHOLDS):
        lines.append(f'{th:>9.2f} | {s[0][k]:>12.3f} {s[1][k]:>7.3f} {s[2][k]:>7.3f} | '
                     f'{t3[0][k]:>13.3f} {t3[1][k]:>7.3f} {t3[2][k]:>7.3f} | {t3[2][k] - s[2][k]:>+8.3f}')
    lines += [
        '',
        f'*** chair-only  chair F1@{KEY_T}: {s[2][K05]:.3f} ***',
        f'*** three-class chair F1@{KEY_T}: {t3[2][K05]:.3f} ***',
        f'*** difference (three-class - chair-only): {t3[2][K05] - s[2][K05]:+.3f} ***',
        '',
        f'per-sample F1@{KEY_T} (paired over the same {len(records)} samples)',
        f'  chair-only : mean {f05["single"].mean():.3f}, std {f05["single"].std():.3f}, median {f05["single"].median():.3f}',
        f'  three-class: mean {f05["three"].mean():.3f}, std {f05["three"].std():.3f}, median {f05["three"].median():.3f}',
        f'  difference : mean {diff.mean():+.3f}, std {diff.std():.3f}',
        f'  three-class higher on {(diff > 0).sum().item()}, lower on {(diff < 0).sum().item()}, '
        f'equal on {(diff == 0).sum().item()} samples',
        f'  empty predicted surfaces: chair-only {sum(x["single"]["empty"] for x in records)}, '
        f'three-class {sum(x["three"]["empty"] for x in records)}',
        '',
        f'supplementary voxel statistics (32^3 grid, prediction = sigmoid(logits) > {args.vox_threshold})',
        f'  voxel IoU vs GT voxels: chair-only {iou["single"].mean():.4f}, three-class {iou["three"].mean():.4f}',
        f'  occupied voxels per object: chair-only {n_occ["single"].mean():.1f} (std {n_occ["single"].std():.1f}), '
        f'three-class {n_occ["three"].mean():.1f} (std {n_occ["three"].std():.1f}), '
        f'GT {gt_n_occ.mean():.1f} (std {gt_n_occ.std():.1f})',
        '  mean pairwise IoU between the grids of different objects (higher = shapes more alike each other):',
        f'    chair-only {pair_iou["single"]:.4f}, three-class {pair_iou["three"]:.4f}, GT {pair_iou["gt"]:.4f}',
        '',
        'qualitative examples (chair_comparison_*.png): ' + ', '.join(
            f'#{n}: dataset idx {r["dataset_idx"]}, {r["model_id"]}, view {r["view_idx"]}' for n, r in enumerate(rows)),
    ]
    write_text(os.path.join(args.output_dir, 'comparison_metrics.txt'), lines)
    write_csv(os.path.join(args.output_dir, 'comparison_per_sample.csv'), records,
              keys=[('chair_only', 'single'), ('three_class', 'three')], extra=lambda x: {'gt_n_occ': x['gt_n_occ']})
    print(f'Saved outputs to {args.output_dir}/')


# ---------------------------------------------------------------------------- output helpers

def write_text(path, lines):
    text = '\n'.join(lines) + '\n'
    with open(path, 'w') as f:
        f.write(text)
    print('\n' + text)


def write_csv(path, records, keys, extra=lambda x: {}):
    """keys: [(column prefix, record sub-dict key or None for top level)]."""
    rows = []
    for x in records:
        row = {'dataset_idx': x['dataset_idx'], 'model_id': x['model_id'], 'view_idx': x['view_idx']}
        row.update(extra(x))
        for prefix, k in keys:
            d = x if k is None else x[k]
            row.update({f'{prefix}_F1@{t}': round(d['f1'][j].item(), 4) for j, t in enumerate(THRESHOLDS)})
            row[f'{prefix}_empty'] = d['empty']
            for extra_key in ('iou', 'n_occ'):
                if extra_key in d:
                    row[f'{prefix}_{extra_key}'] = d[extra_key]
        rows.append(row)
    with open(path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]) if rows else ['dataset_idx'])
        w.writeheader()
        w.writerows(rows)


def main(args):
    if args.batch_size is None:
        args.batch_size = 32 if args.mode == 'train' else 1
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    {'train': train, 'eval': evaluate, 'compare': compare}[args.mode](args)


if __name__ == '__main__':
    main(get_args_parser().parse_args())
