"""Q2: qualitative + quantitative evaluation of a trained single-view-to-3D checkpoint.

Evaluates the whole test split with eval_model.py's metric (F1 on a 0-100 scale), and renders
`Input RGB | Prediction | GT mesh` panels for a few test examples.

    python q2_viz.py --type vox --device cuda --load_checkpoint
"""
import argparse
import math
import os
import random

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import mcubes
import numpy as np
import torch
from pytorch3d.datasets.r2n2.utils import collate_batched_R2N2
from pytorch3d.ops import sample_points_from_meshes
from pytorch3d.structures import Meshes, Pointclouds
from pytorch3d.transforms import Rotate, axis_angle_to_matrix

import dataset_location
import eval_model
import q1_viz
import utils_vox
from model import SingleViewto3D
from r2n2_custom import R2N2

THRESHOLDS = [0.01, 0.02, 0.03, 0.04, 0.05]
KEY_T = 0.05
PRED_COLOR = q1_viz.SRC_COLOR
GT_COLOR = q1_viz.TGT_COLOR
CENTER_SAMPLES = 10000  # points used to estimate centroids for the vox render frame


def get_args_parser():
    # eval_model's parser supplies --type/--device/--batch_size/--num_workers/--n_points/
    # --load_feat/--load_checkpoint/--arch with identical defaults.
    parser = argparse.ArgumentParser('Q2 Visualization', parents=[eval_model.get_args_parser()])
    parser.add_argument('--checkpoint', default=None, type=str,
                        help='checkpoint path (default: checkpoint_{type}.pth, as in eval_model.py)')
    parser.add_argument('--num_examples', default=3, type=int)
    parser.add_argument('--vis_idxs', default=None, type=int, nargs='*',
                        help='test-set indices to visualize (default: the first --num_examples)')
    parser.add_argument('--output_dir', default='outputs/q2', type=str)
    parser.add_argument('--image_size', default=256, type=int)
    parser.add_argument('--dist', default=1.4, type=float)
    parser.add_argument('--elev', default=20.0, type=float)
    parser.add_argument('--azim', default=45.0, type=float)
    parser.add_argument('--point_radius', default=0.01, type=float)
    parser.add_argument('--vox_threshold', default=0.5, type=float,
                        help='occupancy threshold on sigmoid(logits)')
    parser.add_argument('--max_samples', default=None, type=int,
                        help='only evaluate the first N test samples (for smoke tests)')
    parser.add_argument('--seed', default=0, type=int,
                        help='seeds the random input view per item and point sampling')
    parser.add_argument('--allow_untrained', action='store_true',
                        help='debug only: run without a checkpoint (random decoder weights)')
    return parser


# ---------------------------------------------------------------------------- model / checkpoint

def load_model(args):
    ckpt_path = args.checkpoint or f'checkpoint_{args.type}.pth'
    if not args.load_checkpoint and not args.allow_untrained:
        raise SystemExit('Refusing to evaluate an untrained model: pass --load_checkpoint '
                         '(or --allow_untrained for pipeline debugging only).')
    if args.load_checkpoint and not os.path.isfile(ckpt_path):
        raise SystemExit(f'Checkpoint not found: {os.path.abspath(ckpt_path)}\n'
                         f'Train first: python train_model.py --type {args.type}'
                         f'{" --load_feat" if args.load_feat else ""}')

    model = SingleViewto3D(args)
    if not hasattr(model, 'decoder'):
        raise SystemExit(f'model.py defines no decoder for --type {args.type} yet (still a TODO).')
    model.to(args.device)
    model.eval()

    if not args.load_checkpoint:
        print('WARNING: --allow_untrained set, evaluating RANDOMLY INITIALIZED decoder weights.')
        return model, None

    try:
        checkpoint = torch.load(ckpt_path, map_location=args.device, weights_only=True)
    except TypeError:  # torch without weights_only
        checkpoint = torch.load(ckpt_path, map_location=args.device)
    state = checkpoint['model_state_dict']
    check_compatibility(model, state, args)
    model.load_state_dict(state)
    print(f'Loaded {ckpt_path} (step {checkpoint.get("step", "?")})')
    return model, ckpt_path


def check_compatibility(model, state, args):
    """Explain --load_feat / --n_points / architecture mismatches instead of a raw state_dict error."""
    own = model.state_dict()
    ckpt_has_encoder = any(k.startswith('encoder.') for k in state)
    problems = []
    if ckpt_has_encoder and args.load_feat:
        problems.append('checkpoint contains image-encoder weights (trained WITHOUT --load_feat), '
                        'but --load_feat was passed: drop --load_feat')
    if not ckpt_has_encoder and not args.load_feat:
        problems.append('checkpoint has no image-encoder weights (trained WITH --load_feat): '
                        'pass --load_feat')
    decoder_mismatch = False
    for k, v in state.items():
        if k in own and own[k].shape != v.shape:
            problems.append(f'shape mismatch for {k}: checkpoint {tuple(v.shape)} vs model {tuple(own[k].shape)}')
            decoder_mismatch = True
    if decoder_mismatch and args.type == 'point':
        problems.append(f'for --type point this usually means --n_points differs from training '
                        f'(currently --n_points {args.n_points})')
    missing = [k for k in own if k not in state and not k.startswith('encoder.')]
    unexpected = [k for k in state if k not in own and not k.startswith('encoder.')]
    if missing:
        problems.append(f'keys missing from checkpoint: {missing[:5]}{" ..." if len(missing) > 5 else ""}')
    if unexpected:
        problems.append(f'unexpected checkpoint keys: {unexpected[:5]}{" ..." if len(unexpected) > 5 else ""}')
    if problems:
        raise SystemExit('Checkpoint is incompatible with the current arguments/model.py:\n  - '
                         + '\n  - '.join(problems))


# ---------------------------------------------------------------------------- metrics

def evaluate_sample(pred, mesh_gt, args):
    """Per-sample metrics dict, exactly eval_model.evaluate; returns (metrics, empty_surface)."""
    if args.type == 'vox':
        probs = torch.sigmoid(pred)
        t = args.vox_threshold
        # no crossing of the threshold anywhere -> marching cubes yields no surface
        if not ((probs > t).any() and (probs < t).any()):
            zeros = torch.zeros(1)
            return {f'{m}@{th:f}': zeros for th in THRESHOLDS for m in ('Precision', 'Recall', 'F1')}, True
        # eval_model.evaluate runs marching cubes at isovalue 0.5; shifting the probabilities moves
        # that level set to sigmoid(logits) == vox_threshold (identical to 0.5 when t == 0.5).
        pred = probs - t + 0.5
    return eval_model.evaluate(pred, mesh_gt, THRESHOLDS, args), False


# ---------------------------------------------------------------------------- rendering

def vox_pred_mesh_eval_frame(logits, args):
    """Predicted voxel surface in the same frame eval_model uses for F1 (or None if empty)."""
    probs = torch.sigmoid(logits).detach().cpu()
    H, W, D = probs.shape[2:]
    verts, faces = mcubes.marching_cubes(probs.squeeze().numpy(), isovalue=args.vox_threshold)
    if len(faces) == 0:
        return None
    verts = utils_vox.Mem2Ref(torch.tensor(verts).float()[None], H, W, D)
    rot = axis_angle_to_matrix(torch.as_tensor(np.array([[0.0, -math.pi, 0.0]])))
    verts = Rotate(rot).transform_points(verts)[0].float()
    mesh = Meshes([verts], [torch.tensor(faces.astype(np.int64))])
    verts = verts - sample_points_from_meshes(mesh, CENTER_SAMPLES)[0].mean(0)  # eval re-centering
    return Meshes([verts.to(args.device)], [mesh.faces_packed().to(args.device)])


def gt_mesh_for_render(mesh_gt, args):
    verts, faces = mesh_gt.verts_packed(), mesh_gt.faces_packed()
    if args.type == 'vox':  # eval_model re-centers GT samples in vox mode only
        verts = verts - sample_points_from_meshes(mesh_gt, CENTER_SAMPLES)[0].mean(0)
    return Meshes([verts.to(args.device)], [faces.to(args.device)])


def pred_renderable(pred, args):
    if args.type == 'vox':
        mesh = vox_pred_mesh_eval_frame(pred, args)
        return q1_viz.colored_mesh(mesh, PRED_COLOR) if mesh is not None else None
    if args.type == 'point':
        points = pred.detach()[0]
        return Pointclouds(points=[points], features=[torch.ones_like(points) * torch.tensor(PRED_COLOR, device=points.device)])
    return q1_viz.colored_mesh(pred, PRED_COLOR)


def render(obj, args):
    return q1_viz.render_views(obj, [args.azim], args)[0]


def panel_title(example_num, f1):
    return f'Example {example_num} — F1@{KEY_T} = {f1:.1f}'


def draw_row(axes, row, args):
    pred_label = {'vox': 'Predicted voxels', 'point': 'Predicted point cloud', 'mesh': 'Predicted mesh'}[args.type]
    for ax, img, label in zip(axes, [row['rgb'], row['pred'], row['gt']], ['Input RGB', pred_label, 'Ground truth mesh']):
        ax.imshow(img)
        ax.set_title(label, fontsize=10)
        ax.axis('off')


def save_examples(rows, args):
    for i, row in enumerate(rows):
        fig, axes = plt.subplots(1, 3, figsize=(9, 3.4))
        draw_row(axes, row, args)
        fig.suptitle(panel_title(i + 1, row['f1']) + f'  (test idx {row["idx"]})')
        fig.tight_layout()
        fig.savefig(os.path.join(args.output_dir, f'{args.type}_example_{i}.png'), dpi=120, bbox_inches='tight')
        plt.close(fig)

    if not rows:
        return
    fig, axes = plt.subplots(len(rows), 3, figsize=(9, 3.2 * len(rows)), squeeze=False)
    for i, row in enumerate(rows):
        draw_row(axes[i], row, args)
        axes[i][0].text(-0.08, 0.5, panel_title(i + 1, row['f1']), transform=axes[i][0].transAxes,
                        rotation=90, va='center', ha='right', fontsize=10)
    fig.suptitle(f'{args.type}: Input RGB | Prediction | GT mesh')
    fig.tight_layout()
    fig.savefig(os.path.join(args.output_dir, f'{args.type}_examples.png'), dpi=120, bbox_inches='tight')
    plt.close(fig)


# ---------------------------------------------------------------------------- outputs

def save_f1_plot(avg_f1, args):
    fig, ax = plt.subplots()
    ax.plot(THRESHOLDS, avg_f1.numpy(), marker='o')
    ax.set_xlabel('Threshold')
    ax.set_ylabel('F1-score')
    ax.set_title(f'Evaluation {args.type}')
    ax.grid(alpha=0.3)
    fig.savefig(os.path.join(args.output_dir, f'{args.type}_f1.png'), bbox_inches='tight')
    plt.close(fig)


def write_summary(avg_p, avg_r, avg_f1, n_eval, n_empty, ckpt_path, args):
    k = THRESHOLDS.index(KEY_T)
    lines = [
        f'type: {args.type}',
        f'checkpoint: {os.path.abspath(ckpt_path) if ckpt_path else "NONE (untrained, --allow_untrained)"}',
        f'n_points: {args.n_points}',
        f'load_feat: {args.load_feat}',
        f'samples evaluated: {n_eval}' + (' (partial, --max_samples)' if args.max_samples else ' (full test split)'),
        f'seed: {args.seed}',
    ]
    if args.type == 'vox':
        lines += [f'vox_threshold (on sigmoid(logits)): {args.vox_threshold}',
                  f'samples with empty predicted surface (scored 0): {n_empty}']
    lines += ['', f'{"threshold":>9} {"precision":>10} {"recall":>10} {"F1":>10}']
    lines += [f'{t:>9.2f} {avg_p[i]:>10.3f} {avg_r[i]:>10.3f} {avg_f1[i]:>10.3f}' for i, t in enumerate(THRESHOLDS)]
    lines += ['', f'*** F1@{KEY_T}: {avg_f1[k]:.3f} ***',
              f'Average Precision@{KEY_T}: {avg_p[k]:.3f}',
              f'Average Recall@{KEY_T}: {avg_r[k]:.3f}',
              f'Average F1@{KEY_T}: {avg_f1[k]:.3f}']
    text = '\n'.join(lines) + '\n'
    with open(os.path.join(args.output_dir, f'{args.type}_metrics.txt'), 'w') as f:
        f.write(text)
    print('\n' + text)


# ---------------------------------------------------------------------------- main

def main(args):
    if args.type == 'vox' and args.batch_size != 1:
        raise SystemExit('--type vox requires --batch_size 1 (eval_model.evaluate squeezes the batch dim).')
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    model, ckpt_path = load_model(args)

    dataset = R2N2('test', dataset_location.SHAPENET_PATH, dataset_location.R2N2_PATH,
                   dataset_location.SPLITS_PATH, return_voxels=True, return_feats=args.load_feat)
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=args.batch_size, num_workers=args.num_workers,
        collate_fn=collate_batched_R2N2, pin_memory=True, drop_last=True)

    vis_idxs = args.vis_idxs if args.vis_idxs is not None else list(range(args.num_examples))
    vis_rows = {}
    all_p, all_r, all_f1 = [], [], []
    n_empty = 0
    n_total = len(loader) * args.batch_size
    if args.max_samples:
        n_total = min(n_total, args.max_samples)

    with torch.no_grad():
        for step, feed_dict in enumerate(loader):
            if step * args.batch_size >= n_total:
                break
            # images: exactly the (randomly chosen) view the model sees; with --load_feat the model
            # instead gets that same view's precomputed feature.
            images, mesh_gt = eval_model.preprocess(feed_dict, args)
            predictions = model(images, args)

            for b in range(images.shape[0]):
                idx = step * args.batch_size + b
                pred_b = predictions[b] if args.type == 'mesh' else predictions[b:b + 1]
                metrics, empty = evaluate_sample(pred_b, mesh_gt[b], args)
                n_empty += int(empty)
                all_p.append(torch.tensor([metrics['Precision@%f' % t].item() for t in THRESHOLDS]))
                all_r.append(torch.tensor([metrics['Recall@%f' % t].item() for t in THRESHOLDS]))
                all_f1.append(torch.tensor([metrics['F1@%f' % t].item() for t in THRESHOLDS]))

                if idx in vis_idxs:
                    vis_rows[idx] = {
                        'idx': idx,
                        'f1': all_f1[-1][THRESHOLDS.index(KEY_T)].item(),
                        'rgb': feed_dict['images'][b].cpu().numpy().clip(0, 1),
                        'pred': render(pred_renderable(pred_b, args), args),
                        'gt': render(q1_viz.colored_mesh(gt_mesh_for_render(mesh_gt[b], args), GT_COLOR), args),
                    }

            n_done = len(all_f1)
            if n_done % 50 == 0 or n_done == n_total:
                f1_05 = torch.stack(all_f1)[:, THRESHOLDS.index(KEY_T)].mean()
                print(f'[{n_done:4d}/{n_total:4d}] running avg F1@{KEY_T}: {f1_05:.3f}')
            del predictions, images

    avg_p, avg_r, avg_f1 = (torch.stack(x).mean(0) for x in (all_p, all_r, all_f1))
    missing = [i for i in vis_idxs if i not in vis_rows]
    if missing:
        print(f'WARNING: visualization indices not evaluated (out of range): {missing}')
    save_examples([vis_rows[i] for i in vis_idxs if i in vis_rows], args)
    save_f1_plot(avg_f1, args)
    write_summary(avg_p, avg_r, avg_f1, len(all_f1), n_empty, ckpt_path, args)
    print(f'Saved outputs to {args.output_dir}/{args.type}_*')


if __name__ == '__main__':
    main(get_args_parser().parse_args())
