"""Q2.5: input-occlusion sensitivity of the trained Part 2 voxel model (inference only).

For each chair test image, half of the image (top / bottom / left / right) is replaced by a constant
fill and the SAME trained voxel model (model.SingleViewto3D, type='vox', checkpoint_vox.pth) is run
on the original and the four occluded inputs. The change in the prediction is measured as voxel IoU
between the occluded-input occupancy and the original-input occupancy (sigmoid(logits) > threshold).
Higher IoU = the prediction changed less when that half was hidden.

Fill value: model.py feeds RGB in [0, 1] through Normalize(mean=ImageNet mean, std=ImageNet std).
Masked pixels are set to the ImageNet mean RGB (0.485, 0.456, 0.406) in that [0, 1] space, so after
normalization they are exactly 0 in every channel, i.e. the "average ImageNet pixel" the encoder
was trained around. Note it is not the R2N2 background colour (black).

Input views: identical to q2_viz.py with the same --seed. q2_viz seeds Python `random` once and loads
the test split in order with num_workers=0; r2n2_custom.R2N2 draws one random.randint per item to pick
the view, so the view for test index i is the (i+1)-th draw. That sequence is replayed here.

Rendering: q2_viz.pred_renderable / q2_viz.render (Part 2 voxel style: marching cubes at
--vox_threshold, eval frame, re-centred on the predicted surface). Because every prediction is
re-centred, a pure translation is not visible in the renders; the IoU is computed on the raw 32^3
grids and does include it.

    python q2_5.py --device cuda
    python q2_5.py --device cuda --vis_idxs 5 17 42
"""
import argparse
import math
import os
import random

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
from pytorch3d.structures import Meshes

import dataset_location
import q2_viz
from model import SingleViewto3D
from r2n2_custom import R2N2

IMAGENET_MEAN = (0.485, 0.456, 0.406)  # model.py's Normalize mean
CONDITIONS = ['original', 'top', 'bottom', 'left', 'right']
MASKS = CONDITIONS[1:]
KEY_T = q2_viz.KEY_T  # 0.05


def get_args_parser():
    parser = argparse.ArgumentParser('Q2.5 occlusion sensitivity of the voxel model')
    parser.add_argument('--device', default='cuda', type=str)
    parser.add_argument('--checkpoint', default='checkpoint_vox.pth', type=str)
    parser.add_argument('--output_dir', default='outputs/q2_5', type=str)
    parser.add_argument('--seed', default=0, type=int, help='same meaning as in q2_viz.py (input view choice)')
    parser.add_argument('--vis_idxs', default=[0, 1, 2], type=int, nargs='+', help='chair test-set indices')
    parser.add_argument('--vox_threshold', default=0.5, type=float, help='occupancy threshold on sigmoid(logits)')
    parser.add_argument('--point_radius', default=0.01, type=float,
                        help='read by q1_viz.render_views (also on its empty-prediction path)')
    # rendering / F1 settings, same defaults as q2_viz.py / eval_model.py
    parser.add_argument('--arch', default='resnet18', type=str)
    parser.add_argument('--n_points', default=1000, type=int)
    parser.add_argument('--image_size', default=256, type=int)
    parser.add_argument('--dist', default=1.4, type=float)
    parser.add_argument('--elev', default=20.0, type=float)
    parser.add_argument('--azim', default=45.0, type=float)
    return parser


# ---------------------------------------------------------------------------- data

def load_examples(dataset, idxs, seed):
    """Items for `idxs` with exactly the input views q2_viz.py shows for the same --seed.

    Replays q2_viz's single `random` stream: one randint per test item, in test-set order.
    Items not requested only consume their draw (no loading).
    """
    random.seed(seed)
    items = {}
    for i in range(max(idxs) + 1):
        if i in idxs:
            views = dataset.views_per_model_list[i]
            state = random.getstate()
            view_idx = views[random.randint(0, len(views) - 1)]
            random.setstate(state)
            item = dataset[i]  # makes the same draw
            item['view_idx'] = view_idx
            items[i] = item
        else:
            random.randint(0, len(dataset.views_per_model_list[i]) - 1)
    return [items[i] for i in idxs]


def occlude(image, where):
    """image: H x W x 3 in [0, 1]. Returns a copy with half the image set to the ImageNet mean.

    Odd sizes (R2N2 images are 137 x 137): each mask covers ceil(size / 2) rows/columns, so the
    centre row/column is hidden by both opposite masks and every mask covers >= 50% of the image.
    """
    out = image.clone()
    H, W = image.shape[:2]
    h, w = math.ceil(H / 2), math.ceil(W / 2)
    region = {'top': (slice(0, h), slice(None)), 'bottom': (slice(H - h, H), slice(None)),
              'left': (slice(None), slice(0, w)), 'right': (slice(None), slice(W - w, W))}[where]
    out[region] = torch.tensor(IMAGENET_MEAN, dtype=image.dtype)
    return out


# ---------------------------------------------------------------------------- model / metrics

def load_model(args):
    if not os.path.isfile(args.checkpoint):
        raise SystemExit(f'Checkpoint not found: {os.path.abspath(args.checkpoint)}')
    ckpt = torch.load(args.checkpoint, map_location=args.device, weights_only=True)
    if not any(k.startswith('encoder.') for k in ckpt['model_state_dict']):
        raise SystemExit(f'{args.checkpoint} has no image-encoder weights (trained with --load_feat); '
                         'occlusion needs the live ResNet18 encoder on RGB input.')
    model = SingleViewto3D(args).to(args.device)
    q2_viz.check_compatibility(model, ckpt['model_state_dict'], args)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    print(f'Loaded {args.checkpoint} (step {ckpt["step"]})')
    return model, ckpt['step']


def occupancy(logits, args):
    return torch.sigmoid(logits) > args.vox_threshold


def voxel_iou(a, b):
    union = (a | b).sum().item()
    return (a & b).sum().item() / union if union else 1.0  # both empty: identical


# ---------------------------------------------------------------------------- figures

def draw_example(axes, row, args):
    for c, cond in enumerate(CONDITIONS):
        r = row['conds'][cond]
        axes[0][c].imshow(r['rgb'])
        axes[0][c].set_title('Original RGB' if cond == 'original' else f'{cond.capitalize()} half masked', fontsize=10)
        axes[1][c].imshow(r['render'])
        axes[1][c].set_title('Prediction' + (' (reference)' if cond == 'original' else ''), fontsize=10)
        sub = 'IoU vs original = 1 (reference)' if cond == 'original' else f'IoU vs original = {r["iou"]:.3f}'
        axes[1][c].text(0.5, 0.02, f'{sub}\noccupied voxels = {r["n_occ"]}\nF1@{KEY_T} vs GT = {r["f1"]:.1f}',
                        transform=axes[1][c].transAxes, ha='center', va='bottom', fontsize=9)
        for ax in (axes[0][c], axes[1][c]):
            ax.axis('off')


def save_figures(rows, args):
    for n, row in enumerate(rows):
        fig, axes = plt.subplots(2, 5, figsize=(15, 6.4))
        draw_example(axes, row, args)
        fig.suptitle(f'Occlusion sensitivity, chair test idx {row["idx"]} (view {row["view_idx"]}): '
                     'input (top row) -> predicted voxels (bottom row)', y=1.02)
        fig.tight_layout()
        fig.savefig(os.path.join(args.output_dir, f'example_{n}.png'), dpi=110, bbox_inches='tight')
        plt.close(fig)

    fig, axes = plt.subplots(2 * len(rows), 5, figsize=(15, 6.2 * len(rows)), squeeze=False)
    for n, row in enumerate(rows):
        draw_example(axes[2 * n:2 * n + 2], row, args)
        axes[2 * n][0].text(-0.12, 0.0, f'test idx {row["idx"]}', transform=axes[2 * n][0].transAxes,
                            rotation=90, va='center', ha='right', fontsize=12)
    fig.suptitle('Q2.5 occlusion sensitivity of the voxel model: masked input -> prediction', y=1.005)
    fig.tight_layout()
    fig.savefig(os.path.join(args.output_dir, 'examples.png'), dpi=100, bbox_inches='tight')
    plt.close(fig)


def write_metrics(rows, step, args):
    lines = [
        'Q2.5 occlusion sensitivity of the Part 2 voxel model (inference only, no training)',
        f'checkpoint: {os.path.abspath(args.checkpoint)} (step {step})',
        'model: model.SingleViewto3D type=vox (ImageNet ResNet18 encoder on RGB, load_feat=False)',
        f'test split: {dataset_location.SPLITS_PATH} (chair test)',
        f'test indices: {args.vis_idxs}; input views (R2N2 view id): {[r["view_idx"] for r in rows]} '
        f'(same as q2_viz.py with --seed {args.seed})',
        f'mask: one half of the {rows[0]["size"][0]}x{rows[0]["size"][1]} image '
        f'({math.ceil(rows[0]["size"][0] / 2)} rows / columns), filled with ImageNet mean RGB '
        f'{IMAGENET_MEAN} in [0, 1] (exactly 0 after the model\'s normalization)',
        f'occupancy: sigmoid(logits) > {args.vox_threshold} on the 32^3 grid',
        'IoU vs original: |occ_masked AND occ_orig| / |occ_masked OR occ_orig| '
        '(1 = unchanged; lower = hiding that half changed the prediction more)',
        f'F1@{KEY_T} vs GT: q2_viz.evaluate_sample (Part 2 voxel convention), secondary, for context only',
        '',
        f'{"test idx":>8} {"condition":>9} {"IoU vs orig":>12} {"occupied":>9} {"delta occ":>10} {"F1@0.05":>8}',
    ]
    for row in rows:
        n0 = row['conds']['original']['n_occ']
        for cond in CONDITIONS:
            r = row['conds'][cond]
            lines.append(f'{row["idx"]:>8} {cond:>9} {r["iou"]:>12.4f} {r["n_occ"]:>9d} {r["n_occ"] - n0:>+10d} {r["f1"]:>8.2f}')
        lines.append('')
    lines += [f'mean over the {len(rows)} examples', f'{"mask":>9} {"IoU vs orig":>12} {"|delta occ|":>12} {"F1@0.05":>8}']
    for cond in CONDITIONS:
        iou = np.mean([row['conds'][cond]['iou'] for row in rows])
        dabs = np.mean([abs(row['conds'][cond]['n_occ'] - row['conds']['original']['n_occ']) for row in rows])
        f1 = np.mean([row['conds'][cond]['f1'] for row in rows])
        lines.append(f'{cond:>9} {iou:>12.4f} {dabs:>12.1f} {f1:>8.2f}')
    ranked = sorted(MASKS, key=lambda c: np.mean([row['conds'][c]['iou'] for row in rows]))
    lines += ['', 'masks ordered by mean IoU vs original (largest change first): ' + ', '.join(ranked),
              'sanity check: original-vs-original IoU = '
              + ', '.join(f'{row["self_iou"]:.1f}' for row in rows),
              '',
              'Scope: this measures sensitivity of these predictions to hiding a large image half for the',
              f'{len(rows)} examples above. It does not establish which image region is semantically important in general.']
    text = '\n'.join(lines) + '\n'
    with open(os.path.join(args.output_dir, 'metrics.txt'), 'w') as f:
        f.write(text)
    print('\n' + text)


# ---------------------------------------------------------------------------- main

def main(args):
    args.type, args.load_feat = 'vox', False  # Part 2 voxel model on live RGB (SingleViewto3D / q2_viz)
    os.makedirs(args.output_dir, exist_ok=True)
    model, step = load_model(args)
    dataset = R2N2('test', dataset_location.SHAPENET_PATH, dataset_location.R2N2_PATH,
                   dataset_location.SPLITS_PATH, return_voxels=True, return_feats=False)
    if max(args.vis_idxs) >= len(dataset):
        raise SystemExit(f'--vis_idxs must be < {len(dataset)} (chair test split size)')

    rows = []
    with torch.no_grad():
        for idx, item in zip(args.vis_idxs, load_examples(dataset, args.vis_idxs, args.seed)):
            image = item['images']  # H x W x 3 in [0, 1], the single view q2_viz shows
            batch = torch.stack([image] + [occlude(image, m) for m in MASKS]).to(args.device)  # 5 x H x W x 3
            # one forward per condition (batch of 1, as in q2_viz): batching changes GPU conv numerics
            logits = torch.cat([model(x[None], args) for x in batch])  # 5 x 1 x 32 x 32 x 32 raw logits
            occ = occupancy(logits, args)
            mesh_gt = Meshes([item['verts']], [item['faces']])
            conds = {}
            for c, cond in enumerate(CONDITIONS):
                torch.manual_seed(args.seed * 1_000_003 + idx)  # same F1 sampling seed for all conditions
                metrics, _ = q2_viz.evaluate_sample(logits[c:c + 1], mesh_gt, args)
                conds[cond] = {'rgb': batch[c].cpu().numpy().clip(0, 1),
                               'render': q2_viz.render(q2_viz.pred_renderable(logits[c:c + 1], args), args),
                               'iou': voxel_iou(occ[c], occ[0]),
                               'n_occ': int(occ[c].sum().item()),
                               'f1': metrics[f'F1@{KEY_T:f}'].item()}
            rows.append({'idx': idx, 'view_idx': item['view_idx'], 'model_id': item['model_id'],
                         'size': tuple(image.shape[:2]), 'conds': conds, 'self_iou': voxel_iou(occ[0], occ[0])})
            print(f'test idx {idx} (view {item["view_idx"]}): IoU vs original ' +
                  ', '.join(f'{m} {conds[m]["iou"]:.3f}' for m in MASKS))
            del logits, batch

    save_figures(rows, args)
    write_metrics(rows, step, args)
    print(f'Saved outputs to {args.output_dir}/')


if __name__ == '__main__':
    main(get_args_parser().parse_args())
