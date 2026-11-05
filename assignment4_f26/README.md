# 16-825 Assignment 4: Multi-View Transformers

This folder contains my completed code for Assignment 4: multi-view data preprocessing (Q1), novel-view synthesis with Mini-LVSM (Q2) and camera/depth estimation with Mini-VGGT (Q3). `main.py` runs every experiment in my report through the existing training and evaluation entry points.

Run every command from this folder, the project root (for example `<andrewid>_code_proj4/`). Config and data paths are relative to it.

## Contents

| Path | Contents |
| --- | --- |
| `main.py` | Launcher for every stage below |
| `a4data/` | Q1 geometry (`a4data/geometry.py`), dataset and loaders |
| `models/lvsm/` | Q2 Mini-LVSM model, loss, training and evaluation |
| `models/vggt/` | Q3 Mini-VGGT model, heads, losses, training and evaluation |
| `scripts/` | Q1 viewer (`visualize_data.py`) and checker (`check_data.py`) |
| `configs/` | Data config, scene splits and the four model configs |
| `docs/` | Handouts and the [data contract](docs/DATA_CONTRACT.md) |

## Setup

Python 3.10 or newer is required, and an NVIDIA GPU with bfloat16 support is strongly recommended. Install a PyTorch build for your machine (see [pytorch.org](https://pytorch.org/get-started/locally/)), then install the assignment dependencies:

```bash
pip install -r requirements.txt
pip install -e . --no-deps   # optional when using main.py
```

`main.py` puts this folder first on `PYTHONPATH` for every job, so it always imports `a4data` and `models` from this copy. The editable install is only needed to run `scripts/*.py` directly.

I ran the experiments with Python 3.10, torch 2.10.0 (CUDA 13.0), torchvision 0.25.0, numpy 1.26, opencv-python-headless 4.11 and plotly 6.9 on one NVIDIA DGX Spark (GB10).

### Dataset

We use prepared iPhone RGB/depth frames from [ScanNet++](https://github.com/scannetpp/scannetpp). The supplied subset contains 280 indoor scenes, split by scene into 224 training, 28 validation and 28 test scenes. Training RGB images and aligned depth maps are 400×300; depth maps are 16-bit PNGs in millimeters, with zero values indicating missing depth.

Download and extract [ScanNetpp_A4.zip](https://huggingface.co/datasets/learning3dvision/ScanNetpp_iphone/blob/main/ScanNetpp_A4.zip) into `data/scannetpp/` (31.9 GB compressed, 40.7 GB extracted):

```bash
pip install huggingface_hub
hf download learning3dvision/ScanNetpp_iphone ScanNetpp_A4.zip --repo-type dataset --local-dir data/scannetpp
unzip data/scannetpp/ScanNetpp_A4.zip -d data/scannetpp
```

The result should be `data/scannetpp/processed/` (training scenes, preprocessed online by my Q1 code) and `data/scannetpp/prepared_eval/` (validation and test frames, 128 per sequence). If the data already exists elsewhere, a symlink works: `ln -s /path/to/scannetpp data/scannetpp`.

### Pretrained DINOv2

Mini-VGGT uses a frozen DINOv2 ViT-B/14 encoder. Nothing has to be downloaded by hand: the first Q3 run calls `torch.hub.load("facebookresearch/dinov2:7764ea0f…", "dinov2_vitb14")`, which fetches the DINOv2 code from GitHub and the official `dinov2_vitb14_pretrain.pth` weights into `$TORCH_HOME/hub` (default `~/.cache/torch/hub`). This needs internet access once. Evaluation loads the encoder weights from the Mini-VGGT checkpoint itself, but still needs the cached DINOv2 code.

### Optional: my Docker environment

I worked in a Docker image built for the DGX Spark (aarch64, CUDA 13). Its `Dockerfile`, `compose.yaml` and `bring_up.bash` live in my parent course folder and are **not** part of this submission. Inside that container the project is mounted at `/workspace/learning3d/assignment4_f26`, and a named volume keeps the DINOv2 cache. This is not needed to grade the code: the pip setup above uses only the files in this folder.

## Quick check (Q1)

With the dataset in place, this runs the Q1 viewers and `check_data.py` on four training samples. It needs no GPU and took about 10 seconds on my machine:

```bash
python main.py --stage q1
```

It writes `outputs/reproduce/unprojection/index.html`, `outputs/reproduce/canonicalization/index.html` and `outputs/reproduce/check_data/train.json`. Open the HTML files in a browser. Expect reprojection errors far below 0.01 px and `centroid_distance` ≈ 1.

## Running the experiments

`python main.py` (or `--help`) lists the stages and exits. `python main.py --stage <name>` runs one stage, and `--dry-run` prints its commands without running them.

| Stage | Question | What it runs | Results in `<output root>/` |
| --- | --- | --- | --- |
| `q1` | Q1 | Unprojection and canonicalization viewers (5 scenes), `check_data.py` on 4 training samples | `unprojection/`, `canonicalization/`, `check_data/train.json` |
| `q2.1` | Q2.1 | Mini-LVSM overfit on one fixed batch, 2,000 steps | `lvsm_overfit/` |
| `q2.2-train` | Q2.2 | Mini-LVSM baseline, 100 epochs, `configs/lvsm.json` | `lvsm/` |
| `q2.2-eval` | Q2.2 | Full validation of `lvsm/best.pt` for S = 1, 2, 4 × correct/shuffled/none rays | `lvsm/eval_val/` |
| `q2.3-train` | Q2.3 | MSE + 0.1 × L1, 100 epochs, `configs/lvsm_mse_l1.json` | `lvsm_mse_l1/` |
| `q2.3-eval` | Q2.3 | Same evaluation for `lvsm_mse_l1/best.pt` | `lvsm_mse_l1/eval_val/` |
| `q3.1` | Q3.1 | Mini-VGGT single-clip overfit, 40 epochs × 25 steps | `vggt_overfit/` |
| `q3.2-train` | Q3.2 | Joint camera + depth training, 1,000 epochs, `configs/vggt.json` | `vggt_train/` |
| `q3.2-eval` | Q3.2 | Full val and test evaluation of `vggt_train/best.pt` | `vggt_train/eval_val/`, `vggt_train/eval_test/` |
| `q3.3-train` | Q3.3 | Depth-only training, 1,000 epochs, `configs/vggt_depth_only.json` | `vggt_depth_only/` |
| `q3.3-eval` | Q3.3 | Full val of joint `last.pt` vs depth-only `last.pt` | `vggt_synergy_eval/` |

Q2.0 and Q3.0 are the model implementations; the stages above exercise them and need no separate jobs. All Q3 evaluations use the official settings: 8 views from the first 32 prepared frames, 224 px, seed 825, on the full split. `q3.3-eval` also prints `depth_abs_rel` and `depth_delta1` for both runs and saves them to `vggt_synergy_eval/summary.json`.

To reproduce everything in order (Q1 → Q3.3):

```bash
python main.py --stage all
```

On one GB10 GPU, each Mini-LVSM training run took about 1 hour and each 1,000-epoch Mini-VGGT run took about 6 hours.

### Evaluating existing checkpoints

The evaluation stages read checkpoints from `--checkpoint-root` (default: the output root) and always write their reports under `--output-root`. With checkpoints laid out as `<dir>/lvsm/best.pt`, `<dir>/lvsm_mse_l1/best.pt`, `<dir>/vggt_train/{best,last}.pt` and `<dir>/vggt_depth_only/last.pt`:

```bash
python main.py --stage q3.2-eval --checkpoint-root <dir> --output-root outputs/eval_check
```

The checkpoints are not in the Canvas ZIP because of their size (about 170 MB per Mini-LVSM run and 1 GB per Mini-VGGT run).

### Checkpoint choice for Q3.2 and Q3.3

- **Q3.2 uses `best.pt`**, the epoch with the lowest validation loss (epoch 721 of 1,000 in my run), as the handout specifies.
- **Q3.3 uses `last.pt`** from both runs, the epoch-1,000 weights. The joint and depth-only runs have different validation losses (the depth-only loss has no camera term), so selecting each run's `best.pt` by its own loss would make the checkpoint choice favor one objective. Comparing the final checkpoints after identical schedules keeps the comparison controlled.

## Outputs and safety

- New runs write to `outputs/reproduce/` by default; change it with `--output-root DIR`.
- My completed experiments live directly in `outputs/` (`outputs/lvsm/`, `outputs/vggt_train/`, …). Nothing writes there by default.
- Before starting anything, `main.py` checks every destination of the selected stage(s). If a run directory or report already exists, it stops with an error instead of appending to `log.jsonl`/`history.jsonl` or overwriting results. Pick a new `--output-root` or remove the old directory yourself.
- An evaluation stage with a missing checkpoint fails before launching. If any job fails, `main.py` stops and exits with that job's exit code.
- Training runs write `best.pt`, `last.pt`, their logs, loss curves and visualizations into their run directory. Mini-VGGT training also writes `visualization/index.html`, a GT/prediction viewer of four validation scenes; `q3.2-eval` writes a separate viewer to `vggt_train/eval_val/index.html`.

## Assignment parts

1. **[Multi-View Data Preprocessing](docs/part1_data.md) (20 points)**: canonicalized cameras, depth, points and ray embeddings.
2. **[Novel-View Synthesis with Mini-LVSM](docs/part2_lvsm.md) (40 points)**: predict target RGB from source images and camera rays.
3. **[Multi-View Geometry with Mini-VGGT](docs/part3_vggt.md) (40 points)**: predict cameras and depth from RGB views.

The handouts list the original per-step commands; `main.py` runs the same entry points with my settings. The results, figures and discussion are on my assignment webpage.
