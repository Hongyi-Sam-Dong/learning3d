# Assignment 5

This project implements PointNet classification and segmentation on point clouds, robustness experiments for both models, and a compact Point Transformer that adds locality.

## Setup

Enter the existing Docker environment from the host, then change to the assignment folder inside the container:

```bash
/home/aims/Desktop/samd/learning3d/bring_up.bash
cd /workspace/learning3d/assignment5
```

All paths below are relative to this folder. Files written here persist on the host.

Required data:

- `data/cls/` and `data/seg/`, each with `data_train.npy`, `label_train.npy`, `data_test.npy` and `label_test.npy`.

If the data is missing, download it from Hugging Face. The existing environment does not include the Hugging Face CLI (`hf`), so install it first with `pip install huggingface_hub`.

```bash
hf download learning3dvision/assignment5 a5_data.zip --repo-type dataset --local-dir data_download
unzip data_download/a5_data.zip -d ./
```

Required checkpoints:

- PointNet (Q1–Q3): `checkpoints/cls/best_model.pt` and `checkpoints/seg/best_model.pt`
- Point Transformer (Q4): `checkpoints/q4/cls_n10000_bs32/best_model.pt` and `checkpoints/q4/seg_n10000_bs32/best_model.pt`

## Reproducing the results

Evaluate the PointNet checkpoints for Q1–Q3:

```bash
python main.py inference --stage all --task all
```

Then evaluate the Point Transformer (Q4) and compare it with the Q1/Q2 results produced above:

```bash
python main.py q4 inference --task all --tag bs32
```

Run the Q1–Q3 command first, because Q4 reads its baseline results from `output/q1_cls/` and `output/q2_seg/`.

Results are written to `output/`:

| Folder | Content |
|---|---|
| `output/q1_cls/` | Q1 classification accuracy, per-object predictions, GIFs and overview image |
| `output/q2_seg/` | Q2 segmentation accuracy, per-object accuracy, GIFs and overview image |
| `output/q3/cls/`, `output/q3/seg/` | Q3 rotation and point-count robustness: metrics, accuracy plots and example images |
| `output/q4/cls_n10000_bs32/`, `output/q4/seg_n10000_bs32/` | Q4 metrics, comparison with Q1/Q2, GIFs and overview image |

Each folder has a `metrics.json` (or `rotation.json` and `num_points.json` for Q3) with the numbers and a `summary.md` that links the visualizations.

## Training from scratch (optional)

Training is not needed when the checkpoints above are available. Retrained models may not reproduce the reported numbers exactly.

PointNet (Q1/Q2), 10,000 points, batch size 32:

```bash
python main.py train --task all
```

Point Transformer (Q4), with the reported settings of 10,000 points, batch size 32 and tag `bs32`:

```bash
python main.py q4 train --task all --batch_size 32 --tag bs32
```

Both commands refuse to overwrite existing checkpoints. Add `--overwrite` to replace them, or move the existing checkpoints first.
