# Q4 segmentation (seg_n10000_bs32)

Test point accuracy: **0.9276** (subset: one 10000-point input per object; accuracy over those points), checkpoint `./checkpoints/q4/seg_n10000_bs32/best_model.pt` (epoch 155, val acc 0.9243358574610245).

| model | point accuracy | mean object accuracy |
|---|---|---|
| Q2 PointNet | 0.9004 | 0.9004 |
| Q4 Point Transformer | 0.9276 | 0.9276 |

Point indices: verified: Q4 indices equal the first 10000 of the 10000 stored Q2 indices per object. identical point sets and inputs. Objects better / worse / equal under Q4: 504 / 113 / 0.

Checkpoint identity (baseline): consistent: current file epoch/val_acc equal the values recorded by the baseline evaluation
Seed match: True  |  points per object: baseline 10000, Q4 10000
Protocol difference: inference batch size 16 vs 8 (no effect on predictions in eval mode)

Colors: part 0 = white, part 1 = magenta, part 2 = cyan, part 3 = yellow, part 4 = blue, part 5 = red

| kind | object | Q2 acc | Q4 acc | ground truth | Q4 prediction |
|---|---|---|---|---|---|
| q2_selection | 50 | 0.9095 | 0.9434 | [gifs/q2_selection_050_gt.gif](gifs/q2_selection_050_gt.gif) | [gifs/q2_selection_050_q4.gif](gifs/q2_selection_050_q4.gif) |
| q2_selection | 177 | 0.9188 | 0.9312 | [gifs/q2_selection_177_gt.gif](gifs/q2_selection_177_gt.gif) | [gifs/q2_selection_177_q4.gif](gifs/q2_selection_177_q4.gif) |
| q2_selection | 602 | 0.9431 | 0.9703 | [gifs/q2_selection_602_gt.gif](gifs/q2_selection_602_gt.gif) | [gifs/q2_selection_602_q4.gif](gifs/q2_selection_602_q4.gif) |
| q2_selection | 351 | 0.4620 | 0.5057 | [gifs/q2_selection_351_gt.gif](gifs/q2_selection_351_gt.gif) | [gifs/q2_selection_351_q4.gif](gifs/q2_selection_351_q4.gif) |
| q2_selection | 26 | 0.4984 | 0.6815 | [gifs/q2_selection_026_gt.gif](gifs/q2_selection_026_gt.gif) | [gifs/q2_selection_026_q4.gif](gifs/q2_selection_026_q4.gif) |
| q4_better | 60 | 0.6975 | 0.9624 | [gifs/q4_better_060_gt.gif](gifs/q4_better_060_gt.gif) | [gifs/q4_better_060_q4.gif](gifs/q4_better_060_q4.gif) |
| q4_better | 543 | 0.6159 | 0.8482 | [gifs/q4_better_543_gt.gif](gifs/q4_better_543_gt.gif) | [gifs/q4_better_543_q4.gif](gifs/q4_better_543_q4.gif) |
| q4_worse | 330 | 0.8794 | 0.6228 | [gifs/q4_worse_330_gt.gif](gifs/q4_worse_330_gt.gif) | [gifs/q4_worse_330_q4.gif](gifs/q4_worse_330_q4.gif) |
| q4_worse | 97 | 0.7686 | 0.5402 | [gifs/q4_worse_097_gt.gif](gifs/q4_worse_097_gt.gif) | [gifs/q4_worse_097_q4.gif](gifs/q4_worse_097_q4.gif) |
