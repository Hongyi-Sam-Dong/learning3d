# Q4 classification (cls_n10000_bs32)

Test accuracy: **0.9790** (933/953), checkpoint `./checkpoints/q4/cls_n10000_bs32/best_model.pt` (epoch 214, val acc 0.9867256637168141), 10000 points per object.

| model | test accuracy |
|---|---|
| Q1 PointNet | 0.9675 |
| Q4 Point Transformer | 0.9790 |

Same 953 test objects. Both correct 909, only Q4 correct 24, only Q1 correct 13, both wrong 7. Point indices: identical: both use all 10000 points.

Checkpoint identity (baseline): consistent: current file epoch/val_acc equal the values recorded by the baseline evaluation
Seed match: True  |  points per object: baseline 10000, Q4 10000
Protocol difference: inference batch size 16 vs 8 (no effect on predictions in eval mode)

| class | objects | misclassified | accuracy |
|---|---|---|---|
| chair | 617 | 2 | 0.9968 |
| vase | 102 | 9 | 0.9118 |
| lamp | 234 | 9 | 0.9615 |

| kind | object | gt | Q1 | Q4 (prob) | gif |
|---|---|---|---|---|---|
| random | 288 | chair | chair | chair (1.000) | [gifs/random_288_gt-chair_q4-chair.gif](gifs/random_288_gt-chair_q4-chair.gif) |
| random | 495 | chair | chair | chair (1.000) | [gifs/random_495_gt-chair_q4-chair.gif](gifs/random_495_gt-chair_q4-chair.gif) |
| random | 530 | chair | chair | chair (1.000) | [gifs/random_530_gt-chair_q4-chair.gif](gifs/random_530_gt-chair_q4-chair.gif) |
| random | 845 | lamp | lamp | lamp (1.000) | [gifs/random_845_gt-lamp_q4-lamp.gif](gifs/random_845_gt-lamp_q4-lamp.gif) |
| random | 944 | lamp | vase | lamp (1.000) | [gifs/random_944_gt-lamp_q4-lamp.gif](gifs/random_944_gt-lamp_q4-lamp.gif) |
| q4_fixes_q1 | 812 | lamp | vase | lamp (1.000) | [gifs/q4_fixes_q1_812_gt-lamp_q4-lamp.gif](gifs/q4_fixes_q1_812_gt-lamp_q4-lamp.gif) |
| q4_fixes_q1 | 770 | lamp | vase | lamp (1.000) | [gifs/q4_fixes_q1_770_gt-lamp_q4-lamp.gif](gifs/q4_fixes_q1_770_gt-lamp_q4-lamp.gif) |
| q4_breaks_q1 | 620 | vase | vase | lamp (1.000) | [gifs/q4_breaks_q1_620_gt-vase_q4-lamp.gif](gifs/q4_breaks_q1_620_gt-vase_q4-lamp.gif) |
| q4_breaks_q1 | 664 | vase | vase | lamp (1.000) | [gifs/q4_breaks_q1_664_gt-vase_q4-lamp.gif](gifs/q4_breaks_q1_664_gt-vase_q4-lamp.gif) |
| q4_failure | 445 | chair | chair | vase (0.506) | [gifs/q4_failure_445_gt-chair_q4-vase.gif](gifs/q4_failure_445_gt-chair_q4-vase.gif) |
| q4_failure | 620 | vase | vase | lamp (1.000) | [gifs/q4_failure_620_gt-vase_q4-lamp.gif](gifs/q4_failure_620_gt-vase_q4-lamp.gif) |
| q4_failure | 827 | lamp | vase | vase (1.000) | [gifs/q4_failure_827_gt-lamp_q4-vase.gif](gifs/q4_failure_827_gt-lamp_q4-vase.gif) |
