# Q1 classification results

Test accuracy: **0.9675** (922/953 objects), checkpoint `./checkpoints/cls/best_model.pt` (epoch 109).

| class | objects | misclassified | accuracy |
|---|---|---|---|
| chair | 617 | 0 | 1.0000 |
| vase | 102 | 12 | 0.8824 |
| lamp | 234 | 19 | 0.9188 |

| kind | object | ground truth | prediction (prob) | gif |
|---|---|---|---|---|
| random | 288 | chair | chair (1.000) | [gifs/random_288_gt-chair_pred-chair.gif](gifs/random_288_gt-chair_pred-chair.gif) |
| random | 495 | chair | chair (1.000) | [gifs/random_495_gt-chair_pred-chair.gif](gifs/random_495_gt-chair_pred-chair.gif) |
| random | 530 | chair | chair (1.000) | [gifs/random_530_gt-chair_pred-chair.gif](gifs/random_530_gt-chair_pred-chair.gif) |
| random | 845 | lamp | lamp (1.000) | [gifs/random_845_gt-lamp_pred-lamp.gif](gifs/random_845_gt-lamp_pred-lamp.gif) |
| random | 944 | lamp | vase (0.796) | [gifs/random_944_gt-lamp_pred-vase.gif](gifs/random_944_gt-lamp_pred-vase.gif) |
| failure | 685 | vase | lamp (1.000) | [gifs/failure_685_gt-vase_pred-lamp.gif](gifs/failure_685_gt-vase_pred-lamp.gif) |
| failure | 827 | lamp | vase (1.000) | [gifs/failure_827_gt-lamp_pred-vase.gif](gifs/failure_827_gt-lamp_pred-vase.gif) |

No misclassified test objects for class 'chair'.
