# Q2 segmentation results

Test point accuracy: **0.9004** over 617 objects x 10000 points, checkpoint `./checkpoints/seg/best_model.pt` (epoch 248).

Colors (same for ground truth and prediction): part 0 = white, part 1 = magenta, part 2 = cyan, part 3 = yellow, part 4 = blue, part 5 = red

| kind | object | accuracy | ground truth | prediction |
|---|---|---|---|---|
| random | 50 | 0.9095 | [gifs/random_050_gt.gif](gifs/random_050_gt.gif) | [gifs/random_050_pred.gif](gifs/random_050_pred.gif) |
| random | 177 | 0.9188 | [gifs/random_177_gt.gif](gifs/random_177_gt.gif) | [gifs/random_177_pred.gif](gifs/random_177_pred.gif) |
| random | 602 | 0.9431 | [gifs/random_602_gt.gif](gifs/random_602_gt.gif) | [gifs/random_602_pred.gif](gifs/random_602_pred.gif) |
| poor | 351 | 0.4620 | [gifs/poor_351_gt.gif](gifs/poor_351_gt.gif) | [gifs/poor_351_pred.gif](gifs/poor_351_pred.gif) |
| poor | 26 | 0.4984 | [gifs/poor_026_gt.gif](gifs/poor_026_gt.gif) | [gifs/poor_026_pred.gif](gifs/poor_026_pred.gif) |
