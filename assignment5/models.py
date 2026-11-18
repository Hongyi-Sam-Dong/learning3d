import torch
import torch.nn as nn
import torch.nn.functional as F


def shared_mlp(channels):
    layers = []
    for c_in, c_out in zip(channels[:-1], channels[1:]):
        layers += [nn.Conv1d(c_in, c_out, 1), nn.BatchNorm1d(c_out), nn.ReLU(inplace=True)]
    return nn.Sequential(*layers)


class PointFeatures(nn.Module):
    def __init__(self):
        super(PointFeatures, self).__init__()
        self.local_mlp = shared_mlp([3, 64, 64])
        self.global_mlp = shared_mlp([64, 64, 128, 1024])

    def forward(self, points):
        local_features = self.local_mlp(points.transpose(1, 2))
        global_feature = self.global_mlp(local_features).max(dim=2)[0]
        return local_features, global_feature


# ------ TO DO ------
class cls_model(nn.Module):
    def __init__(self, num_classes=3):
        super(cls_model, self).__init__()
        self.features = PointFeatures()
        self.head = nn.Sequential(
            nn.Linear(1024, 512), nn.BatchNorm1d(512), nn.ReLU(inplace=True),
            nn.Linear(512, 256), nn.BatchNorm1d(256), nn.ReLU(inplace=True),
            nn.Dropout(p=0.3),
            nn.Linear(256, num_classes),
        )

    def forward(self, points):
        '''
        points: tensor of size (B, N, 3)
                , where B is batch size and N is the number of points per object (N=10000 by default)
        output: tensor of size (B, num_classes)
        '''
        return self.head(self.features(points)[1])



# ------ TO DO ------
class seg_model(nn.Module):
    def __init__(self, num_seg_classes = 6):
        super(seg_model, self).__init__()
        self.features = PointFeatures()
        self.head = nn.Sequential(shared_mlp([1088, 512, 256, 128]), nn.Conv1d(128, num_seg_classes, 1))

    def forward(self, points):
        '''
        points: tensor of size (B, N, 3)
                , where B is batch size and N is the number of points per object (N=10000 by default)
        output: tensor of size (B, N, num_seg_classes)
        '''
        local_features, global_feature = self.features(points)
        global_features = global_feature.unsqueeze(2).expand(-1, -1, local_features.shape[2])
        return self.head(torch.cat([local_features, global_features], dim=1)).transpose(1, 2)



