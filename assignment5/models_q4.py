import torch
import torch.nn as nn
from pytorch3d.ops import knn_gather, knn_points, sample_farthest_points

ARCH = "point_transformer_compact_v1"
DIMS = (32, 64, 128, 256, 512)
STRIDE = 4
CLS_DROPOUT = 0.5


class BN(nn.BatchNorm1d):
    def forward(self, x):
        return super().forward(x.reshape(-1, x.shape[-1])).reshape(x.shape)


def linear_bn_relu(c_in, c_out):
    return nn.Sequential(nn.Linear(c_in, c_out), BN(c_out), nn.ReLU(inplace=True))


@torch.no_grad()
def self_knn(xyz, k):
    return knn_points(xyz, xyz, K=min(k, xyz.shape[1]), return_sorted=False).idx


@torch.no_grad()
def interpolation_weights(fine_xyz, coarse_xyz, eps=1e-8):
    dist2, idx, _ = knn_points(fine_xyz, coarse_xyz, K=min(3, coarse_xyz.shape[1]))
    exact = dist2 <= eps * eps
    weights = 1.0 / dist2.sqrt().clamp_min(eps)
    weights = torch.where(exact.any(dim=2, keepdim=True), exact.to(weights.dtype), weights)
    return idx, weights / weights.sum(dim=2, keepdim=True)


class PointTransformerLayer(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.to_q = nn.Linear(dim, dim)
        self.to_k = nn.Linear(dim, dim)
        self.to_v = nn.Linear(dim, dim)
        self.pos = nn.Sequential(nn.Linear(3, 3), BN(3), nn.ReLU(inplace=True), nn.Linear(3, dim))
        self.attn = nn.Sequential(nn.Linear(dim, dim), BN(dim), nn.ReLU(inplace=True), nn.Linear(dim, dim))

    def forward(self, xyz, feats, idx):
        delta = self.pos(xyz.unsqueeze(2) - knn_gather(xyz, idx))
        keys = knn_gather(self.to_k(feats), idx)
        values = knn_gather(self.to_v(feats), idx)
        weights = torch.softmax(self.attn(self.to_q(feats).unsqueeze(2) - keys + delta), dim=2)
        return (weights * (values + delta)).sum(dim=2)


class PointTransformerBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.pre = linear_bn_relu(dim, dim)
        self.layer = PointTransformerLayer(dim)
        self.mid = nn.Sequential(BN(dim), nn.ReLU(inplace=True))
        self.post = nn.Sequential(nn.Linear(dim, dim), BN(dim))
        self.act = nn.ReLU(inplace=True)

    def forward(self, xyz, feats, idx):
        return self.act(feats + self.post(self.mid(self.layer(xyz, self.pre(feats), idx))))


class TransitionDown(nn.Module):
    def __init__(self, c_in, c_out, k, stride=STRIDE):
        super().__init__()
        self.k = k
        self.stride = stride
        self.mlp = linear_bn_relu(c_in + 3, c_out)

    def forward(self, xyz, feats):
        with torch.no_grad():
            centers = sample_farthest_points(xyz, K=max(1, xyz.shape[1] // self.stride))[0]
            idx = knn_points(centers, xyz, K=min(self.k, xyz.shape[1]), return_sorted=False).idx
        grouped = torch.cat([knn_gather(xyz, idx) - centers.unsqueeze(2), knn_gather(feats, idx)], dim=-1)
        return centers, self.mlp(grouped).max(dim=2)[0]


class TransitionUp(nn.Module):
    def __init__(self, c_coarse, c_fine):
        super().__init__()
        self.coarse = linear_bn_relu(c_coarse, c_fine)
        self.skip = nn.Linear(c_fine, c_fine)

    def forward(self, coarse_xyz, coarse_feats, fine_xyz, fine_feats):
        idx, weights = interpolation_weights(fine_xyz, coarse_xyz)
        interpolated = (knn_gather(self.coarse(coarse_feats), idx) * weights.unsqueeze(-1)).sum(dim=2)
        return interpolated + self.skip(fine_feats)


class PointTransformerEncoder(nn.Module):
    def __init__(self, k=16):
        super().__init__()
        self.k = k
        self.stem = linear_bn_relu(3, DIMS[0])
        self.blocks = nn.ModuleList(PointTransformerBlock(d) for d in DIMS)
        self.downs = nn.ModuleList(TransitionDown(a, b, k) for a, b in zip(DIMS[:-1], DIMS[1:]))

    def forward(self, xyz):
        idx = self_knn(xyz, self.k)
        feats = self.blocks[0](xyz, self.stem(xyz), idx)
        levels = [(xyz, feats, idx)]
        for down, block in zip(self.downs, self.blocks[1:]):
            xyz, feats = down(xyz, feats)
            idx = self_knn(xyz, self.k)
            feats = block(xyz, feats, idx)
            levels.append((xyz, feats, idx))
        return levels


def model_config(task, num_classes, k):
    config = dict(arch=ARCH, task=task, num_classes=num_classes, k=k, dims=list(DIMS), stride=STRIDE)
    if task == "cls":
        config["dropout"] = CLS_DROPOUT
    return config


class PointTransformerCls(nn.Module):
    def __init__(self, num_classes=3, k=16):
        super().__init__()
        self.config = model_config("cls", num_classes, k)
        self.encoder = PointTransformerEncoder(k)
        self.head = nn.Sequential(
            nn.Linear(DIMS[-1], 256), nn.BatchNorm1d(256), nn.ReLU(inplace=True), nn.Dropout(CLS_DROPOUT),
            nn.Linear(256, 128), nn.BatchNorm1d(128), nn.ReLU(inplace=True), nn.Dropout(CLS_DROPOUT),
            nn.Linear(128, num_classes),
        )

    def forward(self, points):
        return self.head(self.encoder(points.float().contiguous())[-1][1].mean(dim=1))


class PointTransformerSeg(nn.Module):
    def __init__(self, num_classes=6, k=16):
        super().__init__()
        self.config = model_config("seg", num_classes, k)
        self.encoder = PointTransformerEncoder(k)
        self.ups = nn.ModuleList(TransitionUp(DIMS[i + 1], DIMS[i]) for i in reversed(range(len(DIMS) - 1)))
        self.blocks = nn.ModuleList(PointTransformerBlock(DIMS[i]) for i in reversed(range(len(DIMS) - 1)))
        self.head = nn.Sequential(linear_bn_relu(DIMS[0], DIMS[0]), nn.Linear(DIMS[0], num_classes))

    def forward(self, points):
        levels = self.encoder(points.float().contiguous())
        xyz, feats, _ = levels[-1]
        for up, block, (fine_xyz, fine_feats, fine_idx) in zip(self.ups, self.blocks, reversed(levels[:-1])):
            feats = block(fine_xyz, up(xyz, feats, fine_xyz, fine_feats), fine_idx)
            xyz = fine_xyz
        return self.head(feats)


def build_model(config):
    if config.get("arch", ARCH) != ARCH:
        raise ValueError("unsupported Q4 architecture '{}', expected '{}'".format(config.get("arch"), ARCH))
    model_class = PointTransformerCls if config["task"] == "cls" else PointTransformerSeg
    return model_class(num_classes=config["num_classes"], k=config["k"])
