import os
from torch.utils.data import DataLoader, Dataset, Subset
import numpy as np
import torch



class CustomDataSet(Dataset):
    """Load data under folders"""
    def __init__(self, args, train=True):
        self.main_dir = args.main_dir
        self.task = args.task

        if train:
            data_path = os.path.join(self.main_dir, self.task, "data_train.npy")
            label_path = os.path.join(self.main_dir, self.task, "label_train.npy")
        else:
            data_path = os.path.join(self.main_dir, self.task, "data_test.npy")
            label_path = os.path.join(self.main_dir, self.task, "label_test.npy")

        self.data = torch.from_numpy(np.load(data_path))
        self.label = torch.from_numpy(np.load(label_path)).to(torch.long) # in cls task, (N,), in seg task, (N, 10000), N is the number of objects


    def __len__(self):
        return self.data.size()[0]

    def __getitem__(self, idx):
        return self.data[idx], self.label[idx]


def get_data_loader(args, train=True):
    """
    Creates training and test data loaders
    """
    dataset = CustomDataSet(args=args, train=train)
    dloader = DataLoader(dataset=dataset, batch_size=args.batch_size, shuffle=train, num_workers=args.num_workers)


    return dloader


def split_train_val(labels, val_frac, seed, stratify):
    if not 0 < val_frac < 1:
        raise ValueError("--val_frac must be in (0, 1), got {}".format(val_frac))
    rng = np.random.default_rng(seed)
    groups = [np.where(labels == c)[0] for c in np.unique(labels)] if stratify else [np.arange(len(labels))]
    val_idx = np.sort(np.concatenate([rng.permutation(g)[:int(round(len(g) * val_frac))] for g in groups]))
    train_idx = np.setdiff1d(np.arange(len(labels)), val_idx)
    return train_idx, val_idx


def get_train_val_loaders(args):
    dataset = CustomDataSet(args=args, train=True)
    labels = dataset.label.numpy() if args.task == "cls" else np.zeros(len(dataset))
    train_idx, val_idx = split_train_val(labels, args.val_frac, args.seed, stratify=args.task == "cls")
    train_loader = DataLoader(Subset(dataset, train_idx.tolist()), batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, drop_last=True)
    val_loader = DataLoader(Subset(dataset, val_idx.tolist()), batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    return train_loader, val_loader, train_idx, val_idx, dataset.data.shape[1]
