import numpy as np
import contextlib
import torch
from torch.utils.data import Dataset
import torch.utils.data
import torch.nn.functional as F

class CustomDataset(Dataset):
    def __init__(self, X, y, sid=None):
        self.X = X
        self.y = y
        self.sid = sid

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        x = self.X[idx]
        y = self.y[idx]
        sid = self.sid[idx]
        data = (x, y, sid)
        return data

class CustomDataset1M(Dataset):
    def __init__(self, X, y, mask, sid=None):
        self.X = X
        self.mask = mask
        self.y = y
        self.sid = sid

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        x = self.X[idx]
        y = self.y[idx]
        mask = self.mask[idx]
        sid = self.sid[idx]
        data = (x, y, mask, sid)
        return data

class CustomDataset2M(Dataset):
    def __init__(self, X1, X2, y, mask1, mask2, sid=None):
        self.X1 = X1
        self.X2 = X2
        self.mask1 = mask1
        self.mask2 = mask2
        self.y = y
        self.sid = sid

    def __len__(self):
        return self.X1.shape[0]

    def __getitem__(self, idx):
        x1 = self.X1[idx]
        x2 = self.X2[idx]
        y = self.y[idx]
        mask1 = self.mask1[idx]
        mask2 = self.mask2[idx]
        sid = self.sid[idx]
        data = (x1, x2, y, mask1, mask2, sid)
        return data

def l2_normalize(d):
    d = d.contiguous()
    d_norm = F.normalize(d.view(d.size(0), -1), dim=1, p=2).view(d.size())
    return d_norm

def entropy_loss(logit_x):
    p = F.softmax(logit_x, dim=1)
    return -(p*F.log_softmax(logit_x, dim=1)).sum(dim=1).mean(dim=0)

def kl_div_with_logit(q_logit, p_logit, average=True):
    q = F.softmax(q_logit, dim=1)
    logq = F.log_softmax(q_logit, dim=1)
    logp = F.log_softmax(p_logit, dim=1)

    if average is True:
        qlogq = (q*logq).sum(dim=1).mean(dim=0)
        qlogp = (q*logp).sum(dim=1).mean(dim=0)
    else:
        qlogq = (q*logq).sum(dim=1, keepdim=True)
        qlogp = (q*logp).sum(dim=1, keepdim=True)

    return qlogq - qlogp

@contextlib.contextmanager
def _disable_tracking_bn_stats(model):
    def switch_attr(m):
        if hasattr(m, 'track_running_stats'):
            m.track_running_stats ^= True
    model.apply(switch_attr)
    yield
    model.apply(switch_attr)

def scale_0_1(x):
    x = (x - x.min()) / (x.max() - x.min())
    return x

class ImbalancedDatasetSampler(torch.utils.data.sampler.Sampler):
    """Samples elements randomly from a given list of indices for imbalanced dataset
    Arguments:
        indices (list, optional): a list of indices
        num_samples (int, optional): number of samples to draw
    """

    def __init__(self, dataset, indices=None, num_samples=None):
        # if indices is not provided, 
        # all elements in the dataset will be considered
        self.indices = list(range(len(dataset))) \
            if indices is None else indices

        # if num_samples is not provided, 
        # draw `len(indices)` samples in each iteration
        self.num_samples = len(self.indices) \
            if num_samples is None else num_samples

        # distribution of classes in the dataset 
        label_to_count = {}
        for idx in self.indices:
            label = self._get_label(dataset, idx)
            if label in label_to_count:
                label_to_count[label] += 1
            else:
                label_to_count[label] = 1

        # weight for each sample
        weights = [1.0 / label_to_count[self._get_label(dataset, idx)]
                   for idx in self.indices]
        self.weights = torch.DoubleTensor(weights)

    def _get_label(self, dataset, idx):
        return dataset.y[idx]

    def __iter__(self):
        return (self.indices[i] for i in torch.multinomial(
            self.weights, self.num_samples, replacement=True))

    def __len__(self):
        return self.num_samples
