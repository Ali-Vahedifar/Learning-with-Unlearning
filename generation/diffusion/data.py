"""CIFAR-10 in [-1, 1], read straight from the raw batches.

`LWU_DATA` is the directory that holds `cifar-10-batches-py/`, the same variable
the classification benchmark uses.
"""

import os
import pickle
from pathlib import Path

import numpy as np
import torch


def root():
    return Path(os.environ.get('LWU_DATA', './data')) / 'cifar-10-batches-py'


def _load(path):
    with open(path, 'rb') as handle:
        return pickle.load(handle, encoding='bytes')


def cifar10():
    xs, ys = [], []
    for i in range(1, 6):
        d = _load(root() / f'data_batch_{i}')
        xs.append(d[b'data'])
        ys += d[b'labels']
    x = np.concatenate(xs).reshape(-1, 3, 32, 32).astype(np.float32) / 127.5 - 1.0
    return torch.from_numpy(x), torch.tensor(ys)


def cifar10_test():
    d = _load(root() / 'test_batch')
    x = d[b'data'].reshape(-1, 3, 32, 32).astype(np.float32) / 127.5 - 1.0
    return torch.from_numpy(x), torch.tensor(d[b'labels'])
