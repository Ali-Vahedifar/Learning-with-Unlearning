import os
import pickle

import numpy as np
from PIL import Image
from torch.utils.data import Dataset


def fine_to_coarse_map(root: str) -> list:
    path = os.path.join(root, 'cifar-100-python', 'train')
    with open(path, 'rb') as handle:
        batch = pickle.load(handle, encoding='latin1')
    fine = np.asarray(batch['fine_labels'])
    coarse = np.asarray(batch['coarse_labels'])
    mapping = [None] * 100
    for f, c in zip(fine, coarse):
        if mapping[f] is None:
            mapping[f] = int(c)
        elif mapping[f] != int(c):
            raise ValueError(f'fine label {f} maps to more than one coarse label')
    if any(m is None for m in mapping):
        raise ValueError('some fine labels never appeared in the training batch')
    return mapping


class CoarseLabelCIFAR100(Dataset):
    def __init__(self, base, fine_to_coarse, expose_coarse_targets=False):
        self.data = base.data
        self.fine_targets = list(base.targets)
        self.fine_to_coarse = fine_to_coarse
        self.expose_coarse_targets = bool(expose_coarse_targets)
        self.targets = (
            [fine_to_coarse[t] for t in self.fine_targets]
            if self.expose_coarse_targets
            else list(self.fine_targets)
        )
        self.transform = base.transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        image = Image.fromarray(self.data[index])
        if self.transform is not None:
            image = self.transform(image)
        return image, self.fine_to_coarse[self.fine_targets[index]]
