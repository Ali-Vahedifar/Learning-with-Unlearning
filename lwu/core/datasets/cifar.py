import numpy as np
import torch
from torchvision import datasets, transforms

from core.datasets.cifar100_coarse import CoarseLabelCIFAR100, fine_to_coarse_map
from core.datasets.tinyimagenet import TinyImageNetDataset


def _transforms(mean, std):
    normalize = transforms.Normalize(mean=mean, std=std)
    train = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]
    )
    test = transforms.Compose([transforms.ToTensor(), normalize])
    return train, test


class CIFAR100Dataset:
    source = datasets.CIFAR100
    num_classes = 100
    mean, std = [0.5071, 0.4867, 0.4408], [0.2675, 0.2565, 0.2761]

    def __init__(self, root='./data', val_split=0.1, seed=42, download=True):
        self.root = root
        self.val_split = val_split
        np.random.seed(seed)
        torch.manual_seed(seed)
        self.train_transform, self.test_transform = _transforms(self.mean, self.std)
        self.train_dataset = self.source(
            root=root, train=True, download=download, transform=self.train_transform
        )
        self.val_dataset = self.source(
            root=root, train=True, download=False, transform=self.test_transform
        )
        self.test_dataset = self.source(
            root=root, train=False, download=download, transform=self.test_transform
        )


class CIFAR10Dataset(CIFAR100Dataset):
    source = datasets.CIFAR10
    num_classes = 10
    mean, std = [0.4914, 0.4822, 0.4465], [0.2470, 0.2435, 0.2616]


class CIFAR20Dataset(CIFAR100Dataset):
    """CIFAR-100 relabelled with its own 20 superclasses.

    `coarse_targets=False` keeps the fine label in `.targets` (so a forget set
    can be selected by fine class) while training and evaluation still use the
    coarse label -- this is what subclass deletion needs.
    """

    num_classes = 20

    def __init__(self, root='./data', coarse_targets=True, **kwargs):
        super().__init__(root, **kwargs)
        mapping = fine_to_coarse_map(root)
        for split in ('train_dataset', 'val_dataset', 'test_dataset'):
            setattr(
                self,
                split,
                CoarseLabelCIFAR100(
                    getattr(self, split), mapping, expose_coarse_targets=coarse_targets
                ),
            )


DATASETS = {
    'cifar10': CIFAR10Dataset,
    'cifar20': CIFAR20Dataset,
    'cifar100': CIFAR100Dataset,
    'tinyimagenet': TinyImageNetDataset,
}


def get_dataset(name, root='./data', **kwargs):
    if name not in DATASETS:
        raise ValueError(f'unknown dataset {name!r}; available: {sorted(DATASETS)}')
    return DATASETS[name](root=root, **kwargs)
