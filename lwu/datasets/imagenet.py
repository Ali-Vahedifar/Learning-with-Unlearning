"""
ImageNet-family continual learning datasets for the LwU framework.

Provides ImageNet-1K (ResNet-18 experiments) and ImageNet-R / ImageNet-A
(ViT-B/16 + LoRA experiments), each split into disjoint class-incremental
tasks using the same protocol as `CIFAR100Dataset`.

Directory layout expected (standard `ImageFolder` structure):

    <root>/imagenet/train/n01440764/*.JPEG
    <root>/imagenet/val/n01440764/*.JPEG
    <root>/imagenet-r/<wnid>/*.jpg
    <root>/imagenet-a/<wnid>/*.jpg

ImageNet-R and ImageNet-A ship as a single split; both are partitioned into
train/test here with a fixed seed so the split is reproducible across runs.

None of these datasets can be downloaded automatically. ImageNet-1K requires
registration at https://image-net.org; ImageNet-R and ImageNet-A are available
from https://github.com/hendrycks/imagenet-r and .../natural-adv-examples.
"""

import os
from typing import List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, transforms

from lwu.datasets.continual_datasets import ContinualDataset


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def _build_transforms(image_size: int = 224, train: bool = True):
    if train:
        return transforms.Compose([
            transforms.RandomResizedCrop(image_size, scale=(0.7, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])
    return transforms.Compose([
        transforms.Resize(int(image_size * 1.14)),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


class _ImageNetFamilyDataset(ContinualDataset):
    """Shared task-splitting logic for the ImageNet-family datasets."""

    num_classes: int = 0
    subdir: str = ''

    def __init__(
        self,
        root: str = './data',
        num_tasks: int = 10,
        val_split: float = 0.1,
        seed: int = 42,
        download: bool = False,
        image_size: int = 224,
        test_split: float = 0.2
    ):
        super().__init__(root, num_tasks, val_split, seed, download=False)

        if self.num_classes % num_tasks != 0:
            raise ValueError(
                f"{type(self).__name__}: {self.num_classes} classes cannot be "
                f"divided evenly into {num_tasks} tasks."
            )

        self.image_size = image_size
        self.test_split = test_split
        self.classes_per_task = self.num_classes // num_tasks

        self.train_transform = _build_transforms(image_size, train=True)
        self.test_transform = _build_transforms(image_size, train=False)

        self._load_datasets()
        self._build_task_splits()

    # -- dataset loading ---------------------------------------------------- #

    def _require_dir(self, path: str) -> str:
        if not os.path.isdir(path):
            raise FileNotFoundError(
                f"Expected dataset directory not found: {path}\n"
                f"{type(self).__name__} cannot be downloaded automatically; "
                f"see the module docstring for the source."
            )
        return path

    def _load_datasets(self) -> None:
        raise NotImplementedError

    # -- task construction -------------------------------------------------- #

    def _build_task_splits(self) -> None:
        rng = np.random.RandomState(self.seed)
        all_classes = np.arange(self.num_classes)
        rng.shuffle(all_classes)

        self.task_classes = []
        for t in range(self.num_tasks):
            start = t * self.classes_per_task
            self.task_classes.append(
                all_classes[start:start + self.classes_per_task].tolist()
            )

        self.class_to_task = {
            cls: task_id
            for task_id, classes in enumerate(self.task_classes)
            for cls in classes
        }

    @staticmethod
    def _targets_of(dataset: Dataset) -> np.ndarray:
        if hasattr(dataset, 'targets'):
            return np.asarray(dataset.targets)
        if isinstance(dataset, Subset):
            parent = np.asarray(dataset.dataset.targets)
            return parent[np.asarray(dataset.indices)]
        raise AttributeError(f"Cannot extract targets from {type(dataset)}")

    def _get_class_indices(self, dataset: Dataset, classes: List[int]) -> List[int]:
        targets = self._targets_of(dataset)
        mask = np.isin(targets, classes)
        return np.where(mask)[0].tolist()

    # -- loaders ------------------------------------------------------------ #

    def get_task_loaders(
        self,
        task_id: int,
        batch_size: int = 64,
        num_workers: int = 4
    ) -> Tuple[DataLoader, DataLoader, DataLoader]:
        classes = self.task_classes[task_id]

        train_indices = self._get_class_indices(self.train_dataset, classes)
        test_indices = self._get_class_indices(self.test_dataset, classes)

        rng = np.random.RandomState(self.seed + task_id)
        rng.shuffle(train_indices)
        val_size = int(len(train_indices) * self.val_split)
        val_indices, train_indices = train_indices[:val_size], train_indices[val_size:]

        def _loader(dataset, indices, shuffle):
            return DataLoader(
                Subset(dataset, indices),
                batch_size=batch_size,
                shuffle=shuffle,
                num_workers=num_workers,
                pin_memory=True,
                persistent_workers=num_workers > 0
            )

        return (
            _loader(self.train_dataset, train_indices, True),
            _loader(self.train_dataset, val_indices, False),
            _loader(self.test_dataset, test_indices, False),
        )

    def get_forget_retain_loaders(
        self,
        forget_classes: List[int],
        batch_size: int = 64,
        num_workers: int = 4
    ) -> Tuple[DataLoader, DataLoader]:
        retain_classes = [c for c in range(self.num_classes) if c not in forget_classes]

        forget_indices = self._get_class_indices(self.train_dataset, forget_classes)
        retain_indices = self._get_class_indices(self.train_dataset, retain_classes)

        def _loader(indices):
            return DataLoader(
                Subset(self.train_dataset, indices),
                batch_size=batch_size,
                shuffle=True,
                num_workers=num_workers,
                pin_memory=True
            )

        return _loader(forget_indices), _loader(retain_indices)

    def get_all_seen_loader(
        self,
        tasks_seen: List[int],
        batch_size: int = 64,
        num_workers: int = 4,
        train: bool = True
    ) -> DataLoader:
        classes: List[int] = []
        for task_id in tasks_seen:
            classes.extend(self.task_classes[task_id])

        dataset = self.train_dataset if train else self.test_dataset
        indices = self._get_class_indices(dataset, classes)

        return DataLoader(
            Subset(dataset, indices),
            batch_size=batch_size,
            shuffle=train,
            num_workers=num_workers,
            pin_memory=True
        )


class ImageNet1KDataset(_ImageNetFamilyDataset):
    """ImageNet-1K (ILSVRC-2012), 1000 classes. Used with ResNet-18."""

    num_classes = 1000
    subdir = 'imagenet'

    def _load_datasets(self) -> None:
        base = os.path.join(self.root, self.subdir)
        train_dir = self._require_dir(os.path.join(base, 'train'))
        val_dir = self._require_dir(os.path.join(base, 'val'))

        self.train_dataset = datasets.ImageFolder(train_dir, transform=self.train_transform)
        self.test_dataset = datasets.ImageFolder(val_dir, transform=self.test_transform)


class _SingleSplitImageNetDataset(_ImageNetFamilyDataset):
    """
    Base for ImageNet-R / ImageNet-A, which ship as one directory and are split
    into train/test here with a fixed seed.
    """

    def _load_datasets(self) -> None:
        base = self._require_dir(os.path.join(self.root, self.subdir))

        full_train = datasets.ImageFolder(base, transform=self.train_transform)
        full_test = datasets.ImageFolder(base, transform=self.test_transform)

        n = len(full_train)
        rng = np.random.RandomState(self.seed)
        perm = rng.permutation(n)
        n_test = int(n * self.test_split)

        test_idx = perm[:n_test].tolist()
        train_idx = perm[n_test:].tolist()

        self.train_dataset = Subset(full_train, train_idx)
        self.test_dataset = Subset(full_test, test_idx)


class ImageNetRDataset(_SingleSplitImageNetDataset):
    """ImageNet-R (rendition shift), 200 classes. Used with ViT-B/16 + LoRA."""

    num_classes = 200
    subdir = 'imagenet-r'


class ImageNetADataset(_SingleSplitImageNetDataset):
    """ImageNet-A (natural adversarial examples), 200 classes."""

    num_classes = 200
    subdir = 'imagenet-a'


IMAGENET_DATASETS = {
    'imagenet1k': ImageNet1KDataset,
    'imagenet-1k': ImageNet1KDataset,
    'imagenet_r': ImageNetRDataset,
    'imagenet-r': ImageNetRDataset,
    'imagenet_a': ImageNetADataset,
    'imagenet-a': ImageNetADataset,
}
