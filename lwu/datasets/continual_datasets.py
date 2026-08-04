"""
Dataset Loaders for Continual Learning and Machine Unlearning

Implements data loading utilities for:
- CIFAR-100
- TinyImageNet  
- RTI (Robotic Tactile Internet)

Supports both Task-IL and Class-IL scenarios with configurable task splits.
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import datasets, transforms
from typing import List, Tuple, Dict, Optional, Union
import pickle


class ContinualDataset:
    """Base class for continual learning datasets."""
    
    def __init__(
        self,
        root: str,
        num_tasks: int = 10,
        val_split: float = 0.1,
        seed: int = 42,
        download: bool = True
    ):
        """
        Initialize continual learning dataset.
        
        Args:
            root: Root directory for dataset
            num_tasks: Number of tasks to split into
            val_split: Fraction of data to use for validation
            seed: Random seed for reproducibility
            download: Whether to download the dataset
        """
        self.root = root
        self.num_tasks = num_tasks
        self.val_split = val_split
        self.seed = seed
        self.download = download
        
        np.random.seed(seed)
        torch.manual_seed(seed)
        
        self.train_dataset = None
        self.test_dataset = None
        self.task_classes = []  # List of class indices for each task
        
    def get_task_loaders(
        self,
        task_id: int,
        batch_size: int = 64,
        num_workers: int = 4
    ) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """
        Get train, validation, and test loaders for a specific task.
        
        Args:
            task_id: Task identifier (0-indexed)
            batch_size: Batch size for data loaders
            num_workers: Number of worker processes
            
        Returns:
            Tuple of (train_loader, val_loader, test_loader)
        """
        raise NotImplementedError
        
    def get_forget_retain_loaders(
        self,
        forget_classes: List[int],
        batch_size: int = 64,
        num_workers: int = 4
    ) -> Tuple[DataLoader, DataLoader]:
        """
        Get loaders for forget and retain sets.
        
        Args:
            forget_classes: List of class indices to forget
            batch_size: Batch size for data loaders
            num_workers: Number of worker processes
            
        Returns:
            Tuple of (forget_loader, retain_loader)
        """
        raise NotImplementedError
        
    def get_all_seen_loader(
        self,
        tasks_seen: List[int],
        batch_size: int = 64,
        num_workers: int = 4
    ) -> DataLoader:
        """Get loader for all classes seen so far."""
        raise NotImplementedError


class CIFAR100Dataset(ContinualDataset):
    """CIFAR-100 dataset for continual learning."""
    
    def __init__(
        self,
        root: str = './data',
        num_tasks: int = 10,
        val_split: float = 0.1,
        seed: int = 42,
        download: bool = True,
        image_size: int = 32
    ):
        super().__init__(root, num_tasks, val_split, seed, download)

        self.num_classes = 100
        self.classes_per_task = self.num_classes // num_tasks
        self.image_size = image_size

        # Data transforms.
        # image_size == 32 is the ResNet-18 setting used in the main tables.
        # image_size == 224 is the ViT-B/16 + LoRA setting; the pretrained
        # backbone expects ImageNet statistics rather than CIFAR statistics.
        if image_size == 32:
            self.train_transform = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.5071, 0.4867, 0.4408],
                    std=[0.2675, 0.2565, 0.2761]
                )
            ])

            self.test_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.5071, 0.4867, 0.4408],
                    std=[0.2675, 0.2565, 0.2761]
                )
            ])
        else:
            imagenet_mean = [0.485, 0.456, 0.406]
            imagenet_std = [0.229, 0.224, 0.225]

            self.train_transform = transforms.Compose([
                transforms.Resize(image_size),
                transforms.RandomCrop(image_size, padding=image_size // 8),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=imagenet_mean, std=imagenet_std)
            ])

            self.test_transform = transforms.Compose([
                transforms.Resize(image_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=imagenet_mean, std=imagenet_std)
            ])
        
        # Load datasets
        self.train_dataset = datasets.CIFAR100(
            root=root, train=True, download=download, transform=self.train_transform
        )
        self.test_dataset = datasets.CIFAR100(
            root=root, train=False, download=download, transform=self.test_transform
        )
        
        # Create task splits (random class assignment)
        all_classes = list(range(self.num_classes))
        np.random.shuffle(all_classes)
        
        self.task_classes = []
        for t in range(num_tasks):
            start = t * self.classes_per_task
            end = start + self.classes_per_task
            self.task_classes.append(all_classes[start:end])
            
        # Create class to task mapping
        self.class_to_task = {}
        for task_id, classes in enumerate(self.task_classes):
            for cls in classes:
                self.class_to_task[cls] = task_id
                
    def _get_class_indices(
        self,
        dataset: Dataset,
        classes: List[int]
    ) -> List[int]:
        """Get indices of samples belonging to specified classes."""
        indices = []
        targets = np.array(dataset.targets)
        for cls in classes:
            cls_indices = np.where(targets == cls)[0].tolist()
            indices.extend(cls_indices)
        return indices
    
    def get_task_loaders(
        self,
        task_id: int,
        batch_size: int = 64,
        num_workers: int = 4
    ) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """Get loaders for a specific task."""
        classes = self.task_classes[task_id]
        
        # Get indices for this task
        train_indices = self._get_class_indices(self.train_dataset, classes)
        test_indices = self._get_class_indices(self.test_dataset, classes)
        
        # Split train into train/val
        np.random.shuffle(train_indices)
        val_size = int(len(train_indices) * self.val_split)
        val_indices = train_indices[:val_size]
        train_indices = train_indices[val_size:]
        
        # Create subsets
        train_subset = Subset(self.train_dataset, train_indices)
        val_subset = Subset(self.train_dataset, val_indices)
        test_subset = Subset(self.test_dataset, test_indices)
        
        # Create loaders
        train_loader = DataLoader(
            train_subset, batch_size=batch_size, shuffle=True,
            num_workers=num_workers, pin_memory=True
        )
        val_loader = DataLoader(
            val_subset, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=True
        )
        test_loader = DataLoader(
            test_subset, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=True
        )
        
        return train_loader, val_loader, test_loader
    
    def get_forget_retain_loaders(
        self,
        forget_classes: List[int],
        batch_size: int = 64,
        num_workers: int = 4
    ) -> Tuple[DataLoader, DataLoader]:
        """Get loaders for forget and retain sets."""
        retain_classes = [c for c in range(self.num_classes) if c not in forget_classes]
        
        forget_indices = self._get_class_indices(self.train_dataset, forget_classes)
        retain_indices = self._get_class_indices(self.train_dataset, retain_classes)
        
        forget_subset = Subset(self.train_dataset, forget_indices)
        retain_subset = Subset(self.train_dataset, retain_indices)
        
        forget_loader = DataLoader(
            forget_subset, batch_size=batch_size, shuffle=True,
            num_workers=num_workers, pin_memory=True
        )
        retain_loader = DataLoader(
            retain_subset, batch_size=batch_size, shuffle=True,
            num_workers=num_workers, pin_memory=True
        )
        
        return forget_loader, retain_loader
    
    def get_all_seen_loader(
        self,
        tasks_seen: List[int],
        batch_size: int = 64,
        num_workers: int = 4,
        train: bool = True
    ) -> DataLoader:
        """Get loader for all classes seen so far."""
        classes = []
        for task_id in tasks_seen:
            classes.extend(self.task_classes[task_id])
            
        dataset = self.train_dataset if train else self.test_dataset
        indices = self._get_class_indices(dataset, classes)
        
        subset = Subset(dataset, indices)
        loader = DataLoader(
            subset, batch_size=batch_size, shuffle=train,
            num_workers=num_workers, pin_memory=True
        )
        
        return loader


class TinyImageNetDataset(ContinualDataset):
    """TinyImageNet dataset for continual learning."""
    
    def __init__(
        self,
        root: str = './data/tiny-imagenet-200',
        num_tasks: int = 10,
        val_split: float = 0.1,
        seed: int = 42,
        download: bool = True
    ):
        super().__init__(root, num_tasks, val_split, seed, download)
        
        self.num_classes = 200
        self.classes_per_task = self.num_classes // num_tasks
        
        # Data transforms
        self.train_transform = transforms.Compose([
            transforms.RandomCrop(64, padding=8),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
        
        self.test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
        
        # Check if dataset exists, if not provide download instructions
        train_dir = os.path.join(root, 'train')
        val_dir = os.path.join(root, 'val')
        
        if not os.path.exists(train_dir):
            raise FileNotFoundError(
                f"TinyImageNet not found at {root}. "
                "Please download from http://cs231n.stanford.edu/tiny-imagenet-200.zip "
                "and extract to the specified root directory."
            )
        
        # Load datasets
        self.train_dataset = datasets.ImageFolder(train_dir, transform=self.train_transform)
        self.test_dataset = datasets.ImageFolder(val_dir, transform=self.test_transform)
        
        # Create task splits
        all_classes = list(range(self.num_classes))
        np.random.shuffle(all_classes)
        
        self.task_classes = []
        for t in range(num_tasks):
            start = t * self.classes_per_task
            end = start + self.classes_per_task
            self.task_classes.append(all_classes[start:end])
            
        # Create class to task mapping
        self.class_to_task = {}
        for task_id, classes in enumerate(self.task_classes):
            for cls in classes:
                self.class_to_task[cls] = task_id
                
    def _get_class_indices(
        self,
        dataset: Dataset,
        classes: List[int]
    ) -> List[int]:
        """Get indices of samples belonging to specified classes."""
        indices = []
        targets = np.array([s[1] for s in dataset.samples])
        for cls in classes:
            cls_indices = np.where(targets == cls)[0].tolist()
            indices.extend(cls_indices)
        return indices
    
    def get_task_loaders(
        self,
        task_id: int,
        batch_size: int = 64,
        num_workers: int = 4
    ) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """Get loaders for a specific task."""
        classes = self.task_classes[task_id]
        
        train_indices = self._get_class_indices(self.train_dataset, classes)
        test_indices = self._get_class_indices(self.test_dataset, classes)
        
        np.random.shuffle(train_indices)
        val_size = int(len(train_indices) * self.val_split)
        val_indices = train_indices[:val_size]
        train_indices = train_indices[val_size:]
        
        train_subset = Subset(self.train_dataset, train_indices)
        val_subset = Subset(self.train_dataset, val_indices)
        test_subset = Subset(self.test_dataset, test_indices)
        
        train_loader = DataLoader(
            train_subset, batch_size=batch_size, shuffle=True,
            num_workers=num_workers, pin_memory=True
        )
        val_loader = DataLoader(
            val_subset, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=True
        )
        test_loader = DataLoader(
            test_subset, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=True
        )
        
        return train_loader, val_loader, test_loader
    
    def get_forget_retain_loaders(
        self,
        forget_classes: List[int],
        batch_size: int = 64,
        num_workers: int = 4
    ) -> Tuple[DataLoader, DataLoader]:
        """Get loaders for forget and retain sets."""
        retain_classes = [c for c in range(self.num_classes) if c not in forget_classes]
        
        forget_indices = self._get_class_indices(self.train_dataset, forget_classes)
        retain_indices = self._get_class_indices(self.train_dataset, retain_classes)
        
        forget_subset = Subset(self.train_dataset, forget_indices)
        retain_subset = Subset(self.train_dataset, retain_indices)
        
        forget_loader = DataLoader(
            forget_subset, batch_size=batch_size, shuffle=True,
            num_workers=num_workers, pin_memory=True
        )
        retain_loader = DataLoader(
            retain_subset, batch_size=batch_size, shuffle=True,
            num_workers=num_workers, pin_memory=True
        )
        
        return forget_loader, retain_loader
    
    def get_all_seen_loader(
        self,
        tasks_seen: List[int],
        batch_size: int = 64,
        num_workers: int = 4,
        train: bool = True
    ) -> DataLoader:
        """Get loader for all classes seen so far."""
        classes = []
        for task_id in tasks_seen:
            classes.extend(self.task_classes[task_id])
            
        dataset = self.train_dataset if train else self.test_dataset
        indices = self._get_class_indices(dataset, classes)
        
        subset = Subset(dataset, indices)
        loader = DataLoader(
            subset, batch_size=batch_size, shuffle=train,
            num_workers=num_workers, pin_memory=True
        )
        
        return loader


class RTIDataset(Dataset):
    """
    SYNTHETIC smoke-test fixture for the RTI data-loading path.

    This class generates haptic-like signals procedurally with numpy. It exists
    so the data pipeline can be exercised without distributing participant data,
    and it is NOT the dataset used for any reported RTI result.

    The real Robotic Tactile Internet collection (20 anonymised participants,
    Novint Falcon haptic device at a 1 kHz control loop, CHAI3D environment
    compliant with IEEE 1918.1.1) is an industrial dataset subject to release
    approval and will be published upon acceptance, together with its data card
    and collection protocol. Until then this fixture is the only RTI loader in
    the repository, and it should not be used to draw conclusions.

    Actions: no movement, pressing, tapping, free air movement, dragging
    """
    
    BASE_ACTIONS = ['no_movement', 'pressing', 'tapping', 'free_air', 'dragging']

    @classmethod
    def default_actions(cls, num_classes: int) -> List[str]:
        """
        Return `num_classes` synthetic action labels.

        The five base names are the interaction primitives; beyond that the
        fixture emits indexed placeholders so that protocols requiring more
        classes than the base set can still be exercised. These placeholders
        carry no semantics and exist only to keep the loading path runnable.
        """
        if num_classes <= len(cls.BASE_ACTIONS):
            return list(cls.BASE_ACTIONS[:num_classes])
        extra = [f'synthetic_action_{i}'
                 for i in range(len(cls.BASE_ACTIONS), num_classes)]
        return list(cls.BASE_ACTIONS) + extra

    def __init__(
        self,
        root: str = './data/rti',
        train: bool = True,
        transform=None,
        num_participants: int = 20,
        actions: List[str] = None,
        sampling_rate: int = 1000,
        sequence_length: int = 100
    ):
        """
        Initialize RTI dataset.
        
        Args:
            root: Root directory for data
            train: Whether to load training data
            transform: Optional transform to apply
            num_participants: Number of participants
            actions: List of action names
            sampling_rate: Sampling rate in Hz
            sequence_length: Length of each sample sequence
        """
        self.root = root
        self.train = train
        self.transform = transform
        self.num_participants = num_participants
        self.sampling_rate = sampling_rate
        self.sequence_length = sequence_length
        
        if actions is None:
            self.actions = ['no_movement', 'pressing', 'tapping', 'free_air', 'dragging']
        else:
            self.actions = actions
            
        self.num_classes = len(self.actions)
        
        # Generate synthetic RTI data (for reproducible experiments)
        self._generate_synthetic_data()
        
    def _generate_synthetic_data(self):
        """Generate synthetic RTI data for experiments."""
        np.random.seed(42 if self.train else 43)
        
        samples_per_action = 200 if self.train else 50
        
        self.data = []
        self.targets = []
        
        for action_idx, action in enumerate(self.actions):
            for _ in range(samples_per_action):
                # Generate synthetic haptic signal based on action type
                signal = self._generate_haptic_signal(action)
                self.data.append(signal)
                self.targets.append(action_idx)
                
        self.data = np.array(self.data, dtype=np.float32)
        self.targets = np.array(self.targets, dtype=np.int64)
        
        # Shuffle
        indices = np.random.permutation(len(self.data))
        self.data = self.data[indices]
        self.targets = self.targets[indices]
        
    def _generate_haptic_signal(self, action: str) -> np.ndarray:
        """Generate synthetic haptic signal for given action."""
        t = np.linspace(0, 1, self.sequence_length)
        
        # Base signal with noise
        base_noise = np.random.randn(self.sequence_length) * 0.1
        
        if action == 'no_movement':
            signal = base_noise
        elif action == 'pressing':
            # Sustained force
            signal = np.sin(2 * np.pi * 0.5 * t) * 0.8 + base_noise
        elif action == 'tapping':
            # Periodic impulses
            impulses = np.zeros(self.sequence_length)
            for i in range(0, self.sequence_length, 20):
                if i < self.sequence_length:
                    impulses[i] = 1.0
            signal = impulses + base_noise
        elif action == 'free_air':
            # Low-frequency oscillation
            signal = np.sin(2 * np.pi * 2 * t) * 0.3 + base_noise
        elif action == 'dragging':
            # Friction-like signal
            signal = np.sin(2 * np.pi * 5 * t) * 0.5 + 0.3 * np.sign(np.sin(2 * np.pi * 0.5 * t)) + base_noise
        else:
            signal = base_noise
            
        # Extract features: mean, std, max, min, etc.
        features = [
            signal.mean(),
            signal.std(),
            signal.max(),
            signal.min(),
            np.abs(signal).mean(),
            np.diff(signal).std(),
            np.sum(signal > 0.5) / len(signal),
            np.sum(signal < -0.5) / len(signal),
            np.correlate(signal, signal)[0] / len(signal),
            np.fft.fft(signal)[:10].real.mean()
        ]
        
        return np.array(features, dtype=np.float32)
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        data = self.data[idx]
        target = self.targets[idx]
        
        if self.transform:
            data = self.transform(data)
        else:
            data = torch.from_numpy(data)
            
        return data, target


class RTIContinualDataset(ContinualDataset):
    """RTI dataset wrapper for continual learning."""
    
    def __init__(
        self,
        root: str = './data/rti',
        num_tasks: int = 10,
        val_split: float = 0.1,
        seed: int = 42,
        download: bool = False,
        classes_per_task: int = 1,
        num_classes: int = None
    ):
        super().__init__(root, num_tasks, val_split, seed, download)

        # The class count follows the requested protocol rather than being
        # hardcoded, so the wrapper can instantiate the 10-task TIL/CIL
        # protocol reported for RTI. Passing `num_classes` explicitly covers
        # the unlearning setting, where a single joint task spans all classes.
        # The backing store below is the SYNTHETIC fixture (see RTIDataset);
        # the real collection is described in Appendix H.
        if num_classes is not None:
            self.num_classes = num_classes
            self.classes_per_task = max(1, num_classes // max(1, num_tasks))
        else:
            self.classes_per_task = classes_per_task
            self.num_classes = num_tasks * classes_per_task
        
        # Load datasets. The fixture is asked for exactly as many action
        # classes as the requested protocol needs.
        actions = RTIDataset.default_actions(self.num_classes)
        self.train_dataset = RTIDataset(root, train=True, actions=actions)
        self.test_dataset = RTIDataset(root, train=False, actions=actions)

        # Create task splits
        self.task_classes = [
            list(range(t * self.classes_per_task, (t + 1) * self.classes_per_task))
            for t in range(num_tasks)
        ]

        # Class to task mapping
        self.class_to_task = {}
        for task_id, classes in enumerate(self.task_classes):
            for cls in classes:
                self.class_to_task[cls] = task_id
        
    def _get_class_indices(
        self,
        dataset: RTIDataset,
        classes: List[int]
    ) -> List[int]:
        """Get indices of samples belonging to specified classes."""
        indices = []
        for cls in classes:
            cls_indices = np.where(dataset.targets == cls)[0].tolist()
            indices.extend(cls_indices)
        return indices
    
    def get_task_loaders(
        self,
        task_id: int,
        batch_size: int = 64,
        num_workers: int = 4
    ) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """Get loaders for a specific task."""
        classes = self.task_classes[task_id]
        
        train_indices = self._get_class_indices(self.train_dataset, classes)
        test_indices = self._get_class_indices(self.test_dataset, classes)
        
        np.random.shuffle(train_indices)
        val_size = int(len(train_indices) * self.val_split)
        val_indices = train_indices[:val_size]
        train_indices = train_indices[val_size:]
        
        train_subset = Subset(self.train_dataset, train_indices)
        val_subset = Subset(self.train_dataset, val_indices)
        test_subset = Subset(self.test_dataset, test_indices)
        
        train_loader = DataLoader(
            train_subset, batch_size=batch_size, shuffle=True,
            num_workers=num_workers, pin_memory=True
        )
        val_loader = DataLoader(
            val_subset, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=True
        )
        test_loader = DataLoader(
            test_subset, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=True
        )
        
        return train_loader, val_loader, test_loader
    
    def get_forget_retain_loaders(
        self,
        forget_classes: List[int],
        batch_size: int = 64,
        num_workers: int = 4
    ) -> Tuple[DataLoader, DataLoader]:
        """Get loaders for forget and retain sets."""
        retain_classes = [c for c in range(self.num_classes) if c not in forget_classes]
        
        forget_indices = self._get_class_indices(self.train_dataset, forget_classes)
        retain_indices = self._get_class_indices(self.train_dataset, retain_classes)
        
        forget_subset = Subset(self.train_dataset, forget_indices)
        retain_subset = Subset(self.train_dataset, retain_indices)
        
        forget_loader = DataLoader(
            forget_subset, batch_size=batch_size, shuffle=True,
            num_workers=num_workers, pin_memory=True
        )
        retain_loader = DataLoader(
            retain_subset, batch_size=batch_size, shuffle=True,
            num_workers=num_workers, pin_memory=True
        )
        
        return forget_loader, retain_loader
    
    def get_all_seen_loader(
        self,
        tasks_seen: List[int],
        batch_size: int = 64,
        num_workers: int = 4,
        train: bool = True
    ) -> DataLoader:
        """Get loader for all classes seen so far."""
        classes = []
        for task_id in tasks_seen:
            classes.extend(self.task_classes[task_id])
            
        dataset = self.train_dataset if train else self.test_dataset
        indices = self._get_class_indices(dataset, classes)
        
        subset = Subset(dataset, indices)
        loader = DataLoader(
            subset, batch_size=batch_size, shuffle=train,
            num_workers=num_workers, pin_memory=True
        )
        
        return loader


def _cifar100_vit(**kwargs) -> ContinualDataset:
    """CIFAR-100 at 224x224 for the ViT-B/16 + LoRA experiments (Table 2)."""
    kwargs.setdefault('image_size', 224)
    return CIFAR100Dataset(**kwargs)


def get_dataset(
    name: str,
    root: str = './data',
    num_tasks: int = 10,
    **kwargs
) -> ContinualDataset:
    """
    Factory function to create datasets.
    
    Args:
        name: Dataset name ('cifar100', 'tinyimagenet', 'rti')
        root: Root directory for dataset
        num_tasks: Number of tasks
        **kwargs: Additional arguments
        
    Returns:
        ContinualDataset instance
    """
    from lwu.datasets.imagenet import IMAGENET_DATASETS

    datasets_map = {
        'cifar100': CIFAR100Dataset,
        'cifar100-vit': _cifar100_vit,
        'cifar100_vit': _cifar100_vit,
        'tinyimagenet': TinyImageNetDataset,
        'rti': RTIContinualDataset
    }
    datasets_map.update(IMAGENET_DATASETS)

    if name.lower() not in datasets_map:
        raise ValueError(f"Unknown dataset: {name}. Available: {list(datasets_map.keys())}")

    # `data_dir` is the flag name used by the training entry points; it is an
    # alias for `root`. Accepting both keeps scripts and library in agreement.
    data_dir = kwargs.pop('data_dir', None)
    if data_dir is not None:
        root = data_dir

    # `scenario` (task / class) selects the evaluation protocol, not the class
    # partition, so it is recorded on the dataset rather than passed to the
    # constructor, whose signature does not accept it.
    scenario = kwargs.pop('scenario', None)

    dataset = datasets_map[name.lower()](root=root, num_tasks=num_tasks, **kwargs)

    if scenario is not None:
        dataset.scenario = scenario

    return dataset
