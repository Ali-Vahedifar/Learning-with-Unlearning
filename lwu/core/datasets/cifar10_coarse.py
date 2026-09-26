from core.datasets.cifar100_coarse import CoarseLabelCIFAR100 as CoarseLabelCIFAR10

CIFAR10_FINE_TO_COARSE = [0, 1, 2, 3, 4, 3, 2, 4, 0, 1]
NUM_COARSE_CLASSES = 5


def fine_to_coarse_map(root: str = None) -> list:
    return list(CIFAR10_FINE_TO_COARSE)


__all__ = [
    'CoarseLabelCIFAR10',
    'CIFAR10_FINE_TO_COARSE',
    'NUM_COARSE_CLASSES',
    'fine_to_coarse_map',
]
