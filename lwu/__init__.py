"""
LwU (Learning with Unlearning) Framework

A unified framework for simultaneous Continual Learning and Machine Unlearning
through principled parameter space decomposition.
"""

from lwu.models.lwu import LwU
from lwu.models.backbones import resnet18, resnet34, SimpleMLP, get_backbone
from lwu.datasets.continual_datasets import (
    CIFAR100Dataset,
    TinyImageNetDataset,
    RTIContinualDataset,
    get_dataset
)
from lwu.evaluation.metrics import (
    AccuracyMatrix,
    evaluate_task,
    evaluate_all_tasks,
    MembershipInferenceAttack,
    UnlearningEvaluator,
    ContinualLearningEvaluator
)
from lwu.utils.config import get_config, ExperimentConfig
from lwu.utils.training import LwUTrainer, BaselineTrainer, set_seed

__version__ = '1.0.0'

__all__ = [
    # Models
    'LwU',
    'resnet18',
    'resnet34',
    'SimpleMLP',
    'get_backbone',
    
    # Datasets
    'CIFAR100Dataset',
    'TinyImageNetDataset',
    'RTIContinualDataset',
    'get_dataset',
    
    # Evaluation
    'AccuracyMatrix',
    'evaluate_task',
    'evaluate_all_tasks',
    'MembershipInferenceAttack',
    'UnlearningEvaluator',
    'ContinualLearningEvaluator',
    
    # Configuration
    'get_config',
    'ExperimentConfig',
    
    # Training
    'LwUTrainer',
    'BaselineTrainer',
    'set_seed',
]
