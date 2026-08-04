"""
Configuration for LwU Experiments

Default hyperparameters and experimental settings.
"""

from dataclasses import dataclass, field
from typing import List, Optional
import yaml
import os


@dataclass
class ModelConfig:
    """Model configuration."""
    backbone: str = 'resnet18'
    num_classes: int = 100
    input_dim: int = 10  # For MLP (RTI dataset)
    hidden_dims: List[int] = field(default_factory=lambda: [256, 128, 64])
    small_input: bool = True  # Use smaller conv for CIFAR/TinyImageNet


@dataclass
class LwUConfig:
    """LwU-specific configuration."""
    tau_r: float = 0.5  # Retain threshold
    tau_f: float = 0.5  # Forget threshold
    momentum_decay: float = 0.95  # varphi for TTU
    adaptation_rate: float = 0.01  # lambda for TTU
    surprise_threshold: float = 0.05  # tau_s for TTU
    stabilization_decay: float = 0.3  # alpha for TTU


@dataclass
class TrainingConfig:
    """Training configuration."""
    lr: float = 0.001
    weight_decay: float = 0.0
    batch_size: int = 64
    num_epochs: int = 200
    patience: int = 10
    seed: int = 42


@dataclass
class DatasetConfig:
    """Dataset configuration."""
    name: str = 'cifar100'
    root: str = './data'
    num_tasks: int = 10
    val_split: float = 0.1


@dataclass
class EWCConfig:
    """EWC-specific configuration."""
    ewc_lambda: float = 100.0
    gamma: float = 1.0


@dataclass
class SIConfig:
    """SI-specific configuration."""
    si_c: float = 1.0
    xi: float = 1.0


@dataclass
class LwFConfig:
    """LwF-specific configuration."""
    temperature: float = 2.0
    alpha: float = 0.5


@dataclass
class ExperimentConfig:
    """Complete experiment configuration."""
    model: ModelConfig = field(default_factory=ModelConfig)
    lwu: LwUConfig = field(default_factory=LwUConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    ewc: EWCConfig = field(default_factory=EWCConfig)
    si: SIConfig = field(default_factory=SIConfig)
    lwf: LwFConfig = field(default_factory=LwFConfig)
    
    # General settings
    device: str = 'cuda'
    output_dir: str = './results'
    experiment_name: str = 'lwu_experiment'
    num_runs: int = 10


# Dataset-specific optimal hyperparameters
OPTIMAL_HYPERPARAMETERS = {
    'rti': {
        'tau_r': 0.54,
        'tau_f': 0.32,
        'lr': 0.001,
        'num_tasks': 5
    },
    'cifar100': {
        'tau_r': 0.49,
        'tau_f': 0.35,
        'lr': 0.001,
        'num_tasks': 10
    },
    'tinyimagenet': {
        'tau_r': 0.49,
        'tau_f': 0.43,
        'lr': 0.001,
        'num_tasks': 10
    }
}


def get_config(dataset: str = 'cifar100') -> ExperimentConfig:
    """
    Get configuration with dataset-specific optimal hyperparameters.
    
    Args:
        dataset: Dataset name ('cifar100', 'tinyimagenet', 'rti')
        
    Returns:
        Experiment configuration
    """
    config = ExperimentConfig()
    
    config.dataset.name = dataset
    
    if dataset.lower() in OPTIMAL_HYPERPARAMETERS:
        opt = OPTIMAL_HYPERPARAMETERS[dataset.lower()]
        config.lwu.tau_r = opt['tau_r']
        config.lwu.tau_f = opt['tau_f']
        config.training.lr = opt['lr']
        config.dataset.num_tasks = opt['num_tasks']
        
    # Dataset-specific model settings
    if dataset.lower() == 'rti':
        config.model.backbone = 'mlp'
        config.model.num_classes = 5
        config.model.input_dim = 10
    elif dataset.lower() == 'cifar100':
        config.model.backbone = 'resnet18'
        config.model.num_classes = 100
    elif dataset.lower() == 'tinyimagenet':
        config.model.backbone = 'resnet18'
        config.model.num_classes = 200
        
    return config


def save_config(config: ExperimentConfig, path: str):
    """Save configuration to YAML file."""
    config_dict = {
        'model': {
            'backbone': config.model.backbone,
            'num_classes': config.model.num_classes,
            'input_dim': config.model.input_dim,
            'hidden_dims': config.model.hidden_dims,
            'small_input': config.model.small_input
        },
        'lwu': {
            'tau_r': config.lwu.tau_r,
            'tau_f': config.lwu.tau_f,
            'momentum_decay': config.lwu.momentum_decay,
            'adaptation_rate': config.lwu.adaptation_rate,
            'surprise_threshold': config.lwu.surprise_threshold,
            'stabilization_decay': config.lwu.stabilization_decay
        },
        'training': {
            'lr': config.training.lr,
            'weight_decay': config.training.weight_decay,
            'batch_size': config.training.batch_size,
            'num_epochs': config.training.num_epochs,
            'patience': config.training.patience,
            'seed': config.training.seed
        },
        'dataset': {
            'name': config.dataset.name,
            'root': config.dataset.root,
            'num_tasks': config.dataset.num_tasks,
            'val_split': config.dataset.val_split
        },
        'device': config.device,
        'output_dir': config.output_dir,
        'experiment_name': config.experiment_name,
        'num_runs': config.num_runs
    }
    
    os.makedirs(os.path.dirname(path), exist_ok=True)
    
    with open(path, 'w') as f:
        yaml.dump(config_dict, f, default_flow_style=False)


def load_config(path: str) -> ExperimentConfig:
    """Load configuration from YAML file."""
    with open(path, 'r') as f:
        config_dict = yaml.safe_load(f)
        
    config = ExperimentConfig()
    
    if 'model' in config_dict:
        config.model.backbone = config_dict['model'].get('backbone', 'resnet18')
        config.model.num_classes = config_dict['model'].get('num_classes', 100)
        config.model.input_dim = config_dict['model'].get('input_dim', 10)
        config.model.hidden_dims = config_dict['model'].get('hidden_dims', [256, 128, 64])
        config.model.small_input = config_dict['model'].get('small_input', True)
        
    if 'lwu' in config_dict:
        config.lwu.tau_r = config_dict['lwu'].get('tau_r', 0.5)
        config.lwu.tau_f = config_dict['lwu'].get('tau_f', 0.5)
        config.lwu.momentum_decay = config_dict['lwu'].get('momentum_decay', 0.95)
        config.lwu.adaptation_rate = config_dict['lwu'].get('adaptation_rate', 0.01)
        config.lwu.surprise_threshold = config_dict['lwu'].get('surprise_threshold', 0.05)
        config.lwu.stabilization_decay = config_dict['lwu'].get('stabilization_decay', 0.3)
        
    if 'training' in config_dict:
        config.training.lr = config_dict['training'].get('lr', 0.001)
        config.training.weight_decay = config_dict['training'].get('weight_decay', 0.0)
        config.training.batch_size = config_dict['training'].get('batch_size', 64)
        config.training.num_epochs = config_dict['training'].get('num_epochs', 200)
        config.training.patience = config_dict['training'].get('patience', 10)
        config.training.seed = config_dict['training'].get('seed', 42)
        
    if 'dataset' in config_dict:
        config.dataset.name = config_dict['dataset'].get('name', 'cifar100')
        config.dataset.root = config_dict['dataset'].get('root', './data')
        config.dataset.num_tasks = config_dict['dataset'].get('num_tasks', 10)
        config.dataset.val_split = config_dict['dataset'].get('val_split', 0.1)
        
    config.device = config_dict.get('device', 'cuda')
    config.output_dir = config_dict.get('output_dir', './results')
    config.experiment_name = config_dict.get('experiment_name', 'lwu_experiment')
    config.num_runs = config_dict.get('num_runs', 10)
    
    return config
