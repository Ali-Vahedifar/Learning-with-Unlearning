#!/usr/bin/env python3
"""
Train Continual Learning Experiments

This script runs continual learning experiments comparing LwU against baselines.
Supports Task-IL and Class-IL scenarios on RTI, CIFAR-100, and TinyImageNet.
"""

import argparse
import os
import sys
import json
import torch
import numpy as np
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lwu.models.lwu import LwU
from lwu.models.backbones import resnet18, SimpleMLP, get_backbone
from lwu.datasets.continual_datasets import (
    CIFAR100Dataset, TinyImageNetDataset, RTIContinualDataset,
    get_dataset
)
from lwu.baselines.continual_methods import EWC, SI, LwF
from lwu.utils.training import LwUTrainer, BaselineTrainer, set_seed
from lwu.utils.config import get_config, save_config, load_config, DatasetConfig
from lwu.evaluation.metrics import ContinualLearningEvaluator


def apply_config_file(args, parser):
    """
    Overlay values from a YAML config file onto the parsed arguments.

    Precedence: explicitly passed command-line flags > config file > argparse
    defaults. This means a config file sets the experiment, and a flag on the
    command line still overrides it for one-off runs.

    Every field written here is echoed into the run's config.json, so each
    reported number traces back to the exact settings that produced it.
    """
    if not getattr(args, 'config', None):
        return args

    config = load_config(args.config)

    # Which options did the user actually type on the command line?
    explicit = set()
    for action in parser._actions:
        for opt in action.option_strings:
            if opt in sys.argv[1:]:
                explicit.add(action.dest)

    flat = {}
    for section in ('lwu', 'training', 'dataset', 'model'):
        section_obj = getattr(config, section, None)
        if section_obj is None:
            continue
        for field, value in vars(section_obj).items():
            flat[field] = value

    applied = {}
    for field, value in flat.items():
        if hasattr(args, field) and field not in explicit:
            setattr(args, field, value)
            applied[field] = value

    args._config_applied = applied
    print(f"Loaded configuration from {args.config}")
    for field, value in sorted(applied.items()):
        print(f"  {field}: {value}")

    return args


def parse_args():
    parser = argparse.ArgumentParser(
        description='Train Continual Learning Experiments'
    )

    parser.add_argument('--config', type=str, default=None,
                        help='Path to a YAML config file (e.g. configs/cifar100.yaml). '
                             'Command-line flags override values from the file.')

    # Dataset settings
    parser.add_argument('--dataset', type=str, default='cifar100',
                        choices=['cifar100', 'tinyimagenet', 'rti',
                                 'imagenet1k', 'imagenet-r', 'imagenet-a',
                                 'cifar100-vit'],
                        help='Dataset to use')
    parser.add_argument('--num_tasks', type=int, default=10,
                        help='Number of tasks')
    parser.add_argument('--scenario', type=str, default='task',
                        choices=['task', 'class'],
                        help='Evaluation scenario (task-IL or class-IL)')
    
    # Method settings
    parser.add_argument('--method', type=str, default='lwu',
                        choices=['lwu', 'ewc', 'si', 'lwf', 'sgd'],
                        help='Method to use')
    
    # LwU specific parameters
    parser.add_argument('--tau_r', type=float, default=0.5,
                        help='Retain threshold for LwU')
    parser.add_argument('--tau_f', type=float, default=0.5,
                        help='Forget threshold for LwU')
    parser.add_argument('--momentum_decay', type=float, default=0.95,
                        help='Momentum decay for TTU')
    parser.add_argument('--adaptation_rate', type=float, default=0.01,
                        help='Adaptation rate for TTU')
    parser.add_argument('--surprise_threshold', type=float, default=0.05,
                        help='Surprise threshold for TTU')
    parser.add_argument('--stabilization_decay', type=float, default=0.3,
                        help='Stabilization decay for TTU')
    
    # EWC specific parameters
    parser.add_argument('--ewc_lambda', type=float, default=100.0,
                        help='EWC regularization strength')
    parser.add_argument('--ewc_gamma', type=float, default=1.0,
                        help='EWC decay factor')
    
    # SI specific parameters
    parser.add_argument('--si_c', type=float, default=0.5,
                        help='SI regularization coefficient')
    parser.add_argument('--si_xi', type=float, default=1.0,
                        help='SI dampening factor')
    
    # LwF specific parameters
    parser.add_argument('--lwf_alpha', type=float, default=0.5,
                        help='LwF distillation weight')
    parser.add_argument('--lwf_temperature', type=float, default=2.0,
                        help='LwF distillation temperature')
    
    # Training settings
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Batch size')
    parser.add_argument('--num_epochs', type=int, default=200,
                        help='Maximum epochs per task')
    parser.add_argument('--patience', type=int, default=10,
                        help='Early stopping patience')
    parser.add_argument('--weight_decay', type=float, default=0.0,
                        help='Weight decay')
    
    # Zone C (orthogonal conflict resolution)
    parser.add_argument('--eta_c', type=float, default=0.005,
                        help='Zone C conflict resolution learning rate')
    parser.add_argument('--n_c', type=int, default=10,
                        help='Number of Zone C projected gradient ascent steps')
    parser.add_argument('--delta_c', type=float, default=0.5,
                        help='Zone C displacement budget')

    # LoRA settings (ViT-B/16 backbone)
    parser.add_argument('--lora_rank', type=int, default=8,
                        help='LoRA rank r')
    parser.add_argument('--lora_alpha', type=float, default=16.0,
                        help='LoRA scaling alpha')
    parser.add_argument('--lora_dropout', type=float, default=0.0,
                        help='LoRA dropout')
    parser.add_argument('--pretrained', action='store_true', default=True,
                        help='Use ImageNet-21K pretrained ViT weights')

    # Experiment settings
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--num_runs', type=int, default=1,
                        help='Number of independent runs')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use')
    parser.add_argument('--output_dir', type=str, default='./results',
                        help='Output directory')
    parser.add_argument('--data_dir', type=str, default='./data',
                        help='Data directory')
    
    args = parser.parse_args()
    return apply_config_file(args, parser)


def get_dataset_config(args):
    """Get dataset configuration based on arguments."""
    if args.dataset == 'cifar100':
        return {
            'num_classes': 100,
            'input_size': (3, 32, 32),
            'backbone': 'resnet18'
        }
    elif args.dataset == 'tinyimagenet':
        return {
            'num_classes': 200,
            'input_size': (3, 64, 64),
            'backbone': 'resnet18'
        }
    elif args.dataset == 'rti':
        return {
            'num_classes': 5,
            'input_size': (25,),  # Feature dimension
            'backbone': 'mlp'
        }
    elif args.dataset == 'imagenet1k':
        return {
            'num_classes': 1000,
            'input_size': (3, 224, 224),
            'backbone': 'resnet18'
        }
    elif args.dataset == 'imagenet-r':
        return {
            'num_classes': 200,
            'input_size': (3, 224, 224),
            'backbone': 'vit_b16_lora'
        }
    elif args.dataset == 'imagenet-a':
        return {
            'num_classes': 200,
            'input_size': (3, 224, 224),
            'backbone': 'vit_b16_lora'
        }
    elif args.dataset == 'cifar100-vit':
        return {
            'num_classes': 100,
            'input_size': (3, 224, 224),
            'backbone': 'vit_b16_lora'
        }
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")


def create_model(args, dataset_config):
    """Create model based on method and dataset."""
    device = args.device
    
    # Create backbone
    backbone_name = dataset_config['backbone']
    backbone_kwargs = {}

    if backbone_name == 'vit_b16_lora':
        backbone_kwargs.update(
            lora_rank=args.lora_rank,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            pretrained=args.pretrained
        )
    elif backbone_name == 'mlp':
        input_size = dataset_config.get('input_size', None)
        if input_size is not None:
            backbone_kwargs['input_dim'] = int(np.prod(input_size))
    elif backbone_name.startswith('resnet'):
        # 224x224 inputs use the standard stem; 32x32 / 64x64 use the small stem
        input_size = dataset_config.get('input_size', (3, 32, 32))
        backbone_kwargs['small_input'] = input_size[-1] < 128

    backbone = get_backbone(
        backbone_name,
        num_classes=dataset_config['num_classes'],
        **backbone_kwargs
    ).to(device)
    
    if args.method == 'lwu':
        model = LwU(
            backbone=backbone,
            num_classes=dataset_config['num_classes'],
            tau_r=args.tau_r,
            tau_f=args.tau_f,
            momentum_decay=args.momentum_decay,
            adaptation_rate=args.adaptation_rate,
            surprise_threshold=args.surprise_threshold,
            stabilization_decay=args.stabilization_decay,
            device=device
        )
    else:
        model = backbone
        
    return model


def create_trainer(args, model, dataset_config):
    """Create appropriate trainer based on method."""
    device = args.device
    
    if args.method == 'lwu':
        trainer = LwUTrainer(
            model=model,
            device=device,
            lr=args.lr,
            weight_decay=args.weight_decay,
            patience=args.patience
        )
    else:
        # Create baseline method
        if args.method == 'ewc':
            method = EWC(
                model=model,
                ewc_lambda=args.ewc_lambda,
                gamma=args.ewc_gamma,
                device=device
            )
        elif args.method == 'si':
            method = SI(
                model=model,
                si_c=args.si_c,
                xi=args.si_xi,
                device=device
            )
        elif args.method == 'lwf':
            method = LwF(
                model=model,
                alpha=args.lwf_alpha,
                temperature=args.lwf_temperature,
                device=device
            )
        else:  # SGD
            method = None
            
        trainer = BaselineTrainer(
            model=model,
            method=method,
            device=device,
            lr=args.lr,
            weight_decay=args.weight_decay,
            patience=args.patience
        )
        
    return trainer


def run_single_experiment(args, run_id):
    """Run a single experiment with given seed."""
    # Set seed
    set_seed(args.seed + run_id)
    
    # Get dataset configuration
    dataset_config = get_dataset_config(args)
    
    # Load dataset
    print(f"\nLoading {args.dataset} dataset...")
    dataset = get_dataset(
        args.dataset,
        root=args.data_dir,
        num_tasks=args.num_tasks,
        scenario=args.scenario
    )
    
    # Reconcile the model configuration with the dataset that was actually
    # constructed. The class count follows from num_tasks, and for
    # feature-vector datasets the input dimension follows from the feature
    # extractor, so neither should be hardcoded in the config table.
    if getattr(dataset, 'num_classes', None):
        dataset_config['num_classes'] = dataset.num_classes

    if dataset_config.get('backbone') == 'mlp':
        sample, _ = dataset.train_dataset[0]
        feat_dim = int(np.prod(np.asarray(sample).shape))
        if feat_dim != int(np.prod(dataset_config['input_size'])):
            print(
                f"Input dimension from data ({feat_dim}) differs from the "
                f"config default ({int(np.prod(dataset_config['input_size']))}); "
                f"using the value from the data."
            )
        dataset_config['input_size'] = (feat_dim,)

    # Create model and trainer
    print(f"Creating {args.method.upper()} model...")
    model = create_model(args, dataset_config)
    trainer = create_trainer(args, model, dataset_config)
    
    # Run continual learning
    print(f"\nStarting continual learning experiment...")
    print(f"Method: {args.method.upper()}")
    print(f"Dataset: {args.dataset}")
    print(f"Scenario: {args.scenario}-IL")
    print(f"Tasks: {args.num_tasks}")
    print(f"Run: {run_id + 1}/{args.num_runs}")
    print("=" * 60)
    
    results = trainer.continual_learning(
        dataset=dataset,
        num_tasks=args.num_tasks,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size
    )
    
    return results


def main():
    args = parse_args()
    
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(
        args.output_dir,
        f"{args.method}_{args.dataset}_{args.scenario}_{timestamp}"
    )
    os.makedirs(output_dir, exist_ok=True)
    
    # Save configuration
    config_path = os.path.join(output_dir, 'config.json')
    with open(config_path, 'w') as f:
        json.dump(vars(args), f, indent=2)
    
    # Run experiments
    all_results = []
    
    for run_id in range(args.num_runs):
        results = run_single_experiment(args, run_id)
        all_results.append(results)
        
        # Save intermediate results
        run_path = os.path.join(output_dir, f'run_{run_id}.json')
        with open(run_path, 'w') as f:
            # Convert numpy arrays to lists for JSON serialization
            save_results = {
                'metrics': results['metrics'],
                'accuracy_matrix': results['accuracy_matrix'].tolist()
            }
            json.dump(save_results, f, indent=2)
    
    # Aggregate results across runs
    metrics_keys = ['ACC', 'BWT', 'FWT', 'PS']
    aggregated = {}
    
    for key in metrics_keys:
        values = [r['metrics'][key] for r in all_results]
        aggregated[key] = {
            'mean': float(np.mean(values)),
            'std': float(np.std(values))
        }
    
    # Save aggregated results
    agg_path = os.path.join(output_dir, 'aggregated_results.json')
    with open(agg_path, 'w') as f:
        json.dump(aggregated, f, indent=2)
    
    # Print final results
    print("\n" + "=" * 60)
    print("FINAL RESULTS")
    print("=" * 60)
    print(f"Method: {args.method.upper()}")
    print(f"Dataset: {args.dataset}")
    print(f"Scenario: {args.scenario}-IL")
    print(f"Number of runs: {args.num_runs}")
    print("-" * 60)
    
    for key in metrics_keys:
        mean = aggregated[key]['mean']
        std = aggregated[key]['std']
        print(f"{key}: {mean:.2f} ± {std:.2f}")
        
    print("=" * 60)
    print(f"\nResults saved to: {output_dir}")


if __name__ == '__main__':
    main()
