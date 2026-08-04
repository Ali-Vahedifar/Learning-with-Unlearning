#!/usr/bin/env python3
"""
Train Machine Unlearning Experiments

This script runs machine unlearning experiments comparing LwU against baselines.
Supports class-level and sample-level unlearning on RTI, CIFAR-100, and TinyImageNet.
"""

import argparse
import os
import sys
import json
import torch
import numpy as np
from datetime import datetime
import time

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lwu.models.lwu import LwU
from lwu.models.backbones import resnet18, SimpleMLP, get_backbone
from lwu.datasets.continual_datasets import (
    CIFAR100Dataset, TinyImageNetDataset, RTIContinualDataset,
    get_dataset
)
from lwu.baselines.unlearning_methods import (
    SSD, BadTeacher, Amnesiac, UNSIR, retrain_from_scratch
)
from lwu.utils.training import set_seed, train_epoch, evaluate
from lwu.utils.config import get_config
from lwu.evaluation.metrics import (
    UnlearningEvaluator, evaluate_task, compute_kl_divergence,
    measure_execution_time
)


def parse_args():
    parser = argparse.ArgumentParser(
        description='Train Machine Unlearning Experiments'
    )
    
    # Dataset settings
    parser.add_argument('--dataset', type=str, default='cifar100',
                        choices=['cifar100', 'tinyimagenet', 'rti'],
                        help='Dataset to use')
    parser.add_argument('--forget_class', type=int, default=None,
                        help='Class to forget (if None, selected automatically)')
    parser.add_argument('--forget_ratio', type=float, default=0.1,
                        help='Ratio of data to forget (for sample-level)')
    
    # Method settings
    parser.add_argument('--method', type=str, default='lwu',
                        choices=['lwu', 'ssd', 'badteacher', 'amnesiac', 'unsir', 'retrain'],
                        help='Unlearning method to use')
    
    # LwU specific parameters
    parser.add_argument('--tau_r', type=float, default=0.5,
                        help='Retain threshold for LwU')
    parser.add_argument('--tau_f', type=float, default=0.5,
                        help='Forget threshold for LwU')
    
    # SSD specific parameters
    parser.add_argument('--ssd_lambda', type=float, default=1.0,
                        help='SSD dampening factor')
    
    # Bad Teacher specific parameters
    parser.add_argument('--bt_kd_weight', type=float, default=0.5,
                        help='Bad Teacher distillation weight')
    parser.add_argument('--bt_temperature', type=float, default=4.0,
                        help='Bad Teacher temperature')
    
    # UNSIR specific parameters
    parser.add_argument('--unsir_impair_epochs', type=int, default=5,
                        help='UNSIR impair epochs')
    parser.add_argument('--unsir_repair_epochs', type=int, default=5,
                        help='UNSIR repair epochs')
    
    # Training settings
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Batch size')
    parser.add_argument('--num_epochs', type=int, default=100,
                        help='Training epochs for baseline model')
    
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
    
    return parser.parse_args()


def get_dataset_config(args):
    """Get dataset configuration based on arguments."""
    configs = {
        'cifar100': {
            'num_classes': 100,
            'input_size': (3, 32, 32),
            'backbone': 'resnet18',
            'default_forget_class': 42  # 'rocket' class
        },
        'tinyimagenet': {
            'num_classes': 200,
            'input_size': (3, 64, 64),
            'backbone': 'resnet18',
            'default_forget_class': 1  # 'goldfish' class
        },
        'rti': {
            'num_classes': 5,
            'input_size': (25,),
            'backbone': 'mlp',
            'default_forget_class': 2  # 'tapping' class
        }
    }
    return configs[args.dataset]


def train_baseline_model(model, train_loader, val_loader, args):
    """Train a baseline model on full dataset."""
    device = args.device
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.lr,
        betas=(0.9, 0.999),
        eps=1e-8
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )
    
    best_val_acc = 0.0
    best_model_state = None
    patience_counter = 0
    
    for epoch in range(args.num_epochs):
        # Training
        train_loss, train_acc = train_epoch(
            model, train_loader, optimizer, criterion, device
        )
        
        # Validation
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        scheduler.step(val_loss)
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = {k: v.clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{args.num_epochs}: "
                  f"Train Loss={train_loss:.4f}, Train Acc={train_acc:.4f}, "
                  f"Val Acc={val_acc:.4f}")
            
        if patience_counter >= 10:
            print(f"Early stopping at epoch {epoch+1}")
            break
    
    # Restore best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    return model


def create_unlearning_method(args, model, dataset_config):
    """Create unlearning method instance."""
    device = args.device
    
    if args.method == 'lwu':
        return LwU(
            backbone=model,
            num_classes=dataset_config['num_classes'],
            tau_r=args.tau_r,
            tau_f=args.tau_f,
            device=device
        )
    elif args.method == 'ssd':
        return SSD(
            model=model,
            dampening_lambda=args.ssd_lambda,
            device=device
        )
    elif args.method == 'badteacher':
        return BadTeacher(
            model=model,
            num_classes=dataset_config['num_classes'],
            kd_weight=args.bt_kd_weight,
            temperature=args.bt_temperature,
            device=device
        )
    elif args.method == 'amnesiac':
        return Amnesiac(
            model=model,
            device=device
        )
    elif args.method == 'unsir':
        return UNSIR(
            model=model,
            impair_epochs=args.unsir_impair_epochs,
            repair_epochs=args.unsir_repair_epochs,
            device=device
        )
    else:
        return None


def run_single_experiment(args, run_id):
    """Run a single unlearning experiment."""
    set_seed(args.seed + run_id)
    
    dataset_config = get_dataset_config(args)
    device = args.device
    
    # Set forget class
    if args.forget_class is None:
        forget_class = dataset_config['default_forget_class']
    else:
        forget_class = args.forget_class
    
    print(f"\nLoading {args.dataset} dataset...")
    # Unlearning uses a single joint task spanning the full label space.
    dataset_kwargs = {}
    if args.dataset == 'rti':
        dataset_kwargs['num_classes'] = dataset_config['num_classes']

    dataset = get_dataset(
        args.dataset,
        root=args.data_dir,
        num_tasks=1,
        scenario='class',
        **dataset_kwargs
    )
    
    # Get full dataset loaders
    full_train_loader, full_val_loader, full_test_loader = dataset.get_task_loaders(
        0, batch_size=args.batch_size
    )
    
    # Split into forget and retain sets
    forget_loader, retain_loader = dataset.get_forget_retain_loaders(
        forget_classes=[forget_class],
        batch_size=args.batch_size
    )
    
    # Create backbone. Feature-vector datasets (MLP backbone) take their input
    # dimension from the data rather than from a hardcoded config entry.
    backbone_kwargs = {}
    if dataset_config['backbone'] == 'mlp':
        sample, _ = dataset.train_dataset[0]
        backbone_kwargs['input_dim'] = int(np.prod(np.asarray(sample).shape))
    elif dataset_config['backbone'].startswith('resnet'):
        input_size = dataset_config.get('input_size', (3, 32, 32))
        backbone_kwargs['small_input'] = input_size[-1] <= 64

    print(f"Creating backbone model...")
    backbone = get_backbone(
        dataset_config['backbone'],
        num_classes=dataset_config['num_classes'],
        **backbone_kwargs
    ).to(device)
    
    # Train baseline model
    print(f"Training baseline model on full dataset...")
    start_time = time.time()
    baseline_model = train_baseline_model(
        backbone, full_train_loader, full_val_loader, args
    )
    baseline_train_time = time.time() - start_time
    
    # Evaluate baseline
    baseline_forget_acc = evaluate_task(baseline_model, forget_loader, device) * 100
    baseline_retain_acc = evaluate_task(baseline_model, retain_loader, device) * 100
    
    print(f"\nBaseline Model:")
    print(f"  Forget Acc: {baseline_forget_acc:.2f}%")
    print(f"  Retain Acc: {baseline_retain_acc:.2f}%")
    
    results = {
        'baseline': {
            'forget_acc': baseline_forget_acc,
            'retain_acc': baseline_retain_acc,
            'train_time': baseline_train_time
        }
    }
    
    # Apply unlearning method
    print(f"\nApplying {args.method.upper()} unlearning...")
    
    if args.method == 'retrain':
        # Retrain from scratch (gold standard)
        start_time = time.time()
        # The Retrain oracle is built from the same backbone factory and the
        # same construction kwargs as the model being unlearned, so the only
        # difference between them is the data they were trained on.
        retrained_model = retrain_from_scratch(
            model_class=lambda **kw: get_backbone(
                dataset_config['backbone'], **kw
            ),
            retain_loader=retain_loader,
            num_classes=dataset_config['num_classes'],
            device=device,
            num_epochs=args.num_epochs,
            lr=args.lr,
            **backbone_kwargs
        )
        unlearn_time = time.time() - start_time
        unlearned_model = retrained_model
        
    else:
        # Create unlearning method
        method = create_unlearning_method(args, baseline_model, dataset_config)
        
        start_time = time.time()
        
        if args.method == 'lwu':
            method.unlearn(forget_loader, retain_loader)
            unlearned_model = method.backbone
        elif args.method == 'ssd':
            method.unlearn(forget_loader, retain_loader)
            unlearned_model = method.model
        elif args.method == 'badteacher':
            method.unlearn(forget_loader, retain_loader, num_epochs=10)
            unlearned_model = method.model
        elif args.method == 'amnesiac':
            method.unlearn(forget_loader)
            unlearned_model = method.model
        elif args.method == 'unsir':
            method.unlearn(
                forget_loader, 
                retain_loader,
                lr=args.lr * 0.1
            )
            unlearned_model = method.model
            
        unlearn_time = time.time() - start_time
    
    # Evaluate unlearned model
    evaluator = UnlearningEvaluator(unlearned_model, device)
    eval_results = evaluator.evaluate(
        forget_loader=forget_loader,
        retain_loader=retain_loader,
        test_loader=full_test_loader
    )
    
    # Add timing
    eval_results['unlearn_time'] = unlearn_time
    
    # Compute KL divergence if retrained model available
    if args.method != 'retrain':
        # Train retrained model for KL comparison
        print("Training retrained model for KL divergence comparison...")
        retrained_model = retrain_from_scratch(
            model_class=lambda **kw: get_backbone(
                dataset_config['backbone'], **kw
            ),
            retain_loader=retain_loader,
            num_classes=dataset_config['num_classes'],
            device=device,
            num_epochs=args.num_epochs,
            lr=args.lr,
            **backbone_kwargs
        )
        
        kl_div = compute_kl_divergence(
            unlearned_model,
            retrained_model,
            full_test_loader,
            device
        )
        eval_results['kl_divergence'] = kl_div
    else:
        eval_results['kl_divergence'] = 0.0  # Retrained is the reference
    
    results['unlearned'] = eval_results
    
    # Print results
    print(f"\n{args.method.upper()} Results:")
    print(f"  Forget Acc: {eval_results['forget_acc']:.2f}%")
    print(f"  Retain Acc: {eval_results['retain_acc']:.2f}%")
    print(f"  MIA: {eval_results['mia']:.2f}%")
    print(f"  KL Divergence: {eval_results['kl_divergence']:.4f}")
    print(f"  Unlearn Time: {unlearn_time:.2f}s")
    
    return results


def main():
    args = parse_args()
    
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(
        args.output_dir,
        f"unlearn_{args.method}_{args.dataset}_{timestamp}"
    )
    os.makedirs(output_dir, exist_ok=True)
    
    # Save configuration
    config_path = os.path.join(output_dir, 'config.json')
    with open(config_path, 'w') as f:
        json.dump(vars(args), f, indent=2)
    
    # Run experiments
    all_results = []
    
    for run_id in range(args.num_runs):
        print(f"\n{'='*60}")
        print(f"Run {run_id + 1}/{args.num_runs}")
        print(f"{'='*60}")
        
        results = run_single_experiment(args, run_id)
        all_results.append(results)
        
        # Save intermediate results
        run_path = os.path.join(output_dir, f'run_{run_id}.json')
        with open(run_path, 'w') as f:
            json.dump(results, f, indent=2)
    
    # Aggregate results
    metrics_keys = ['forget_acc', 'retain_acc', 'mia', 'kl_divergence', 'unlearn_time']
    aggregated = {}
    
    for key in metrics_keys:
        values = [r['unlearned'][key] for r in all_results if r['unlearned'].get(key) is not None]
        if values:
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
    print(f"Number of runs: {args.num_runs}")
    print("-" * 60)
    
    for key in metrics_keys:
        if key in aggregated:
            mean = aggregated[key]['mean']
            std = aggregated[key]['std']
            print(f"{key}: {mean:.2f} ± {std:.2f}")
    
    print("=" * 60)
    print(f"\nResults saved to: {output_dir}")


if __name__ == '__main__':
    main()
