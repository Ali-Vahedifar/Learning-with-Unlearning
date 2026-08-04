#!/usr/bin/env python3
"""
Run All Experiments

This script orchestrates running all continual learning and machine unlearning
experiments as described in the documentation. It runs experiments across:
- Datasets: RTI, CIFAR-100, TinyImageNet
- Methods: LwU, EWC, SI, LwF, SGD (CL) and LwU, SSD, BadTeacher, UNSIR, Amnesiac (MU)
- Scenarios: Task-IL, Class-IL
"""

import argparse
import os
import sys
import json
import subprocess
from datetime import datetime
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# Experiment configurations
CONTINUAL_LEARNING_CONFIGS = {
    'rti': {
        'num_tasks': 10,
        'lwu': {'tau_r': 0.54, 'tau_f': 0.32},
        'ewc': {'ewc_lambda': 100, 'ewc_gamma': 1.0},
        'si': {'si_c': 1.0, 'si_xi': 1.0},
        'lwf': {'lwf_alpha': 0.5, 'lwf_temperature': 2.0}
    },
    'cifar100': {
        'num_tasks': 10,
        'lwu': {'tau_r': 0.49, 'tau_f': 0.35},
        'ewc': {'ewc_lambda': 25, 'ewc_gamma': 1.0},
        'si': {'si_c': 0.5, 'si_xi': 1.0},
        'lwf': {'lwf_alpha': 0.5, 'lwf_temperature': 2.0}
    },
    'tinyimagenet': {
        'num_tasks': 10,
        'lwu': {'tau_r': 0.49, 'tau_f': 0.43},
        'ewc': {'ewc_lambda': 25, 'ewc_gamma': 1.0},
        'si': {'si_c': 0.5, 'si_xi': 1.0},
        'lwf': {'lwf_alpha': 0.5, 'lwf_temperature': 2.0}
    }
}

UNLEARNING_CONFIGS = {
    'rti': {
        'forget_class': 2,  # 'tapping'
        'lwu': {'tau_r': 0.54, 'tau_f': 0.32},
        'ssd': {'ssd_lambda': 1.0},
        'badteacher': {'bt_kd_weight': 0.5, 'bt_temperature': 4.0},
        'unsir': {'unsir_impair_epochs': 5, 'unsir_repair_epochs': 5}
    },
    'cifar100': {
        'forget_class': 42,  # 'rocket'
        'lwu': {'tau_r': 0.49, 'tau_f': 0.35},
        'ssd': {'ssd_lambda': 1.0},
        'badteacher': {'bt_kd_weight': 0.5, 'bt_temperature': 4.0},
        'unsir': {'unsir_impair_epochs': 5, 'unsir_repair_epochs': 5}
    },
    'tinyimagenet': {
        'forget_class': 1,  # 'goldfish'
        'lwu': {'tau_r': 0.49, 'tau_f': 0.43},
        'ssd': {'ssd_lambda': 1.0},
        'badteacher': {'bt_kd_weight': 0.5, 'bt_temperature': 4.0},
        'unsir': {'unsir_impair_epochs': 5, 'unsir_repair_epochs': 5}
    }
}


def parse_args():
    parser = argparse.ArgumentParser(
        description='Run All Experiments for LwU'
    )
    
    parser.add_argument('--experiment', type=str, default='all',
                        choices=['all', 'continual', 'unlearning'],
                        help='Which experiments to run')
    parser.add_argument('--datasets', nargs='+', 
                        default=['rti', 'cifar100', 'tinyimagenet'],
                        help='Datasets to use')
    parser.add_argument('--methods', nargs='+', default=None,
                        help='Methods to run (default: all)')
    parser.add_argument('--scenarios', nargs='+', default=['task', 'class'],
                        help='Scenarios for continual learning')
    parser.add_argument('--num_runs', type=int, default=10,
                        help='Number of independent runs')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use')
    parser.add_argument('--output_dir', type=str, default='./results',
                        help='Output directory')
    parser.add_argument('--data_dir', type=str, default='./data',
                        help='Data directory')
    parser.add_argument('--dry_run', action='store_true',
                        help='Print commands without running')
    
    return parser.parse_args()


def run_continual_learning_experiments(args):
    """Run all continual learning experiments."""
    
    cl_methods = args.methods or ['lwu', 'ewc', 'si', 'lwf', 'sgd']
    
    experiments = []
    
    for dataset in args.datasets:
        config = CONTINUAL_LEARNING_CONFIGS[dataset]
        
        for scenario in args.scenarios:
            for method in cl_methods:
                exp = {
                    'dataset': dataset,
                    'scenario': scenario,
                    'method': method,
                    'num_tasks': config['num_tasks'],
                    'num_runs': args.num_runs,
                    'device': args.device,
                    'output_dir': args.output_dir,
                    'data_dir': args.data_dir
                }
                
                # Add method-specific parameters
                if method in config:
                    exp.update(config[method])
                    
                experiments.append(exp)
    
    return experiments


def run_unlearning_experiments(args):
    """Run all machine unlearning experiments."""
    
    mu_methods = args.methods or ['lwu', 'ssd', 'badteacher', 'amnesiac', 'unsir', 'retrain']
    
    experiments = []
    
    for dataset in args.datasets:
        config = UNLEARNING_CONFIGS[dataset]
        
        for method in mu_methods:
            exp = {
                'dataset': dataset,
                'method': method,
                'forget_class': config['forget_class'],
                'num_runs': args.num_runs,
                'device': args.device,
                'output_dir': args.output_dir,
                'data_dir': args.data_dir
            }
            
            # Add method-specific parameters
            if method in config:
                exp.update(config[method])
                
            experiments.append(exp)
    
    return experiments


def build_continual_command(exp):
    """Build command for continual learning experiment."""
    script_path = Path(__file__).parent / 'train_continual.py'
    
    cmd = [
        sys.executable, str(script_path),
        '--dataset', exp['dataset'],
        '--scenario', exp['scenario'],
        '--method', exp['method'],
        '--num_tasks', str(exp['num_tasks']),
        '--num_runs', str(exp['num_runs']),
        '--device', exp['device'],
        '--output_dir', exp['output_dir'],
        '--data_dir', exp['data_dir']
    ]
    
    # Add method-specific parameters
    if exp['method'] == 'lwu':
        if 'tau_r' in exp:
            cmd.extend(['--tau_r', str(exp['tau_r'])])
        if 'tau_f' in exp:
            cmd.extend(['--tau_f', str(exp['tau_f'])])
    elif exp['method'] == 'ewc':
        if 'ewc_lambda' in exp:
            cmd.extend(['--ewc_lambda', str(exp['ewc_lambda'])])
        if 'ewc_gamma' in exp:
            cmd.extend(['--ewc_gamma', str(exp['ewc_gamma'])])
    elif exp['method'] == 'si':
        if 'si_c' in exp:
            cmd.extend(['--si_c', str(exp['si_c'])])
        if 'si_xi' in exp:
            cmd.extend(['--si_xi', str(exp['si_xi'])])
    elif exp['method'] == 'lwf':
        if 'lwf_alpha' in exp:
            cmd.extend(['--lwf_alpha', str(exp['lwf_alpha'])])
        if 'lwf_temperature' in exp:
            cmd.extend(['--lwf_temperature', str(exp['lwf_temperature'])])
    
    return cmd


def build_unlearning_command(exp):
    """Build command for unlearning experiment."""
    script_path = Path(__file__).parent / 'train_unlearning.py'
    
    cmd = [
        sys.executable, str(script_path),
        '--dataset', exp['dataset'],
        '--method', exp['method'],
        '--forget_class', str(exp['forget_class']),
        '--num_runs', str(exp['num_runs']),
        '--device', exp['device'],
        '--output_dir', exp['output_dir'],
        '--data_dir', exp['data_dir']
    ]
    
    # Add method-specific parameters
    if exp['method'] == 'lwu':
        if 'tau_r' in exp:
            cmd.extend(['--tau_r', str(exp['tau_r'])])
        if 'tau_f' in exp:
            cmd.extend(['--tau_f', str(exp['tau_f'])])
    elif exp['method'] == 'ssd':
        if 'ssd_lambda' in exp:
            cmd.extend(['--ssd_lambda', str(exp['ssd_lambda'])])
    elif exp['method'] == 'badteacher':
        if 'bt_kd_weight' in exp:
            cmd.extend(['--bt_kd_weight', str(exp['bt_kd_weight'])])
        if 'bt_temperature' in exp:
            cmd.extend(['--bt_temperature', str(exp['bt_temperature'])])
    elif exp['method'] == 'unsir':
        if 'unsir_impair_epochs' in exp:
            cmd.extend(['--unsir_impair_epochs', str(exp['unsir_impair_epochs'])])
        if 'unsir_repair_epochs' in exp:
            cmd.extend(['--unsir_repair_epochs', str(exp['unsir_repair_epochs'])])
    
    return cmd


def main():
    args = parse_args()
    
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_output_dir = os.path.join(args.output_dir, f'experiments_{timestamp}')
    os.makedirs(base_output_dir, exist_ok=True)
    
    # Build experiment list
    experiments = []
    
    if args.experiment in ['all', 'continual']:
        cl_experiments = run_continual_learning_experiments(args)
        for exp in cl_experiments:
            exp['type'] = 'continual'
            exp['output_dir'] = os.path.join(base_output_dir, 'continual_learning')
        experiments.extend(cl_experiments)
        
    if args.experiment in ['all', 'unlearning']:
        mu_experiments = run_unlearning_experiments(args)
        for exp in mu_experiments:
            exp['type'] = 'unlearning'
            exp['output_dir'] = os.path.join(base_output_dir, 'machine_unlearning')
        experiments.extend(mu_experiments)
    
    # Print summary
    print("=" * 60)
    print("EXPERIMENT SUMMARY")
    print("=" * 60)
    print(f"Total experiments: {len(experiments)}")
    print(f"Datasets: {args.datasets}")
    print(f"Runs per experiment: {args.num_runs}")
    print(f"Output directory: {base_output_dir}")
    print("=" * 60)
    
    # Save experiment plan
    plan_path = os.path.join(base_output_dir, 'experiment_plan.json')
    with open(plan_path, 'w') as f:
        json.dump(experiments, f, indent=2)
    
    # Run experiments
    results = []
    
    for i, exp in enumerate(experiments):
        print(f"\n{'='*60}")
        print(f"Experiment {i+1}/{len(experiments)}")
        print(f"Type: {exp['type']}")
        print(f"Dataset: {exp['dataset']}")
        print(f"Method: {exp['method']}")
        if exp['type'] == 'continual':
            print(f"Scenario: {exp['scenario']}-IL")
        print("=" * 60)
        
        # Build command
        if exp['type'] == 'continual':
            cmd = build_continual_command(exp)
        else:
            cmd = build_unlearning_command(exp)
        
        if args.dry_run:
            print(f"Would run: {' '.join(cmd)}")
            continue
        
        # Run experiment
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=True
            )
            print(result.stdout)
            status = 'success'
        except subprocess.CalledProcessError as e:
            print(f"Error: {e}")
            print(f"Stdout: {e.stdout}")
            print(f"Stderr: {e.stderr}")
            status = 'failed'
        
        results.append({
            'experiment': exp,
            'status': status
        })
    
    # Save results summary
    summary_path = os.path.join(base_output_dir, 'run_summary.json')
    with open(summary_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Print final summary
    successful = sum(1 for r in results if r['status'] == 'success')
    failed = sum(1 for r in results if r['status'] == 'failed')
    
    print("\n" + "=" * 60)
    print("FINAL SUMMARY")
    print("=" * 60)
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    print(f"Results saved to: {base_output_dir}")
    print("=" * 60)


if __name__ == '__main__':
    main()
