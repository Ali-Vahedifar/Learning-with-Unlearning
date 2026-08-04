#!/usr/bin/env python3
"""
Visualization Utilities for LwU Experiments

Creates publication-quality plots for:
- Accuracy evolution across tasks
- Zone distribution analysis
- Radar plots comparing methods
- Hyperparameter sensitivity analysis
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from typing import Dict, List, Optional, Tuple
import argparse

# Set publication-quality defaults
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 10,
    'axes.labelsize': 12,
    'axes.titlesize': 12,
    'legend.fontsize': 10,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'figure.figsize': (8, 6),
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'axes.grid': True,
    'grid.alpha': 0.3
})

# Color scheme
COLORS = {
    'lwu': '#2E86AB',
    'ewc': '#E94F37',
    'si': '#F39C12',
    'lwf': '#27AE60',
    'sgd': '#8E44AD',
    'wsn': '#1ABC9C',
    'nfl': '#E74C3C',
    'joint': '#34495E'
}

ZONE_COLORS = {
    'A': '#3498DB',  # Blue - Safe Retain
    'B': '#E74C3C',  # Red - Pure Forget
    'C': '#F39C12',  # Orange - Conflict
    'D': '#2ECC71'   # Green - Plastic
}


def plot_accuracy_evolution(
    results: Dict[str, Dict],
    num_tasks: int,
    output_path: str = None,
    scenario: str = 'Class-IL',
    dataset: str = 'CIFAR-100'
):
    """
    Plot accuracy evolution across tasks for multiple methods.
    
    Args:
        results: Dictionary mapping method names to accuracy histories
        num_tasks: Number of tasks
        output_path: Path to save figure
        scenario: 'Task-IL' or 'Class-IL'
        dataset: Dataset name
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    
    tasks = list(range(1, num_tasks + 1))
    
    for method, data in results.items():
        color = COLORS.get(method.lower(), '#333333')
        acc_history = data.get('accuracy_history', data.get('acc', []))
        
        if isinstance(acc_history[0], (list, tuple)):
            acc_history = [np.mean(a) for a in acc_history]
            
        ax.plot(tasks, acc_history[:num_tasks], 
                marker='o', markersize=5, linewidth=2,
                color=color, label=method.upper())
    
    ax.set_xlabel('Task')
    ax.set_ylabel('Average Accuracy (%)')
    ax.set_title(f'{scenario} Accuracy Evolution - {dataset}')
    ax.set_xticks(tasks)
    ax.set_xlim(0.5, num_tasks + 0.5)
    ax.set_ylim(0, 100)
    ax.legend(loc='lower left', framealpha=0.9)
    ax.grid(True, alpha=0.3)
    
    if output_path:
        plt.savefig(output_path)
        print(f"Saved accuracy plot to {output_path}")
    
    plt.close()
    return fig


def plot_zone_distribution(
    zone_stats: Dict[int, Dict[str, float]],
    output_path: str = None,
    dataset: str = 'Dataset'
):
    """
    Plot zone distribution evolution across tasks.
    
    Args:
        zone_stats: Dictionary mapping task_id to zone percentages
        output_path: Path to save figure
        dataset: Dataset name
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    tasks = sorted(zone_stats.keys())
    zones = ['D', 'A', 'B', 'C']  # Order for stacking
    
    bottoms = np.zeros(len(tasks))
    
    for zone in zones:
        values = [zone_stats[t].get(zone, 0) for t in tasks]
        ax.bar(tasks, values, bottom=bottoms, 
               color=ZONE_COLORS[zone], label=f'Zone {zone}', width=0.8)
        bottoms += values
    
    ax.set_xlabel('Task')
    ax.set_ylabel('Parameter Proportion (%)')
    ax.set_title(f'Zone Distribution Evolution - {dataset}')
    ax.set_xticks(tasks)
    ax.set_ylim(0, 100)
    ax.legend(loc='upper right', framealpha=0.9)
    
    if output_path:
        plt.savefig(output_path)
        print(f"Saved zone distribution plot to {output_path}")
    
    plt.close()
    return fig


def plot_radar_comparison(
    results: Dict[str, Dict[str, float]],
    metrics: List[str] = ['ACC', 'BWT', 'FWT', 'PS'],
    output_path: str = None,
    title: str = 'Method Comparison'
):
    """
    Create radar plot comparing methods across metrics.
    
    Args:
        results: Dictionary mapping method names to metric values
        metrics: List of metrics to compare
        output_path: Path to save figure
        title: Plot title
    """
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    
    # Number of metrics
    N = len(metrics)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]  # Close the polygon
    
    for method, values in results.items():
        color = COLORS.get(method.lower(), '#333333')
        
        # Normalize values to [0, 1] range
        vals = []
        for metric in metrics:
            val = values.get(metric, 0)
            # Normalize based on metric (BWT can be negative)
            if metric == 'BWT':
                val = (val + 100) / 200  # Map [-100, 100] to [0, 1]
            elif metric == 'FWT':
                val = (val + 100) / 200
            else:
                val = val / 100
            vals.append(max(0, min(1, val)))
        
        vals += vals[:1]  # Close the polygon
        
        ax.plot(angles, vals, 'o-', linewidth=2, color=color, label=method.upper())
        ax.fill(angles, vals, alpha=0.15, color=color)
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metrics)
    ax.set_ylim(0, 1)
    ax.set_title(title, pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
    
    if output_path:
        plt.savefig(output_path)
        print(f"Saved radar plot to {output_path}")
    
    plt.close()
    return fig


def plot_hyperparameter_sensitivity(
    tau_r_values: np.ndarray,
    tau_f_values: np.ndarray,
    accuracy_grid: np.ndarray,
    output_path: str = None,
    dataset: str = 'Dataset'
):
    """
    Create contour plot for hyperparameter sensitivity.
    
    Args:
        tau_r_values: Array of tau_r values
        tau_f_values: Array of tau_f values
        accuracy_grid: 2D array of accuracies (tau_r x tau_f)
        output_path: Path to save figure
        dataset: Dataset name
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Create meshgrid
    TR, TF = np.meshgrid(tau_r_values, tau_f_values)
    
    # Contour plot
    contour = ax.contourf(TR, TF, accuracy_grid.T, levels=20, cmap='viridis')
    plt.colorbar(contour, ax=ax, label='Accuracy (%)')
    
    # Mark optimal point
    max_idx = np.unravel_index(accuracy_grid.argmax(), accuracy_grid.shape)
    optimal_tau_r = tau_r_values[max_idx[0]]
    optimal_tau_f = tau_f_values[max_idx[1]]
    ax.scatter(optimal_tau_r, optimal_tau_f, c='red', s=100, marker='*',
               label=f'Optimal: ({optimal_tau_r:.2f}, {optimal_tau_f:.2f})')
    
    ax.set_xlabel(r'$\tau_r$ (Retain Threshold)')
    ax.set_ylabel(r'$\tau_f$ (Forget Threshold)')
    ax.set_title(f'Hyperparameter Sensitivity - {dataset}')
    ax.legend(loc='upper right')
    
    if output_path:
        plt.savefig(output_path)
        print(f"Saved sensitivity plot to {output_path}")
    
    plt.close()
    return fig


def plot_unlearning_comparison(
    results: Dict[str, Dict[str, float]],
    output_path: str = None,
    dataset: str = 'Dataset'
):
    """
    Create bar plot comparing unlearning methods.
    
    Args:
        results: Dictionary mapping method names to metrics
        output_path: Path to save figure
        dataset: Dataset name
    """
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    
    methods = list(results.keys())
    x = np.arange(len(methods))
    width = 0.6
    
    # Retain accuracy
    retain_accs = [results[m].get('retain_acc', 0) for m in methods]
    axes[0].bar(x, retain_accs, width, color='#2E86AB')
    axes[0].set_ylabel('Retain Accuracy (%)')
    axes[0].set_title('Utility Preservation')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(methods, rotation=45, ha='right')
    axes[0].set_ylim(0, 100)
    
    # Forget accuracy
    forget_accs = [results[m].get('forget_acc', 0) for m in methods]
    axes[1].bar(x, forget_accs, width, color='#E94F37')
    axes[1].set_ylabel('Forget Accuracy (%)')
    axes[1].set_title('Forgetting Effectiveness')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(methods, rotation=45, ha='right')
    axes[1].set_ylim(0, 100)
    
    # MIA
    mia_accs = [results[m].get('mia', 50) for m in methods]
    axes[2].bar(x, mia_accs, width, color='#F39C12')
    axes[2].axhline(y=50, color='red', linestyle='--', label='Random (50%)')
    axes[2].set_ylabel('MIA Success Rate (%)')
    axes[2].set_title('Privacy Protection')
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(methods, rotation=45, ha='right')
    axes[2].set_ylim(0, 100)
    axes[2].legend()
    
    fig.suptitle(f'Machine Unlearning Comparison - {dataset}', fontsize=14)
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path)
        print(f"Saved unlearning comparison to {output_path}")
    
    plt.close()
    return fig


def plot_accuracy_matrix(
    matrix: np.ndarray,
    output_path: str = None,
    title: str = 'Accuracy Matrix'
):
    """
    Plot accuracy matrix heatmap.
    
    Args:
        matrix: Accuracy matrix (num_tasks x num_tasks)
        output_path: Path to save figure
        title: Plot title
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    
    im = ax.imshow(matrix, cmap='Blues', vmin=0, vmax=100)
    plt.colorbar(im, ax=ax, label='Accuracy (%)')
    
    num_tasks = matrix.shape[0]
    ax.set_xticks(range(num_tasks))
    ax.set_yticks(range(num_tasks))
    ax.set_xticklabels([f'T{i+1}' for i in range(num_tasks)])
    ax.set_yticklabels([f'After T{i+1}' for i in range(num_tasks)])
    
    # Add text annotations
    for i in range(num_tasks):
        for j in range(num_tasks):
            if j <= i:
                text = ax.text(j, i, f'{matrix[i, j]:.1f}',
                             ha='center', va='center', fontsize=8)
    
    ax.set_xlabel('Task')
    ax.set_ylabel('Evaluation Point')
    ax.set_title(title)
    
    if output_path:
        plt.savefig(output_path)
        print(f"Saved accuracy matrix to {output_path}")
    
    plt.close()
    return fig


def create_results_summary_table(
    results: Dict[str, Dict[str, float]],
    output_path: str = None
) -> str:
    """
    Create LaTeX table of results.
    
    Args:
        results: Dictionary mapping method names to metrics
        output_path: Path to save table
        
    Returns:
        LaTeX table string
    """
    methods = list(results.keys())
    metrics = ['ACC', 'BWT', 'FWT', 'PS']
    
    # Find best values
    best = {m: max(results[method].get(m, -float('inf')) for method in methods) 
            for m in metrics}
    
    # Build table
    lines = [
        r'\begin{tabular}{l' + 'c' * len(metrics) + '}',
        r'\toprule',
        r'Method & ' + ' & '.join(metrics) + r' \\',
        r'\midrule'
    ]
    
    for method in methods:
        row = [method.upper()]
        for metric in metrics:
            val = results[method].get(metric, 0)
            std = results[method].get(f'{metric}_std', 0)
            
            if val == best[metric]:
                row.append(rf'\textbf{{{val:.2f} $\pm$ {std:.2f}}}')
            else:
                row.append(f'{val:.2f} $\\pm$ {std:.2f}')
        
        lines.append(' & '.join(row) + r' \\')
    
    lines.extend([r'\bottomrule', r'\end{tabular}'])
    
    table = '\n'.join(lines)
    
    if output_path:
        with open(output_path, 'w') as f:
            f.write(table)
        print(f"Saved LaTeX table to {output_path}")
    
    return table


def main():
    """Generate example visualizations."""
    parser = argparse.ArgumentParser(description='Generate visualizations')
    parser.add_argument('--results_dir', type=str, default='./results',
                        help='Results directory')
    parser.add_argument('--output_dir', type=str, default='./figures',
                        help='Output directory for figures')
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Example: Generate plots with dummy data
    num_tasks = 10
    
    # Dummy accuracy evolution data
    results = {
        'LwU': {'acc': [95 - i * 0.5 for i in range(num_tasks)]},
        'EWC': {'acc': [90 - i * 2 for i in range(num_tasks)]},
        'SI': {'acc': [88 - i * 2.5 for i in range(num_tasks)]},
        'LwF': {'acc': [85 - i * 3 for i in range(num_tasks)]},
        'SGD': {'acc': [80 - i * 5 for i in range(num_tasks)]}
    }
    
    plot_accuracy_evolution(
        results, num_tasks,
        output_path=os.path.join(args.output_dir, 'accuracy_evolution.png'),
        scenario='Class-IL',
        dataset='CIFAR-100'
    )
    
    # Dummy zone distribution
    zone_stats = {
        i: {'A': i * 1.5, 'B': 2, 'C': 1, 'D': 95 - i * 1.5}
        for i in range(1, num_tasks + 1)
    }
    
    plot_zone_distribution(
        zone_stats,
        output_path=os.path.join(args.output_dir, 'zone_distribution.png'),
        dataset='CIFAR-100'
    )
    
    # Dummy radar data
    radar_results = {
        'LwU': {'ACC': 80, 'BWT': 0, 'FWT': 8, 'PS': 70},
        'EWC': {'ACC': 50, 'BWT': -10, 'FWT': -4, 'PS': 40},
        'SI': {'ACC': 48, 'BWT': -12, 'FWT': -5, 'PS': 35}
    }
    
    plot_radar_comparison(
        radar_results,
        output_path=os.path.join(args.output_dir, 'radar_comparison.png'),
        title='Continual Learning Method Comparison'
    )
    
    print(f"\nExample figures saved to {args.output_dir}")


if __name__ == '__main__':
    main()
