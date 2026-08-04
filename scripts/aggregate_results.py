#!/usr/bin/env python3
"""
Aggregate per-seed results into a summary table.

Walks a results directory for `run_*.json` / `*_seed*.json` files written by
`train_continual.py`, collects ACC / BWT / FWT / PS per seed, and emits both a
per-seed CSV and a mean +/- std summary. Reporting per-seed values alongside the
aggregate lets every table cell be traced back to the run that produced it.

Usage:
    python scripts/aggregate_results.py --input results/table1 \
        --output results/table1/table1_summary.csv
"""

import argparse
import csv
import json
import os
import re
from collections import defaultdict
from typing import Dict, List

import numpy as np


METRICS = ['ACC', 'BWT', 'FWT', 'PS']


def parse_run_name(path: str, input_dir: str = '') -> Dict[str, str]:
    """
    Extract experiment descriptors from a results path.

    `train_continual.py` writes `run_<id>.json` inside a per-condition
    `--output_dir`, so the condition name lives in the parent directory. Bare
    filenames such as `lwu_class_10t_seed3.json` are also supported.
    """
    stem = os.path.splitext(os.path.basename(path))[0]
    parent = os.path.basename(os.path.dirname(path))

    # If the filename is a bare run index, descriptors come from the directory.
    if re.fullmatch(r'run[_-]?\d+', stem) and parent:
        run_idx = re.search(r'(\d+)', stem).group(1)
        stem = f"{parent}_run{run_idx}"

    seed = None
    m = re.search(r'seed[_-]?(\d+)', stem)
    if m:
        seed = m.group(1)
    else:
        m = re.search(r'run[_-]?(\d+)', stem)
        if m:
            seed = m.group(1)

    tasks = None
    m = re.search(r'(\d+)t(?![a-zA-Z0-9])', stem)
    if m:
        tasks = m.group(1)

    scenario = None
    for token in ('cil', 'til', 'class', 'task'):
        if re.search(rf'[_-]{token}[_-]', f'_{stem}_'):
            scenario = token
            break

    condition = re.sub(r'[_-]?seed[_-]?\d+', '', stem)
    condition = re.sub(r'[_-]?run[_-]?\d+', '', condition) or stem

    return {
        'condition': condition,
        'seed': seed if seed is not None else 'NA',
        'tasks': tasks if tasks is not None else 'NA',
        'scenario': scenario if scenario is not None else 'NA',
    }


def load_results(input_dir: str) -> List[Dict]:
    """Load every per-run JSON under `input_dir`."""
    records = []
    for root, _, files in os.walk(input_dir):
        for fname in sorted(files):
            if not fname.endswith('.json'):
                continue
            if fname in ('config.json', 'aggregated_results.json'):
                continue

            path = os.path.join(root, fname)
            try:
                with open(path, 'r') as f:
                    payload = json.load(f)
            except (json.JSONDecodeError, OSError) as exc:
                print(f"  skipped {path}: {exc}")
                continue

            metrics = payload.get('metrics')
            if not isinstance(metrics, dict):
                continue

            record = parse_run_name(path, input_dir)
            for key in METRICS:
                record[key] = metrics.get(key)
            record['path'] = os.path.relpath(path, input_dir)
            records.append(record)

    return records


def write_per_seed(records: List[Dict], path: str) -> None:
    fields = ['condition', 'scenario', 'tasks', 'seed'] + METRICS + ['path']
    with open(path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for r in sorted(records, key=lambda x: (x['condition'], str(x['seed']))):
            writer.writerow({k: r.get(k) for k in fields})


def write_summary(records: List[Dict], path: str) -> List[Dict]:
    grouped = defaultdict(list)
    for r in records:
        grouped[(r['condition'], r['scenario'], r['tasks'])].append(r)

    rows = []
    for (condition, scenario, tasks), group in sorted(grouped.items()):
        row = {
            'condition': condition,
            'scenario': scenario,
            'tasks': tasks,
            'n_seeds': len(group),
        }
        for key in METRICS:
            values = [g[key] for g in group if g.get(key) is not None]
            if values:
                row[f'{key}_mean'] = round(float(np.mean(values)), 4)
                row[f'{key}_std'] = round(float(np.std(values, ddof=1)), 4) if len(values) > 1 else 0.0
                row[f'{key}_min'] = round(float(np.min(values)), 4)
                row[f'{key}_max'] = round(float(np.max(values)), 4)
            else:
                for suffix in ('mean', 'std', 'min', 'max'):
                    row[f'{key}_{suffix}'] = None
        rows.append(row)

    fields = ['condition', 'scenario', 'tasks', 'n_seeds']
    for key in METRICS:
        fields += [f'{key}_mean', f'{key}_std', f'{key}_min', f'{key}_max']

    with open(path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    return rows


def main():
    parser = argparse.ArgumentParser(description='Aggregate per-seed LwU results')
    parser.add_argument('--input', type=str, required=True,
                        help='Directory containing per-run JSON results')
    parser.add_argument('--output', type=str, default=None,
                        help='Path for the summary CSV')
    args = parser.parse_args()

    if not os.path.isdir(args.input):
        raise SystemExit(f"Input directory not found: {args.input}")

    records = load_results(args.input)
    if not records:
        raise SystemExit(f"No result files with a 'metrics' field found under {args.input}")

    output = args.output or os.path.join(args.input, 'summary.csv')
    os.makedirs(os.path.dirname(output) or '.', exist_ok=True)

    per_seed = os.path.join(os.path.dirname(output) or '.', 'per_seed.csv')
    write_per_seed(records, per_seed)
    rows = write_summary(records, output)

    print(f"Loaded {len(records)} runs across {len(rows)} conditions.")
    print(f"  per-seed : {per_seed}")
    print(f"  summary  : {output}\n")

    header = f"{'condition':<34}{'scen':<6}{'tasks':<7}{'n':<4}{'ACC (mean +/- std)':<24}{'BWT':<10}{'PS':<8}"
    print(header)
    print('-' * len(header))
    for r in rows:
        acc = (f"{r['ACC_mean']:.2f} +/- {r['ACC_std']:.2f}"
               if r.get('ACC_mean') is not None else 'n/a')
        bwt = f"{r['BWT_mean']:.2f}" if r.get('BWT_mean') is not None else 'n/a'
        ps = f"{r['PS_mean']:.3f}" if r.get('PS_mean') is not None else 'n/a'
        print(f"{r['condition']:<34}{r['scenario']:<6}{str(r['tasks']):<7}"
              f"{r['n_seeds']:<4}{acc:<24}{bwt:<10}{ps:<8}")


if __name__ == '__main__':
    main()
