#!/usr/bin/env python3
"""Collect every finished cell into one CSV and a per-seed summary.

    python tools/aggregate_results.py [--work $LWU_WORK] [--out results/]

Reads `<work>/<dataset>/mu/<backbone>_adam_<mode>/seed<k>/final.json` (and
`ulira/ulira.json` where the attack has been run) and writes:

    results.csv          one row per dataset x backbone x mode x seed x method
    summary.csv          mean and std over seeds

|delta D_f| is |D_f(method) - D_f(Retrain)| inside the same cell-seed, which is
the reported quantity: forgetting is measured against the retrained
oracle, not against zero.
"""

import argparse
import csv
import json
import os
import statistics
from pathlib import Path

METRICS = [
    'forget_acc',
    'delta_df',
    'retain_acc',
    'test_acc',
    'mia',
    'mia_auc',
    'output_kl_divergence',
    'kl_divergence',
    'unlearn_time',
    'relearn_final',
    'ulira_auc',
]


def cells(work):
    for final in sorted(Path(work).glob('*/mu/*/seed*/final.json')):
        seed_dir = final.parent
        backbone, _, mode = seed_dir.parent.name.split('_')
        yield {
            'dataset': seed_dir.parents[2].name,
            'backbone': backbone,
            'mode': mode,
            'seed': int(seed_dir.name.removeprefix('seed')),
            'final': json.loads(final.read_text()),
            'ulira': _read(seed_dir / 'ulira' / 'ulira.json'),
        }


def _read(path):
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return {}


def rows(work):
    for cell in cells(work):
        results = cell['final'].get('results', {})
        reference = results.get('retrain', {}).get('metrics', {}).get('forget_acc')
        attack = cell['ulira'].get('results', {})
        for method, record in results.items():
            metrics = record.get('metrics', {})
            if not metrics:
                continue
            relearn = metrics.get('relearn_forget_acc') or []
            row = {key: cell[key] for key in ('dataset', 'backbone', 'mode', 'seed')}
            row.update(
                {
                    'method': method,
                    'feasible': record.get('feasible'),
                    'config': json.dumps(record.get('config', {}), sort_keys=True),
                    'delta_df': (
                        None
                        if reference is None or metrics.get('forget_acc') is None
                        else abs(metrics['forget_acc'] - reference)
                    ),
                    'relearn_final': relearn[-1] if relearn else None,
                    'ulira_auc': attack.get(method, {}).get('auc'),
                }
            )
            row.update({key: metrics.get(key) for key in METRICS if key not in row})
            yield row


def summarise(all_rows):
    grouped = {}
    for row in all_rows:
        key = (row['dataset'], row['backbone'], row['mode'], row['method'])
        grouped.setdefault(key, []).append(row)
    for key, group in sorted(grouped.items()):
        summary = dict(zip(('dataset', 'backbone', 'mode', 'method'), key))
        summary['seeds'] = len(group)
        for metric in METRICS:
            values = [row[metric] for row in group if row.get(metric) is not None]
            summary[f'{metric}_mean'] = statistics.fmean(values) if values else None
            summary[f'{metric}_std'] = (
                statistics.stdev(values) if len(values) > 1 else 0.0 if values else None
            )
        yield summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--work', default=os.environ.get('LWU_WORK', './work'))
    parser.add_argument('--out', default='./results')
    args = parser.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    all_rows = list(rows(args.work))
    if not all_rows:
        raise SystemExit(f'no final.json under {args.work}')
    fields = ['dataset', 'backbone', 'mode', 'seed', 'method', 'feasible', 'config'] + METRICS
    with (out / 'results.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(all_rows)

    summaries = list(summarise(all_rows))
    with (out / 'summary.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summaries[0]))
        writer.writeheader()
        writer.writerows(summaries)
    print(f'{len(all_rows)} rows, {len(summaries)} method-cells -> {out}/')


if __name__ == '__main__':
    main()
