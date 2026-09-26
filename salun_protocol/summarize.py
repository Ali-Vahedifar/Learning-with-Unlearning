"""Mean ± SD over the ten test trials, with the absolute gap to the paired retrain models."""

import json
import statistics

from classification import GRID, WORK

METHODS = ['retrain', *GRID]
KEYS = [('UA', 'D_f acc'), ('RA', 'D_r acc'), ('TA', 'Test acc'), ('MIA', 'MIA')]


def value(record, key):
    return 100 - record[key] if key == 'UA' else record[key]  # report D_f accuracy, not UA


def main():
    lines = [
        '# SalUn-protocol classification study (CIFAR-10, ResNet-18)',
        '',
        'Settings selected on pilot seed 1 (validation); test trials are seeds 2-11, each with '
        'its own source model and paired retrain reference. Cells: mean ± SD (absolute gap to '
        'Retrain; smaller is better).',
        '',
    ]
    for ratio in (10, 50):
        data = {m: [] for m in METHODS}
        for seed in range(2, 12):
            for m in METHODS:
                path = WORK / f'seed{seed}' / f'forget{ratio}' / f'{m}.json'
                if path.exists():
                    data[m].append(json.loads(path.read_text()))
        refs = {r['seed']: r for r in data['retrain']}
        lines += [
            f'## {ratio}% random forgetting',
            '',
            '| Method | Trials | '
            + ' | '.join(f'{label} (gap)' for _, label in KEYS)
            + ' | Avg. gap ↓ | RTE (min) ↓ |',
            '|---|' + '---:|' * (len(KEYS) + 3),
        ]
        for m, records in data.items():
            records = [r for r in records if r['seed'] in refs]  # matched seeds only
            if not records:
                continue
            cells, gaps = [], []
            for key, _ in KEYS:
                v = [value(r, key) for r in records]
                mean = statistics.mean(v)
                sd = statistics.stdev(v) if len(v) > 1 else 0.0
                gap = abs(mean - statistics.mean(value(refs[r['seed']], key) for r in records))
                gaps.append(gap)
                cells.append(f'{mean:.2f} ± {sd:.2f} ({gap:.2f})')
            rte = statistics.mean(r['RTE_min'] for r in records)
            lines.append(
                f'| {m} | {len(records)} | '
                + ' | '.join(cells)
                + f' | {statistics.mean(gaps):.2f} | {rte:.2f} |'
            )
        lines.append('')
    path = WORK / 'summary.md'
    path.write_text('\n'.join(lines))
    print(path)


if __name__ == '__main__':
    main()
