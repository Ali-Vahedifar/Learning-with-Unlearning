"""Ten independent CIFAR-10 DDPM class deletions: LwU, SalUn and a random mask.

    python -m diffusion.run_classes                 # train, sample, score
    python -m diffusion.run_classes --classes 0,1   # a subset

Every method starts from one source checkpoint ($LWU_WORK/generation/ddpm/source.pt,
see train_source.py), deletes one class, and is sampled with the same seed, step
count and guidance. Artifacts go to $LWU_WORK/generation/classes/<method>/class_<c>/,
the report and metrics to generation/diffusion/results/.
"""

import argparse
import gc
import json
import os
import time
from pathlib import Path

import torch

from diffusion import lwu as lwu_method
from diffusion.baselines import load_source, saliency_masks, train_masked
from diffusion.data import cifar10
from diffusion.ddpm import Diffusion
from diffusion.evaluate import CLASSES, classify, generate, judge

OUT = Path(__file__).resolve().parent / 'results'
ROWS = ['Source', 'LwU', 'SalUn', 'Random mask']
CONFIG = dict(
    seed=42,
    mask_seed=4242,
    steps=1000,
    lr=1e-4,
    batch=128,
    mask_fraction=0.5,
    sampling_seed=10043,
    sampling_steps=1000,
    guidance=2.0,
    samples_per_condition=16,
    target_rule='(forgotten_class + 1) % 10',
)


def work():
    return Path(os.environ.get('LWU_WORK', './work')) / 'generation' / 'classes'


def source_checkpoint():
    return Path(os.environ.get('LWU_WORK', './work')) / 'generation' / 'ddpm' / 'source.pt'


def status(stage, **kw):
    value = dict(stage=stage, utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), **kw)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / 'status.json').write_text(json.dumps(value, indent=2))
    print(value, flush=True)


def samples_for(model, labels):
    torch.manual_seed(CONFIG['sampling_seed'])
    grid = generate(
        model,
        Diffusion(),
        labels,
        bs=160,
        guidance=CONFIG['guidance'],
        steps=CONFIG['sampling_steps'],
    )
    assert torch.isfinite(grid).all()
    return grid.reshape(10, CONFIG['samples_per_condition'], 3, 32, 32)


def method_samples(name, source, x, y, c, labels):
    """Train (or reuse) one method's deletion of class `c` and sample its grid."""
    wd = work() / name / f'class_{c}'
    wd.mkdir(parents=True, exist_ok=True)
    path = wd / 'samples.pt'
    if path.exists():
        return torch.load(path, map_location='cpu', weights_only=True)
    if name == 'lwu':
        model = lwu_method.unlearn(source, x, y, c, wd, log=lambda row: status(**row))
    else:
        masks = saliency_masks(
            source,
            x[y == c],
            y[y == c],
            mask_seed=CONFIG['mask_seed'],
            fraction=CONFIG['mask_fraction'],
        )
        kind = 'salun' if name == 'salun' else 'random'
        model, history = train_masked(
            source,
            masks[kind],
            x[y != c],
            y[y != c],
            x[y == c],
            y[y == c],
            steps=CONFIG['steps'],
            lr=CONFIG['lr'],
            bs=CONFIG['batch'],
        )
        torch.save(
            dict(
                base=model.stem.out_channels,
                model=model.state_dict(),
                config=CONFIG,
                forgotten_class=c,
                history=history,
            ),
            wd / f'{kind}.pt',
        )
    status('sampling', method=name, forgotten_class=c)
    grid = samples_for(model.eval(), labels)
    torch.save(grid, path)
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return grid


def score(grids, c, clf):
    rows = {}
    n = CONFIG['samples_per_condition']
    for name, grid in grids.items():
        pred = classify(grid.flatten(0, 1).cuda(), clf).cpu().reshape(10, n)
        correct = pred == torch.arange(10)[:, None]
        rows[name] = dict(
            forget_correct=int(correct[c].sum()),
            forget_n=n,
            retain_correct=int(correct[torch.arange(10) != c].sum()),
            retain_n=9 * n,
        )
    return rows


def report(metrics):
    n = CONFIG['samples_per_condition']
    lines = [
        '# CIFAR-10 DDPM class removal',
        '',
        f'{len(metrics)}/10 independent class deletions. Every method starts from the same '
        'source checkpoint and gets the same update budget; SalUn and the random mask differ '
        'only in which half of the weights may move.',
        '',
        '| Forgotten class | Method | Forgotten-condition accuracy (%) | Retained-condition accuracy (%) |',
        '|---|---|---:|---:|',
    ]
    for c, rows in sorted(metrics.items(), key=lambda kv: int(kv[0])):
        for name in ROWS:
            r = rows[name]
            lines.append(
                f'| {CLASSES[int(c)]} | {name} | '
                f'{100*r["forget_correct"]/r["forget_n"]:.2f} | '
                f'{100*r["retain_correct"]/r["retain_n"]:.2f} |'
            )
    lines += [
        '',
        'Low forgotten-condition accuracy alone does not establish removal: a model that '
        'generates nothing recognisable also scores low. Read it together with the retained '
        'column and the samples. Single training seed, '
        f'{n} samples per condition; the samples are saved as tensors next to each '
        'checkpoint.',
    ]
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / 'report.md').write_text('\n'.join(lines) + '\n')
    (OUT / 'metrics.json').write_text(json.dumps(metrics, indent=2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--classes', default='0,1,2,3,4,5,6,7,8,9')
    args, _ = ap.parse_known_args()
    classes = [int(c) for c in args.classes.split(',') if c.strip() != '']

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / 'config.json').write_text(json.dumps(CONFIG, indent=2))
    torch.set_num_threads(4)
    source = load_source(source_checkpoint()).eval()
    x, y = cifar10()
    x, y = x.cuda(), y.cuda()
    clf = judge()
    labels = torch.arange(10).repeat_interleave(CONFIG['samples_per_condition'])

    source_dir = work() / 'source'
    source_dir.mkdir(parents=True, exist_ok=True)
    if (source_dir / 'samples.pt').exists():
        source_grid = torch.load(source_dir / 'samples.pt', map_location='cpu', weights_only=True)
    else:
        status('sampling', method='source')
        source_grid = samples_for(source, labels)
        torch.save(source_grid, source_dir / 'samples.pt')

    metrics = {}
    if (OUT / 'metrics.json').exists():
        metrics = json.loads((OUT / 'metrics.json').read_text())
    for c in classes:
        grids = {'Source': source_grid}
        grids['LwU'] = method_samples('lwu', source, x, y, c, labels)
        grids['SalUn'] = method_samples('salun', source, x, y, c, labels)
        grids['Random mask'] = method_samples('random', source, x, y, c, labels)
        metrics[str(c)] = score(grids, c, clf)
        report(metrics)
        status('class_complete', forgotten_class=c, metrics=metrics[str(c)])
        gc.collect()
        torch.cuda.empty_cache()
    status('complete')


if __name__ == '__main__':
    main()
