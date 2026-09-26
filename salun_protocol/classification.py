"""SalUn-protocol classification study: CIFAR-10, released ResNet-18, random forgetting.

Drives the baselines of the official SalUn release (vendored in salun_official/,
pinned commit in README.md) through one paper-aligned protocol:

    python classification.py --seed 1 --ratio .1 --tune          # pilot: search, write selected.json
    python classification.py --seed 2 --ratio .1 --settings selected/forget10.json

Source and retrain: 182 epochs of cosine SGD (lr .1, momentum .9, wd 5e-4,
batch 256). Metrics: UA = 100 - forget accuracy, RA, TA, and the release's
confidence-SVC MIA. Tuning selects, per method, the candidate with the smallest
mean absolute gap to the paired retrain model on UA/RA/validation accuracy/MIA.
"""

import argparse
import copy
import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'salun_official' / 'Classification'))
import arg_parser  # noqa: E402
import unlearn  # noqa: E402
import utils  # noqa: E402
from evaluation.SVC_MIA import SVC_fit_predict, collect_prob  # noqa: E402
from trainer import train  # noqa: E402

DATA = os.environ.get('LWU_DATA', './data')
WORK = Path(os.environ.get('LWU_WORK', './work')) / 'salun_protocol'
EPOCHS = 182
# Our name -> the release's unlearn entry point.
RELEASED = {
    'FT': 'FT',
    'RL': 'RL',
    'GA': 'GA',
    'IU': 'wfisher',
    'BE': 'boundary_expanding',
    'BS': 'boundary_shrink',
    'l1_sparse': 'FT_l1',
    'SalUn': 'RL',
    'SalUn_soft': 'RL_proximal',
}
# (lr, epochs, alpha) used when neither --tune nor --settings is given.
DEFAULTS = {
    'FT': (0.01, 10, 0.2),
    'RL': (0.01, 10, 0.2),
    'GA': (0.0001, 5, 0.2),
    'IU': (0.01, 1, 10),
    'BE': (0.00001, 10, 0.2),
    'BS': (0.00001, 10, 0.2),
    'l1_sparse': (0.01, 10, 0.00001),
    'SalUn': (0.013, 10, 0.2),
    'SalUn_soft': (0.013, 10, 0.2),
}
# Coarse grid inside the published ranges: (lr, epochs, alpha, mask density).
GRID = {
    'FT': [(v, 10, 0.2, 0.5) for v in [0.001, 0.01, 0.1]],
    'RL': [(v, 10, 0.2, 0.5) for v in [0.001, 0.01, 0.1]],
    'GA': [(v, 5, 0.2, 0.5) for v in [0.00001, 0.0001, 0.001]],
    'IU': [(0.01, 1, v, 0.5) for v in [1, 10, 20]],
    'BE': [(v, 10, 0.2, 0.5) for v in [0.000001, 0.00001, 0.0001]],
    'BS': [(v, 10, 0.2, 0.5) for v in [0.000001, 0.00001, 0.0001]],
    'l1_sparse': [
        (lr, 10, v, 0.5) for lr in [0.001, 0.01, 0.1] for v in [0.000001, 0.00001, 0.0001]
    ],
    'SalUn': [(lr, 10, 0.2, d) for lr in [0.0005, 0.005, 0.013, 0.05] for d in [0.1, 0.5, 0.9]],
    'SalUn_soft': [
        (lr, 10, 0.2, d) for lr in [0.0005, 0.005, 0.013, 0.05] for d in [0.1, 0.5, 0.9]
    ],
}
METHODS = ['retrain', *GRID]


def dump(path, obj):
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(obj, indent=2))
    tmp.replace(path)


def loader(dataset, shuffle=False):
    return DataLoader(dataset, batch_size=256, shuffle=shuffle, num_workers=0, pin_memory=True)


def evaluate(model, retain, forget, test):
    probs, acc = {}, {}
    for name, dataset in (('retain', retain), ('forget', forget), ('test', test)):
        dataset = copy.deepcopy(dataset)
        dataset.transform = transforms.ToTensor()
        p, y = collect_prob(loader(dataset), model)
        acc[name] = 100 * float((p.argmax(1) == y).float().mean())
        probs[name] = p.gather(1, y[:, None])
    n = len(test)
    mia = 100 * float(
        SVC_fit_predict(probs['retain'][:n], probs['test'], torch.zeros((0, 1)), probs['forget'])
    )
    return dict(UA=100 - acc['forget'], RA=acc['retain'], TA=acc['test'], MIA=mia)


def saliency_mask(model, forget, density):
    """SalUn: keep the `density` fraction of weights with the largest forget-loss gradient."""
    model.eval()
    grads = {n: torch.zeros_like(p) for n, p in model.named_parameters()}
    for x, y in loader(forget, True):
        model.zero_grad(set_to_none=True)
        nn.functional.cross_entropy(model(x.cuda()), y.cuda()).backward()
        for n, p in model.named_parameters():
            if p.grad is not None:
                grads[n].add_(p.grad)
    flat = torch.cat([g.abs().flatten() for g in grads.values()])
    keep = torch.zeros_like(flat)
    keep[torch.argsort(flat, descending=True)[: int(len(flat) * density)]] = 1
    mask, start = {}, 0
    for n, p in model.named_parameters():
        mask[n] = keep[start : start + p.numel()].reshape_as(p)
        start += p.numel()
    model.zero_grad(set_to_none=True)
    return mask


def fit(model, dataset, args, path):
    """Source/retrain training, checkpointed every 10 epochs so it resumes."""
    opt = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    begin = time.perf_counter()
    history = []
    start = 0
    partial = path.with_name(path.stem + '_progress.pt')
    if partial.exists():
        ck = torch.load(partial, weights_only=False)
        model.load_state_dict(ck['model'])
        opt.load_state_dict(ck['optimizer'])
        sched.load_state_dict(ck['scheduler'])
        history = ck['history']
        start = ck['epoch'] + 1
        torch.set_rng_state(ck['rng_cpu'])
        torch.cuda.set_rng_state(ck['rng_cuda'])
        np.random.set_state(ck['rng_numpy'])
        random.setstate(ck['rng_python'])
        begin -= ck['seconds']
    for epoch in range(start, EPOCHS):
        train(loader(dataset, True), model, nn.CrossEntropyLoss(), opt, epoch, args)
        sched.step()
        history.append({'epoch': epoch, 'seconds': time.perf_counter() - begin})
        print(path.name, 'epoch', epoch, 'seconds', history[-1]['seconds'], flush=True)
        if epoch % 10 == 0 or epoch == EPOCHS - 1:
            torch.save(
                dict(
                    model=model.state_dict(),
                    optimizer=opt.state_dict(),
                    scheduler=sched.state_dict(),
                    epoch=epoch,
                    history=history,
                    seconds=history[-1]['seconds'],
                    rng_cpu=torch.get_rng_state(),
                    rng_cuda=torch.cuda.get_rng_state(),
                    rng_numpy=np.random.get_state(),
                    rng_python=random.getstate(),
                ),
                partial,
            )
    torch.save(
        {
            'state_dict': model.state_dict(),
            'history': history,
            'seconds': time.perf_counter() - begin,
        },
        path,
    )
    if partial.exists():
        partial.unlink()
    return time.perf_counter() - begin


def drop_reject_class(model):
    """BE trains with a temporary eleventh class; remove it for ten-class evaluation."""
    for module in model.modules():
        for name, child in list(module.named_children()):
            if isinstance(child, nn.Linear) and child.out_features == 11:
                head = nn.Linear(child.in_features, 10, bias=child.bias is not None).cuda()
                with torch.no_grad():
                    head.weight.copy_(child.weight[:10])
                    if child.bias is not None:
                        head.bias.copy_(child.bias[:10])
                setattr(module, name, head)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seed', type=int, required=True)
    ap.add_argument('--ratio', type=float, choices=[0.1, 0.5], required=True)
    ap.add_argument('--methods', default=','.join(METHODS))
    ap.add_argument('--source_only', action='store_true')
    ap.add_argument('--tune', action='store_true')
    ap.add_argument('--settings', type=Path)
    a = ap.parse_args()
    sys.argv = [sys.argv[0]]
    args = arg_parser.parse_args()  # the release's own defaults
    args.seed = args.train_seed = a.seed
    args.data = DATA
    args.print_freq = 10000
    args.mask_ratio = 0.5
    args.save_dir = str(WORK / f'seed{a.seed}')
    Path(args.save_dir).mkdir(parents=True, exist_ok=True)
    # The release splits train/validation with seed=1 regardless of the run seed.
    original = utils.cifar10_dataloaders

    def same_split(*v, **kw):
        kw.setdefault('seed', a.seed)
        return original(*v, **kw)

    utils.cifar10_dataloaders = same_split
    model, tr, va, te, _ = utils.setup_model_dataset(args)
    model = model.cuda()
    initial = copy.deepcopy(model.state_dict())
    source = Path(args.save_dir) / 'source.pt'
    if source.exists():
        model.load_state_dict(torch.load(source, weights_only=False)['state_dict'])
    else:
        fit(model, tr.dataset, args, source)
    if a.source_only:
        return

    out = Path(args.save_dir) / f'forget{int(a.ratio*100)}'
    out.mkdir(exist_ok=True)
    # Same random marking rule as the release's replace_class(..., class=-1).
    n = len(tr.dataset)
    rng = np.random.RandomState(a.seed - 1)
    chosen = rng.choice(n, int(n * a.ratio), replace=False)
    flag = np.zeros(n, dtype=bool)
    flag[chosen] = True
    np.savez(out / 'split.npz', forget=chosen, retain=np.flatnonzero(~flag))
    forget = copy.deepcopy(tr.dataset)
    retain = copy.deepcopy(tr.dataset)
    forget.data = forget.data[flag]
    forget.targets = np.asarray(forget.targets)[flag]
    retain.data = retain.data[~flag]
    retain.targets = np.asarray(retain.targets)[~flag]
    data = {'forget': loader(forget, True), 'retain': loader(retain, True), 'test': te, 'val': va}
    source_state = copy.deepcopy(model.state_dict())

    settings = dict(DEFAULTS)
    chosen_settings = {}
    if a.settings:
        chosen_settings = json.loads(a.settings.read_text())
        for name, v in chosen_settings.items():
            if name in settings:
                settings[name] = (v['lr'], v['epochs'], v['alpha'])
    names = a.methods.split(',')
    jobs = []
    for name in names:
        if a.tune and name != 'retrain':
            jobs += [(name, f'{name}_candidate{i}', v) for i, v in enumerate(GRID[name])]
        else:
            density = chosen_settings.get(name, {}).get('density', 0.5)
            jobs.append((name, name, (*settings.get(name, (0, 0, 0)), density)))
    if a.tune:
        out = out / 'tuning'
        out.mkdir(exist_ok=True)

    masks = {}
    for name, key, hp in jobs:
        result = out / f'{key}.json'
        if result.exists():
            continue
        utils.setup_seed(a.seed)
        student = copy.deepcopy(model)
        student.load_state_dict(source_state)
        start = time.perf_counter()
        if name == 'retrain':
            hp = (0.1, EPOCHS, 0, 1)
            ck = (out.parent if a.tune else out) / 'retrain.pt'
            if ck.exists():
                obj = torch.load(ck, weights_only=False)
                student.load_state_dict(obj['state_dict'])
                seconds = obj['seconds']
            else:
                student.load_state_dict(initial)
                seconds = fit(student, retain, args, ck)
        else:
            args.unlearn = RELEASED[name]
            args.unlearn_lr, args.unlearn_epochs, args.alpha, args.mask_ratio = hp
            args.save_dir = str(out / key)
            Path(args.save_dir).mkdir(exist_ok=True)
            mask = None
            if name == 'SalUn':
                if hp[3] not in masks:
                    masks[hp[3]] = saliency_mask(model, forget, hp[3])
                mask = masks[hp[3]]
            unlearn.get_unlearn_method(args.unlearn)(
                data, student, nn.CrossEntropyLoss(), args, mask
            )
            if name == 'BE':
                drop_reject_class(student)
            seconds = time.perf_counter() - start
        metrics = evaluate(student, retain, forget, va.dataset if a.tune else te.dataset)
        metrics.update(
            RTE_min=seconds / 60,
            seed=a.seed,
            ratio=a.ratio,
            method=name,
            evaluation_split='validation' if a.tune else 'test',
            hyperparameters=dict(lr=hp[0], epochs=hp[1], alpha=hp[2], density=hp[3]),
        )
        dump(result, metrics)
        print('RESULT', metrics, flush=True)

    if a.tune:
        ref = json.loads((out / 'retrain.json').read_text())
        selected = {}
        for name in names:
            if name == 'retrain':
                continue
            candidates = [json.loads(p.read_text()) for p in out.glob(f'{name}_candidate*.json')]
            for r in candidates:
                r['selection_gap'] = sum(abs(r[k] - ref[k]) for k in ('UA', 'RA', 'TA', 'MIA')) / 4
            best = min(candidates, key=lambda r: r['selection_gap'])
            selected[name] = {**best['hyperparameters'], 'validation_gap': best['selection_gap']}
        # Merge, so tuning one added method keeps the settings already frozen for the others.
        path = out / 'selected.json'
        prior = json.loads(path.read_text()) if path.exists() else {}
        prior.update(selected)
        dump(path, prior)


if __name__ == '__main__':
    main()
