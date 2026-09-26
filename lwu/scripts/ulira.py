#!/usr/bin/env python3
"""U-LiRA: per-example likelihood-ratio membership attack on the final models.

Hayes et al. (2024), "Inexact unlearning needs more careful evaluations".
For every forget example x the attacker gets the target model's logit-scaled
confidence phi(x) = log p_y - log(1 - p_y) and asks whether it looks like

  IN : a model trained WITH x, then unlearned with method M  (shadow, x in S_k)
  OUT: a model trained WITHOUT x, then unlearned on the rest (shadow, x not in S_k)

Shadow k is trained on D_r plus a random half S_k of D_f with the benchmark's
own recipe, then unlearned with M's selected configuration.  Gaussians are
fitted per example (means) with a shared variance, which is what keeps the
attack usable at 16 shadows.  The score is the LLR over the forget set where
positives come from M's actual final model and negatives from the Retrain
reference on the same examples: AUC 50 means M is indistinguishable from
Retrain on D_f; the source model ("baseline") gives the no-unlearning ceiling.

Runs after `benchmark_unlearning.py --phase all` with the same arguments plus
--shadows.  Everything is cached under <output_dir>/ulira/, so it resumes.
"""

import argparse
import json
import sys
import time
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, roc_curve
from torch.utils.data import DataLoader, Subset

REPOSITORY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY))

from LwU import ZoneDecomposition  # noqa: E402
from core.utils.training import set_seed  # noqa: E402
from scripts import benchmark_unlearning as bench_module  # noqa: E402
from scripts.benchmark_unlearning import Benchmark, atomic_json, sync  # noqa: E402
from scripts.train_unlearning import train_baseline_model  # noqa: E402


def parse_args():
    own = argparse.ArgumentParser(add_help=False)
    own.add_argument('--shadows', type=int, default=16)
    own.add_argument(
        '--ulira_methods',
        default=None,
        help='Comma-separated subset of the final methods (default: all)',
    )
    mine, rest = own.parse_known_args()
    sys.argv = [sys.argv[0]] + rest
    args = bench_module.parse_args()
    args.shadows = mine.shadows
    args.ulira_methods = mine.ulira_methods
    if args.shadows < 4:
        own.error('--shadows must be at least 4')
    return args


@torch.no_grad()
def logit_confidence(model, loader, device):
    """phi = log p_y - log(1 - p_y) for every example, in loader order."""
    adapt = isinstance(model, ZoneDecomposition)
    model.eval()
    scores = []
    stream = model.test_stream() if adapt else _null()
    with stream:
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)
            logits = model(inputs, test_time_adapt=True) if adapt else model(inputs)
            probabilities = F.softmax(logits.float(), dim=1)
            p = probabilities.gather(1, targets.view(-1, 1)).squeeze(1)
            p = p.clamp(1e-7, 1 - 1e-7)
            scores.append((p.log() - (1 - p).log()).cpu())
    return torch.cat(scores).numpy().astype(np.float32)


@contextmanager
def _null():
    yield


def balanced_masks(shadows, n, seed):
    """(K, n) bool: every example is IN for exactly K // 2 shadows.

    Random coin flips leave some example all-IN or all-OUT (3e-5 per example
    at K = 16 -- enough to hit once in 4500), which makes its LLR undefined.
    """
    rng = np.random.default_rng(seed)
    return np.argsort(rng.random((shadows, n)), axis=0) < shadows // 2


def likelihood_ratio(shadow_scores, in_mask, target_scores):
    """Per-example LLR of ``target_scores`` under IN vs OUT Gaussians.

    shadow_scores: (K, n) phi of every shadow on every forget example.
    in_mask:       (K, n) bool, x_i in S_k.
    Returns an (n,) array; larger = more like "trained then unlearned".
    """
    in_count = in_mask.sum(0)
    out_count = (~in_mask).sum(0)
    if (in_count == 0).any() or (out_count == 0).any():
        raise ValueError('every forget example needs at least one IN and one OUT shadow')
    mean_in = np.where(in_mask, shadow_scores, 0).sum(0) / in_count
    mean_out = np.where(~in_mask, shadow_scores, 0).sum(0) / out_count
    # Shared variance: per-example variances from ~8 shadows are too noisy.
    var_in = np.mean(((shadow_scores - mean_in) ** 2)[in_mask]) + 1e-6
    var_out = np.mean(((shadow_scores - mean_out) ** 2)[~in_mask]) + 1e-6

    def log_pdf(x, mean, var):
        return -0.5 * ((x - mean) ** 2 / var + np.log(2 * np.pi * var))

    return log_pdf(target_scores, mean_in, var_in) - log_pdf(target_scores, mean_out, var_out)


def attack_metrics(positive, negative):
    labels = np.concatenate([np.ones(len(positive)), np.zeros(len(negative))])
    scores = np.concatenate([positive, negative])
    fpr, tpr, _ = roc_curve(labels, scores)
    return {
        'auc': 100.0 * float(roc_auc_score(labels, scores)),
        'tpr_at_1pct_fpr': 100.0 * float(np.interp(0.01, fpr, tpr)),
        'tpr_at_0.1pct_fpr': 100.0 * float(np.interp(0.001, fpr, tpr)),
        'balanced_acc': 50.0 * (float((positive > 0).mean()) + float((negative <= 0).mean())),
    }


class ULiRA:
    def __init__(self, args):
        self.args = args
        self.bench = Benchmark(args)
        self.device = args.device
        self.out = Path(args.output_dir) / 'ulira'
        self.out.mkdir(parents=True, exist_ok=True)
        self.search = json.loads((Path(args.output_dir) / 'search.json').read_text())
        self.final = json.loads((Path(args.output_dir) / 'final.json').read_text())
        loaders = self.bench.loaders
        self.train_dataset = loaders['forget_train'].dataset.dataset
        self.forget_idx = np.asarray(loaders['forget_train'].dataset.indices)
        self.retain_idx = np.asarray(loaders['retain_full'].dataset.indices)
        self.score_loader = loaders['forget_train_eval']
        assert list(self.score_loader.dataset.indices) == list(self.forget_idx)
        self.methods = self._methods()
        self.in_masks = balanced_masks(args.shadows, len(self.forget_idx), args.seed * 1000)

    def _methods(self):
        names = list(self.final['results'])
        if self.args.ulira_methods:
            wanted = [n.strip() for n in self.args.ulira_methods.split(',') if n.strip()]
            missing = set(wanted) - set(names)
            if missing:
                raise ValueError(f'not in final.json: {sorted(missing)}')
            names = wanted
        # amnesiac's ledger is recorded during source training, so no per-shadow
        # equivalent exists; retrain is the U-LiRA reference, not a target.
        return [n for n in names if n not in ('retrain', 'amnesiac')]

    def _config(self, name):
        record = self.final['results'][name]
        if 'ablations' in record:
            # One Zone-C rule per run in this campaign; take the first.
            return next(iter(record['ablations'].values()))['config'], name
        return record.get('config', {}), name

    def _loader(self, indices, shuffle):
        a = self.args
        return DataLoader(
            Subset(self.train_dataset, [int(i) for i in indices]),
            batch_size=a.batch_size,
            shuffle=shuffle,
            num_workers=a.num_workers,
            pin_memory=str(a.device).startswith('cuda'),
        )

    @contextmanager
    def shadow_world(self, in_idx, k):
        """Point the benchmark's loaders at shadow k's training world."""
        rng = np.random.default_rng(self.args.seed * 7919 + k)
        small_n = max(1, int(round(len(self.retain_idx) * self.args.retain_ratio)))
        small = rng.choice(self.retain_idx, size=small_n, replace=False)
        swapped = {
            'forget_train': self._loader(in_idx, True),
            'retain_small': self._loader(small, True),
            'retain_full': self._loader(self.retain_idx, True),
            'full_train': self._loader(np.concatenate([self.retain_idx, in_idx]), True),
        }
        saved = {key: self.bench.loaders[key] for key in swapped}
        self.bench.loaders.update(swapped)
        try:
            yield
        finally:
            self.bench.loaders.update(saved)

    def shadow(self, k):
        path = self.out / f'shadow_{k}.pt'
        in_mask = self.in_masks[k]
        if path.exists():
            payload = torch.load(path, map_location='cpu', weights_only=False)
            if not np.array_equal(payload['in_mask'], in_mask):
                raise RuntimeError(
                    f'{path} was built with a different --shadows/seed; '
                    'delete the ulira/ directory to rebuild'
                )
            return payload['state'], in_mask
        print(
            f'shadow {k}/{self.args.shadows}: training on D_r + {int(in_mask.sum())} of D_f',
            flush=True,
        )
        set_seed(self.args.seed + 1000 + k)
        model = self.bench.factory().to(self.device)
        training_args = self.bench.training_args
        training_args.method = 'retrain'
        started = time.perf_counter()
        with self.shadow_world(self.forget_idx[in_mask], k):
            model, _ = train_baseline_model(
                model,
                self.bench.loaders['full_train'],
                self.bench.loaders['full_val'],
                training_args,
                -1,
            )
        sync(self.device)
        state = {n: v.detach().cpu().clone() for n, v in model.state_dict().items()}
        torch.save(
            {
                'state': state,
                'in_mask': in_mask,
                'train_time': time.perf_counter() - started,
                'training': model.training_summary,
            },
            path,
        )
        return state, in_mask

    def shadow_scores(self, name, config, k, state, in_mask):
        path = self.out / f'scores_{name}_{k}.npy'
        if path.exists():
            return np.load(path)
        model = self.bench.factory().to(self.device)
        model.load_state_dict(state)
        if name == 'baseline':
            unlearned = model
        else:
            set_seed(self.args.seed)
            with self.shadow_world(self.forget_idx[in_mask], k):
                method = self.bench._build(name, model, config, selection=False)
                try:
                    unlearned = self.bench._apply(name, method, config, selection=False)
                except (FloatingPointError, RuntimeError, ValueError) as exc:
                    print(f'  shadow {k} {name}: {type(exc).__name__}: {exc}', flush=True)
                    unlearned = None
        scores = (
            np.full(len(self.forget_idx), np.nan, dtype=np.float32)
            if unlearned is None
            else logit_confidence(unlearned, self.score_loader, self.device)
        )
        np.save(path, scores)
        return scores

    def target_scores(self, name, config):
        path = self.out / f'target_{name}.npy'
        if path.exists():
            return np.load(path)
        model, _ = self.bench._final_method(name, config)
        scores = logit_confidence(model, self.score_loader, self.device)
        np.save(path, scores)
        return scores

    def run(self):
        report_path = self.out / 'ulira.json'
        report = {
            'protocol': self.bench.protocol,
            'shadows': self.args.shadows,
            'n_forget': int(len(self.forget_idx)),
            'results': {},
        }
        reference = self.out / 'target_retrain.npy'
        if reference.exists():
            negatives_phi = np.load(reference)
        else:
            negatives_phi = logit_confidence(self.bench.reference, self.score_loader, self.device)
            np.save(reference, negatives_phi)

        shadows = [self.shadow(k) for k in range(self.args.shadows)]
        in_masks = np.stack([m for _, m in shadows])
        for name in self.methods:
            config, _ = self._config(name)
            print(f'{name}: unlearning {self.args.shadows} shadows', flush=True)
            phi = np.stack(
                [
                    self.shadow_scores(name, config, k, state, mask)
                    for k, (state, mask) in enumerate(shadows)
                ]
            )
            usable = ~np.isnan(phi).any(1)
            if usable.sum() < 4:
                report['results'][name] = {'error': f'only {int(usable.sum())} finite shadows'}
                continue
            target = self.target_scores(name, config)
            llr_target = likelihood_ratio(phi[usable], in_masks[usable], target)
            llr_reference = likelihood_ratio(phi[usable], in_masks[usable], negatives_phi)
            metrics = attack_metrics(llr_target, llr_reference)
            metrics['finite_shadows'] = int(usable.sum())
            report['results'][name] = metrics
            print(
                f'  AUC={metrics["auc"]:.2f} TPR@1%FPR={metrics["tpr_at_1pct_fpr"]:.2f} '
                f'bal.acc={metrics["balanced_acc"]:.2f}',
                flush=True,
            )
            atomic_json(report_path, report)
        report['results']['retrain'] = {'auc': 50.0, 'note': 'reference by construction'}
        atomic_json(report_path, report)
        return report


def main():
    ULiRA(parse_args()).run()


if __name__ == '__main__':
    main()
