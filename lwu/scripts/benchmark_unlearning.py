#!/usr/bin/env python3
"""Leakage-free CIFAR-100 class-unlearning benchmark.

This is the canonical runner for the manuscript table.  It enforces five
comparability requirements that ad-hoc per-method invocations cannot enforce:

1. every method starts from one byte-identical source checkpoint;
2. Retrain uses the source model's optimiser, scheduler, validation, and early
   stopping recipe, changing only the training data;
3. hyperparameters are selected using validation data only;
4. the test set is evaluated once, after the complete Cartesian grids finish;
5. reported method time includes method-specific preparation (notably SSD's
   full-data Fisher), while source training is excluded for every post-hoc
   method and included for Retrain by definition.

The membership attack learns its threshold and score direction from dedicated
calibration subsets, selects hyperparameters on separate validation subsets,
and reports on disjoint final member samples plus the CIFAR-100 test split.
"""

import argparse
import copy
import hashlib
import itertools
import json
import os
import sys
import tempfile
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

REPOSITORY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY))
from LwU import LwU, ZoneDecomposition  # noqa: E402
from Machine_Unlearning_baselines import (  # noqa: E402
    Amnesiac,
    BadTeacher,
    BoundaryExpand,
    BoundaryShrink,
    FineTune,
    GradientAscent,
    InfluenceUnlearning,
    L1Sparse,
    RandomLabel,
    SCRUB,
    SSD,
    SalUn,
    UNSIR,
    UniCLUN,
)
from core.datasets.cifar import get_dataset  # noqa: E402
from core.datasets.cifar100_coarse import (  # noqa: E402
    CoarseLabelCIFAR100,
    fine_to_coarse_map,
)
from core.datasets.cifar10_coarse import (  # noqa: E402
    NUM_COARSE_CLASSES as CIFAR10_COARSE_CLASSES,
    fine_to_coarse_map as cifar10_fine_to_coarse_map,
)
from core.datasets.tinyimagenet import (  # noqa: E402
    CoarseLabelDataset,
    coarse_class_count as tinyimagenet_coarse_count,
    fine_to_coarse_map as tinyimagenet_fine_to_coarse_map,
)
from core.evaluation.metrics import UnlearningEvaluator, evaluate_task  # noqa: E402
from core.models.backbones import get_backbone  # noqa: E402
from core.utils.training import set_seed  # noqa: E402
from scripts.train_unlearning import (  # noqa: E402
    build_instance_unlearning_loaders,
    build_unlearning_loaders,
    train_baseline_model,
)

PROTOCOL_VERSION = 'cifar100-mu-leakage-free-v3-e20'
DEFAULT_METHODS = [
    'baseline',
    'retrain',
    'finetune',
    'rl',
    'ga',
    'l1sparse',
    'iu',
    'bs',
    'be',
    'badteacher',
    'amnesiac',
    'unsir',
    'ssd',
    'uniclun',
    'scrub',
    'salun',
]
# LwU is the proposed method: Fisher-importance zoning, teacher-guided repair
# and a test-time fast-weight memory.  It is optional only in the sense that a
# cell can be run with baselines alone.
OPTIONAL_METHODS = ['lwu']
SEARCH_PRIORITY = [
    'finetune',
    'rl',
    'ga',
    'l1sparse',
    'iu',
    'bs',
    'be',
    'ssd',
    'badteacher',
    'unsir',
    'scrub',
    'salun',
    'uniclun',
    'lwu',
]

STATIC_GRIDS = {
    'finetune': {
        'lr': [1e-4, 1e-3, 1e-2],
        'epochs': [1, 3, 5, 10],
    },
    'badteacher': {
        'temperature': [1.0, 2.0, 4.0],
        'epochs': [1, 3, 5, 10],
        'lr': [1e-4, 1e-3, 1e-2],
    },
    'unsir': {
        'impair_epochs': [1, 2],
        'repair_epochs': [1, 2, 3],
        'impair_lr': [1e-3, 1e-2, 1e-1],
        'repair_lr': [1e-3, 1e-2],
    },
    'ssd': {
        'alpha': [1.0, 5.0, 10.0, 50.0, 100.0],
        'dampening_constant': [0.1, 0.5, 1.0, 5.0, 10.0],
    },
    'uniclun': {
        'lr': [1e-4, 1e-3, 1e-2],
        'epochs': [3, 5, 10],
        'buffer_size': [250, 500, 1000],
    },
    'scrub': {
        'msteps': [1, 2, 3],
        'lr': [1e-4, 5e-4, 1e-3, 5e-3],
        'epochs': [5, 10],
        # Max-step gradient-norm bound: on the FPECNN backbone the unbounded
        # ascent diverged for every lr >= 5e-4.
        'max_grad_norm': [5.0],
    },
    'salun': {
        'sparsity': [0.1, 0.3, 0.5, 0.7, 0.9],
        'lr': [1e-4, 1e-3, 1e-2],
        'epochs': [3, 5, 10],
    },
    # SalUn's objective without the saliency mask, so the same grid minus
    # sparsity: any difference between the two rows is the mask.
    'rl': {
        'lr': [1e-4, 1e-3, 1e-2],
        'epochs': [3, 5, 10],
    },
    # Ascent is unanchored, so the useful range sits an order of magnitude
    # below every descent method's: at 1e-2 the model is destroyed in one epoch.
    'ga': {
        'lr': [1e-5, 1e-4, 1e-3],
        'epochs': [1, 3, 5],
    },
    'l1sparse': {
        'lr': [1e-4, 1e-3, 1e-2],
        'epochs': [5, 10],
        'alpha': [1e-5, 5e-5, 1e-4],
    },
    # One closed-form step: alpha is the whole step size, so it is the only
    # thing to search, over the official range.
    'iu': {
        'alpha': [0.1, 0.5, 1.0, 5.0, 10.0, 20.0],
    },
    'bs': {
        'lr': [1e-5, 1e-4, 1e-3],
        'epochs': [1, 3, 5],
        # 0.1 is the paper's fixed FGSM bound; searched as a single value so it
        # is recorded with every result rather than buried in the class.
        'bound': [0.1],
    },
    'be': {
        'lr': [1e-5, 1e-4, 1e-3],
        'epochs': [1, 3, 5],
    },
}

# What each post-hoc method needs beyond the source checkpoint.  Reported next
# to the numbers so a cheap method that needs a per-batch training ledger is
# not read as cheap.
ACCESS = {
    'baseline': 'source only',
    'retrain': 'full D_r, training recipe',
    'amnesiac': 'per-batch update ledger recorded during source training',
    'finetune': 'D_r (labels)',
    'badteacher': 'D_f, 10% D_r, frozen source copy, random teacher',
    'unsir': 'D_f, 10% D_r (labels)',
    'ssd': 'D_f, full D_train Fisher',
    'uniclun': 'D_f, 10% D_r (labels), replay buffer',
    'scrub': 'D_f, 10% D_r (labels), frozen source copy',
    'salun': 'D_f, 10% D_r (labels), forget-gradient mask',
    'rl': 'D_f, 10% D_r (labels)',
    'ga': 'D_f only',
    'l1sparse': 'D_r (labels)',
    'iu': 'D_f, 10% D_r (labels), per-example retain gradients',
    'bs': 'D_f, frozen source copy (adversarial relabelling)',
    'be': 'D_f, one extra classifier unit during unlearning',
    'lwu': 'D_f, 10% D_r (labels), Fisher on both, frozen source copy, '
    'test-time fast-weight memory',
}

# LwU fast-memory settings that won the CIFAR-100 paired selection
# (36 cells): the first two took 17/36, the third 6/36; nothing else >3.
MEMORY_PRESETS = {
    'm01_s025_c00': {'memory_lr': 0.01, 'memory_scale': 0.25, 'memory_confidence': 0.0},
    'm01_s025_c08': {'memory_lr': 0.01, 'memory_scale': 0.25, 'memory_confidence': 0.8},
    'm10_s100_c08': {'memory_lr': 0.1, 'memory_scale': 1.0, 'memory_confidence': 0.8},
}


def finite_json(value):
    """NaN/inf -> None so a diverged candidate is recorded instead of killing the run."""
    if isinstance(value, dict):
        return {key: finite_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [finite_json(item) for item in value]
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def atomic_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix='.mu-', suffix='.json', dir=path.parent)
    try:
        with os.fdopen(fd, 'w') as handle:
            json.dump(finite_json(payload), handle, indent=2, allow_nan=False)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def state_digest(state):
    digest = hashlib.sha256()
    for name, value in sorted(state.items()):
        digest.update(name.encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def cartesian(grid):
    keys = list(grid)
    return [dict(zip(keys, values)) for values in itertools.product(*(grid[key] for key in keys))]


def sync(device):
    if str(device).startswith('cuda') and torch.cuda.is_available():
        torch.cuda.synchronize(torch.device(device))


def finite_metrics(metrics):
    required = ('forget_acc', 'retain_acc', 'mia', 'kl_divergence')
    return all(metrics.get(key) is not None and np.isfinite(metrics[key]) for key in required)


class Benchmark:
    def __init__(self, args):
        self.args = args
        self.device = args.device
        self.output = Path(args.output_dir)
        self.cache = Path(args.cache_dir)
        self.output.mkdir(parents=True, exist_ok=True)
        self.cache.mkdir(parents=True, exist_ok=True)
        self.search_path = self.output / 'search.json'
        self.final_path = self.output / 'final.json'

        set_seed(args.seed)
        self.dataset_name = args.dataset
        default_backbone = 'fpecnn' if args.dataset == 'cifar10' else 'resnet18'
        backbone_name = {'cnn': 'fpecnn', 'resnet18': 'resnet18'}.get(
            args.backbone, default_backbone
        )
        self.num_classes = {'cifar10': 10, 'cifar20': 20, 'cifar100': 100, 'tinyimagenet': 200}[
            args.dataset
        ]
        self.report_methods = args.methods_list
        if args.dataset == 'cifar20':
            # CIFAR-100's own 20 superclasses are the label space.  Class
            # deletion removes one superclass, so the forget set is selected by
            # the coarse label; subclass deletion removes one of the five fine
            # classes inside a superclass, so there it is selected by the fine
            # label while the model still trains on the coarse one.
            self.dataset = get_dataset(
                'cifar20', root=args.data_dir, coarse_targets=args.forget_mode != 'subclass'
            )
        else:
            self.dataset = get_dataset(args.dataset, root=args.data_dir)
        if args.forget_mode == 'subclass' and args.dataset != 'cifar20':
            # CIFAR-100: 20 superclasses of 5 fine classes, read from the raw
            # batches.  CIFAR-10 has no shipped taxonomy, so it uses the 5x2
            # grouping defined in core/datasets/cifar10_coarse.py; TinyImageNet
            # lifts its 200 WordNet synsets to 56 superclasses of 2-12 classes.
            # Either way the wrapper trains/evaluates on the coarse label while
            # .targets keeps the fine label the forget set is selected by.
            wrapper = CoarseLabelCIFAR100
            if args.dataset == 'tinyimagenet':
                fine_to_coarse = tinyimagenet_fine_to_coarse_map(args.data_dir)
                self.num_classes = tinyimagenet_coarse_count(args.data_dir)
                wrapper = CoarseLabelDataset
            elif args.dataset == 'cifar100':
                fine_to_coarse = fine_to_coarse_map(args.data_dir)
                self.num_classes = 20
            else:
                fine_to_coarse = cifar10_fine_to_coarse_map(args.data_dir)
                self.num_classes = CIFAR10_COARSE_CLASSES
            for split in ('train_dataset', 'val_dataset', 'test_dataset'):
                setattr(self.dataset, split, wrapper(getattr(self.dataset, split), fine_to_coarse))
        loader_args = SimpleNamespace(
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            device=args.device,
            retain_ratio=args.retain_ratio,
        )
        if args.forget_mode == 'instance':
            self.loaders = build_instance_unlearning_loaders(
                self.dataset, args.num_forget, loader_args, args.seed
            )
        else:
            # 'class' forgets a fine-grained CIFAR-100 class directly (its own
            # coarse-vs-fine identity is irrelevant to plain class removal);
            # 'subclass' forgets the same way, but the model above was
            # rewrapped to train/evaluate on the coarse label, so a single
            # fine subclass is removed while its siblings under the same
            # superclass remain in the retain set.
            self.loaders = build_unlearning_loaders(
                self.dataset, args.forget_class, loader_args, args.seed
            )
        # FPECNN's default 64-unit head is the CIFAR-10 setting; it is a
        # bottleneck once the label space is large (64 units feeding 100
        # classes limits every method).
        backbone_kwargs = (
            {'hidden_size': args.fpecnn_hidden_size}
            if backbone_name == 'fpecnn'
            else {'small_input': True}
        )
        self.factory = lambda: get_backbone(
            backbone_name, num_classes=self.num_classes, **backbone_kwargs
        )

        self.training_args = SimpleNamespace(
            device=args.device,
            lr=args.source_lr,
            num_epochs=args.epochs,
            method='amnesiac',
            optimizer=args.optimizer,
            weight_decay=args.source_weight_decay,
            momentum=args.source_momentum,
            scheduler=args.source_scheduler,
            min_lr=args.source_min_lr,
            lr_milestones=args.source_lr_milestones,
            lr_gamma=args.source_lr_gamma,
            early_stopping_patience=args.early_stopping_patience,
            min_epochs=args.min_epochs,
        )
        self.protocol = {
            'protocol_version': (
                args.protocol_version
                if args.protocol_version
                else (
                    PROTOCOL_VERSION
                    if args.dataset == 'cifar100' and args.epochs == 20
                    else f'{args.dataset}-mu-v2-e{args.epochs}'
                )
            ),
            'dataset': args.dataset,
            'backbone': backbone_name,
            'optimizer': args.optimizer,
            'forget_mode': args.forget_mode,
            'forget_class': (
                args.forget_class if args.forget_mode in ('class', 'subclass') else None
            ),
            'num_forget': (args.num_forget if args.forget_mode == 'instance' else None),
            'seed': args.seed,
            'epochs': args.epochs,
            'batch_size': args.batch_size,
            'source_lr': args.source_lr,
            'source_weight_decay': args.source_weight_decay,
            'source_momentum': args.source_momentum,
            'source_scheduler': args.source_scheduler,
            'source_min_lr': args.source_min_lr,
            'source_lr_milestones': args.source_lr_milestones,
            'source_lr_gamma': args.source_lr_gamma,
            'early_stopping_patience': args.early_stopping_patience,
            'min_epochs': args.min_epochs,
            'retain_ratio': args.retain_ratio,
            'selection_split': f'{args.dataset}_train_validation_only',
            'test_split': f'{args.dataset}_official_test_once_after_selection',
            'mia': 'independent_calibration_selection_final_splits',
            'selection': args.selection,
            'utility_tolerance': args.utility_tolerance,
            'relearn_epochs': args.relearn_epochs,
            'relearn_lr': args.relearn_lr,
        }
        # Recorded beside the protocol, never inside it: `protocol` is compared
        # for equality to decide whether a cached source and an existing
        # search/final artifact may be reused, so adding a field here would
        # invalidate every finished cell.
        self.run_config = {
            'fpecnn_hidden_size': args.fpecnn_hidden_size,
            'dominance_rule': 'scale_free',
        }
        self.source_state, self.amnesiac_state = self._source()
        self.reference_state, self.retrain_time = self._reference()
        self.reference = self.factory().to(self.device)
        self.reference.load_state_dict(self.reference_state)
        self.reference.eval()
        self.grids = {**STATIC_GRIDS}
        self.grids['lwu'] = self._lwu_grid(args)

    def _cache_matches(self):
        manifest_path = self.cache / 'manifest.json'
        if not manifest_path.exists():
            return False
        try:
            return json.loads(manifest_path.read_text()).get('protocol') == self.protocol
        except (OSError, ValueError):
            return False

    def _write_manifest(self, **updates):
        path = self.cache / 'manifest.json'
        payload = {'protocol': self.protocol}
        if path.exists():
            try:
                payload.update(json.loads(path.read_text()))
            except (OSError, ValueError):
                pass
        payload['protocol'] = self.protocol
        payload.update(updates)
        atomic_json(path, payload)

    def _source(self):
        source_path = self.cache / 'source.pt'
        ledger_path = self.cache / 'amnesiac.pt'
        if self._cache_matches() and source_path.exists() and ledger_path.exists():
            state = torch.load(source_path, map_location='cpu')
            ledger = torch.load(ledger_path, map_location='cpu')
            if ledger.get('source_sha256') == state_digest(state):
                print('loaded shared source and Amnesiac ledger', flush=True)
                return state, ledger

        print(f'training one shared source ({self.args.epochs} epochs)', flush=True)
        set_seed(self.args.seed)
        model = self.factory().to(self.device)
        self.training_args.method = 'amnesiac'
        started = time.perf_counter()
        model, tracker = train_baseline_model(
            model,
            self.loaders['full_train'],
            self.loaders['full_val'],
            self.training_args,
            self.args.forget_class if self.args.forget_mode == 'class' else -1,
        )
        sync(self.device)
        elapsed = time.perf_counter() - started
        state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
        update_sum = tracker.update_sum or {
            name: torch.zeros_like(parameter, device='cpu')
            for name, parameter in model.named_parameters()
        }
        ledger = {
            'update_sum': {
                name: value.detach().cpu().clone() for name, value in update_sum.items()
            },
            'num_recorded': tracker.num_recorded,
            'source_sha256': state_digest(state),
        }
        torch.save(state, source_path)
        torch.save(ledger, ledger_path)
        self._write_manifest(
            source_train_time=elapsed,
            source_training=model.training_summary,
            source_sha256=ledger['source_sha256'],
        )
        return state, ledger

    def _reference(self):
        path = self.cache / 'retrain_reference.pt'
        manifest_path = self.cache / 'manifest.json'
        if self._cache_matches() and path.exists():
            manifest = json.loads(manifest_path.read_text())
            if manifest.get('retrain_time') is not None:
                print('loaded shared Retrain reference', flush=True)
                return torch.load(path, map_location='cpu'), manifest['retrain_time']

        print(f'training same-recipe retain-only reference ({self.args.epochs} epochs)', flush=True)
        set_seed(self.args.seed)
        model = self.factory().to(self.device)
        self.training_args.method = 'retrain'
        sync(self.device)
        started = time.perf_counter()
        model, _ = train_baseline_model(
            model,
            self.loaders['retain_full'],
            self.loaders['retain_val'],
            self.training_args,
            self.args.forget_class if self.args.forget_mode == 'class' else -1,
        )
        sync(self.device)
        elapsed = time.perf_counter() - started
        state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
        torch.save(state, path)
        self._write_manifest(
            retrain_time=elapsed,
            retrain_training=model.training_summary,
            retrain_sha256=state_digest(state),
        )
        return state, elapsed

    def _fresh_source(self):
        model = self.factory().to(self.device)
        model.load_state_dict(self.source_state)
        return model

    @staticmethod
    def _lwu_grid(args=None):
        # The slow zones use the Fisher importance map, the orthogonal
        # Zone-C rule, dominance zoning and no context lr; the fast memory
        # takes the presets above.  Instance and subclass deletion get a
        # looser forget quantile, i.e. more of the forget-related parameters
        # are reset; class deletion keeps the tight quantile that already
        # reaches D_f = 0.
        wide = args is not None and args.forget_mode in ('instance', 'subclass')
        return {
            'forget_quantile': [0.9, 0.99] if wide else [0.99],
            # The forget-priority rule is not searched any more: it reinitialises
            # every forget-active weight, and a direct sweep showed it pinned at
            # chance accuracy even with the full retain set and 30 repair epochs.
            'instance_forget_priority': [False],
            'dominance_margin': [1.0, 3.0],
            # 30 epochs is in the grid because the repair budget, not the zoning,
            # was the larger limit: on ResNet-18 instance, widening it took
            # retain accuracy from 31 to 75.
            'distill_epochs': [10, 30],
            'distill_lr': [1e-3, 1e-2],
            'sdft_lambda': [0.5],
            'weight_align': [False, True],
            'memory_preset': ['m01_s025_c08', 'm10_s100_c08'],
        }

    def evaluate(self, model, split):
        if split == 'selection':
            member = self.loaders['mia_member_select']
            nonmember = self.loaders['mia_nonmember_select']
            forget_eval = self.loaders['forget_val']
            retain_eval = self.loaders['retain_val']
            overall = self.loaders['full_val']
        elif split == 'final':
            member = self.loaders['mia_member_final']
            nonmember = (
                self.loaders['mia_nonmember_final']
                if self.args.forget_mode == 'instance'
                else self.loaders['forget_test']
            )
            forget_eval = self.loaders['forget_test']
            retain_eval = self.loaders['retain_test']
            overall = self.loaders['full_test']
        else:
            raise ValueError(split)
        metrics = UnlearningEvaluator(
            model, self.device, test_time_adapt=isinstance(model, ZoneDecomposition)
        ).evaluate(
            forget_loader=member,
            retain_loader=self.loaders['retain_small'],
            test_loader=overall,
            retrained_model=self.reference,
            non_member_loader=nonmember,
            forget_eval_loader=forget_eval,
            retain_eval_loader=retain_eval,
            mia_calibration_member_loader=self.loaders['mia_member_cal'],
            mia_calibration_non_member_loader=self.loaders['mia_nonmember_cal'],
        )
        return {name: float(value) for name, value in metrics.items() if value is not None}

    def selection_gold(self):
        return self.evaluate(self.reference, 'selection')

    @staticmethod
    def gap(metrics, gold):
        return float(
            np.mean(
                [
                    abs(metrics['forget_acc'] - gold['forget_acc']),
                    abs(metrics['retain_acc'] - gold['retain_acc']),
                    abs(metrics['mia'] - gold['mia']),
                ]
            )
        )

    def annotate(self, row, gold):
        """Utility budget and forgetting score of one valid candidate."""
        metrics = row['selection_metrics']
        tolerance = self.args.utility_tolerance
        row['forget_gap'] = abs(metrics['forget_acc'] - gold['forget_acc'])
        row['feasible'] = bool(
            metrics['retain_acc'] >= gold['retain_acc'] - tolerance
            and metrics['test_acc'] >= gold['test_acc'] - tolerance
        )
        return row

    def select(self, valid, gold):
        """Pick one candidate.

        'utility' (default): among candidates whose retain and test accuracy
        stay within ``utility_tolerance`` points of Retrain, take the one whose
        D_f accuracy is closest to Retrain's (ties: higher retain accuracy).
        The reported gap metric is not part of the criterion, so it cannot
        be tuned on.  If no candidate meets the budget, the least damaging one
        is returned and flagged ``feasible: False``.
        'gap': the legacy rule, minimum mean |delta| to Retrain over
        D_f / D_r / MIA.
        """
        for row in valid:
            self.annotate(row, gold)
        if self.args.selection == 'gap':
            return min(valid, key=lambda row: row['gap_to_retrain'])
        pool = [row for row in valid if row['feasible']]
        if pool:
            return min(
                pool, key=lambda row: (row['forget_gap'], -row['selection_metrics']['retain_acc'])
            )
        return max(
            valid,
            key=lambda row: (
                row['selection_metrics']['retain_acc'] + row['selection_metrics']['test_acc']
            ),
        )

    def _build(self, name, model, config, selection=False, fisher=None):
        if name == 'finetune':
            return FineTune(model=model, device=self.device, **config)
        if name == 'badteacher':
            return BadTeacher(model=model, device=self.device, **config)
        if name == 'unsir':
            return UNSIR(model=model, device=self.device, **config)
        if name == 'ssd':
            return SSD(model=model, device=self.device, full_fisher=fisher, **config)
        if name == 'uniclun':
            return UniCLUN(model=model, device=self.device, **config)
        if name == 'scrub':
            return SCRUB(model=model, device=self.device, **config)
        if name == 'salun':
            return SalUn(
                model=model,
                device=self.device,
                num_classes=self.num_classes,
                seed=self.args.seed,
                **config,
            )
        if name == 'rl':
            return RandomLabel(
                model=model,
                device=self.device,
                num_classes=self.num_classes,
                seed=self.args.seed,
                **config,
            )
        if name == 'ga':
            return GradientAscent(model=model, device=self.device, **config)
        if name == 'l1sparse':
            return L1Sparse(model=model, device=self.device, **config)
        if name == 'iu':
            return InfluenceUnlearning(model=model, device=self.device, **config)
        if name == 'bs':
            return BoundaryShrink(model=model, device=self.device, **config)
        if name == 'be':
            return BoundaryExpand(model=model, device=self.device, **config)
        if name == 'lwu':
            return LwU(
                backbone=model,
                num_classes=self.num_classes,
                importance='fisher',
                zone_c='orthogonal',
                retain_quantile=0.9,
                forget_quantile=config['forget_quantile'],
                per_instance_forget_fisher=self.args.forget_mode == 'instance',
                distill_epochs=config['distill_epochs'],
                distill_lr=config['distill_lr'],
                temperature=4.0,
                ce_weight=1.0,
                zone_rule='dominance',
                dominance_margin=config['dominance_margin'],
                context_window=1,
                context_lr=0.0,
                context_temperature=0.25,
                context_decay=1.0,
                context_momentum=0.9,
                context_steps=1,
                weight_align=config['weight_align'],
                sdft_lambda=config['sdft_lambda'],
                mask_forgotten_classes=self.args.forget_mode == 'class',
                instance_forget_priority=config['instance_forget_priority'],
                persist_memory_across_streams=False,
                device=self.device,
                **MEMORY_PRESETS[config['memory_preset']],
            )
        raise KeyError(name)

    def _apply(self, name, method, config, selection=False):
        if name == 'l1sparse':
            # The sparse variant of fine-tuning, so it gets fine-tuning's access.
            method.unlearn(self.loaders['forget_train'], self.loaders['retain_full'])
            return method.model
        if name == 'finetune':
            method.unlearn(self.loaders['forget_train'], self.loaders['retain_full'])
            return method.model
        if name == 'ssd':
            if not selection:
                method.prepare(self.loaders['full_train'])
            method.unlearn(self.loaders['forget_train'])
            return method.model
        if name == 'lwu':
            method.unlearn(
                self.loaders['forget_train'],
                self.loaders['retain_small'],
                eta_c=config.get('eta_c', 1e-2),
                n_c=config.get('n_c', 5),
                delta_c=config.get('delta_c', 0.1),
            )
            return method
        method.unlearn(self.loaders['forget_train'], self.loaders['retain_small'])
        return method.model

    def search(self, methods):
        tunable = [name for name in methods if name in self.grids]
        state = {
            'protocol': self.protocol,
            'run_config': self.run_config,
            'grids': self.grids,
            'gold_selection': self.selection_gold(),
            'results': {},
        }
        if self.search_path.exists():
            previous = json.loads(self.search_path.read_text())
            if previous.get('protocol') == self.protocol:
                state = previous
        gold = state['gold_selection']

        # Reusing Fisher is an acceleration for grid selection only.  The final
        # reported SSD run recomputes it inside the measured interval.
        fisher = None
        if 'ssd' in tunable:
            probe = self._fresh_source()
            fisher = SSD(probe, device=self.device).prepare(self.loaders['full_train'])

        for name in tunable:
            all_candidates = cartesian(self.grids[name])
            candidates = [
                config
                for index, config in enumerate(all_candidates)
                if index % self.args.candidate_shard_count == self.args.candidate_shard_index
            ]
            rows = state['results'].setdefault(name, [])
            complete = {json.dumps(row['config'], sort_keys=True) for row in rows}
            print(f'\n{name}: {len(complete)}/{len(candidates)} candidates complete', flush=True)
            for index, config in enumerate(candidates, 1):
                key = json.dumps(config, sort_keys=True)
                if key in complete:
                    continue
                set_seed(self.args.seed)
                model = self._fresh_source()
                method = self._build(
                    name, model, config, selection=True, fisher=fisher if name == 'ssd' else None
                )
                sync(self.device)
                started = time.perf_counter()
                error = None
                try:
                    unlearned = self._apply(name, method, config, selection=True)
                    sync(self.device)
                    elapsed = max(
                        0.0,
                        time.perf_counter()
                        - started
                        - float(getattr(method, 'diagnostic_time', 0.0)),
                    )
                    metrics = self.evaluate(unlearned, 'selection')
                    valid = finite_metrics(metrics)
                except (FloatingPointError, RuntimeError, ValueError) as exc:
                    sync(self.device)
                    elapsed = time.perf_counter() - started
                    metrics = {}
                    valid = False
                    error = f'{type(exc).__name__}: {exc}'
                row = {
                    'config': config,
                    'selection_metrics': metrics,
                    'selection_time': elapsed,
                    'gap_to_retrain': self.gap(metrics, gold) if valid else None,
                    'valid': valid,
                }
                if valid:
                    self.annotate(row, gold)
                if error is not None:
                    row['error'] = error
                if hasattr(method, 'diagnostics'):
                    row['diagnostics'] = method.diagnostics
                rows = [old for old in rows if json.dumps(old['config'], sort_keys=True) != key]
                rows.append(row)
                state['results'][name] = rows
                atomic_json(self.search_path, state)
                if valid:
                    print(
                        f'  [{index}/{len(candidates)}] gap={row["gap_to_retrain"]} '
                        f'Df={metrics["forget_acc"]:.2f} '
                        f'Dr={metrics["retain_acc"]:.2f} '
                        f'MIA={metrics["mia"]:.2f}',
                        flush=True,
                    )
                else:
                    print(f'  [{index}/{len(candidates)}] INVALID {error}', flush=True)

        best = dict(state.get('best', {}))
        for name in tunable:
            expected = sum(
                1
                for index, _ in enumerate(cartesian(self.grids[name]))
                if index % self.args.candidate_shard_count == self.args.candidate_shard_index
            )
            attempted = state['results'].get(name, [])
            valid = [row for row in attempted if row.get('valid')]
            if len(attempted) != expected:
                raise RuntimeError(f'{name} grid incomplete: {len(attempted)}/{expected}')
            if not valid:
                raise RuntimeError(f'{name} grid has no finite candidate')
            best[name] = self.select(valid, gold)
        state['best'] = best
        state['selection_rule'] = self.args.selection
        state['complete_methods'] = sorted(best)
        atomic_json(self.search_path, state)
        return state

    def _final_method(self, name, config):
        self._last_diagnostics = None
        if name == 'baseline':
            return self._fresh_source(), 0.0
        if name == 'retrain':
            model = self.factory().to(self.device)
            model.load_state_dict(self.reference_state)
            return model, self.retrain_time
        if name == 'amnesiac':
            model = self._fresh_source()
            tracker = Amnesiac(model, device=self.device)
            tracker.update_sum = {
                key: value.to(self.device)
                for key, value in self.amnesiac_state['update_sum'].items()
            }
            tracker.num_recorded = self.amnesiac_state['num_recorded']
            sync(self.device)
            started = time.perf_counter()
            tracker.unlearn()
            sync(self.device)
            return tracker.model, time.perf_counter() - started

        set_seed(self.args.seed)
        model = self._fresh_source()
        method = self._build(name, model, config, selection=False)
        sync(self.device)
        started = time.perf_counter()
        unlearned = self._apply(name, method, config, selection=False)
        sync(self.device)
        self._last_diagnostics = getattr(method, 'diagnostics', None)
        elapsed = max(
            0.0,
            time.perf_counter() - started - float(getattr(method, 'diagnostic_time', 0.0)),
        )
        return unlearned, elapsed

    def final(self, methods):
        tunable = [name for name in methods if name in self.grids]
        search = {'best': {}}
        if tunable:
            if not self.search_path.exists():
                raise RuntimeError('run the complete validation grid before final evaluation')
            search = json.loads(self.search_path.read_text())
            if search.get('protocol') != self.protocol:
                raise RuntimeError('search artifact is from another protocol')
            for name in tunable:
                expected = len(cartesian(self.grids[name]))
                attempted = search.get('results', {}).get(name, [])
                if len(attempted) != expected or name not in search.get('best', {}):
                    raise RuntimeError(
                        f'{name} validation grid is incomplete: {len(attempted)}/{expected}'
                    )
        state = {'protocol': self.protocol, 'run_config': self.run_config, 'results': {}}
        if self.final_path.exists():
            previous = json.loads(self.final_path.read_text())
            if previous.get('protocol') == self.protocol:
                state = previous
                state['run_config'] = self.run_config

        for name in methods:
            if name in state['results']:
                continue
            config = search.get('best', {}).get(name, {}).get('config', {})
            print(f'final test: {name} {config}', flush=True)
            model, elapsed = self._final_method(name, config)
            metrics = self.evaluate(model, 'final')
            metrics['unlearn_time'] = float(elapsed)
            if not finite_metrics(metrics):
                raise RuntimeError(f'{name} produced non-finite final metrics: {metrics}')
            metrics.update(self.relearn_probe(model))
            state['results'][name] = {
                'config': config,
                'feasible': search.get('best', {}).get(name, {}).get('feasible'),
                'access': ACCESS.get(name),
                'metrics': metrics,
                'source_sha256': state_digest(self.source_state),
            }
            if self._last_diagnostics is not None:
                state['results'][name]['diagnostics'] = self._last_diagnostics
            atomic_json(self.final_path, state)
            self.write_table(state)
            print(
                f'  Df={metrics["forget_acc"]:.2f} Dr={metrics["retain_acc"]:.2f} '
                f'MIA={metrics["mia"]:.2f} time={elapsed:.2f}s '
                f'KL={metrics["kl_divergence"]:.4f}',
                flush=True,
            )
        state['complete'] = set(methods).issubset(state['results'])
        atomic_json(self.final_path, state)
        self.write_table(state)
        return state

    def relearn_probe(self, model):
        """Recovery attack: a few epochs of Adam on D_f, D_f accuracy after each.

        A method that only suppressed the forget outputs snaps back in one
        epoch; Retrain's curve is the reference for genuine forgetting.  Runs
        on a copy, so the evaluated model is untouched.
        """
        if self.args.relearn_epochs <= 0:
            return {}
        probe = copy.deepcopy(model)
        adapt = isinstance(probe, LwU)
        optimizer = torch.optim.Adam(probe.parameters(), lr=self.args.relearn_lr)
        set_seed(self.args.seed)
        curve = []
        for _ in range(self.args.relearn_epochs):
            probe.train()
            for inputs, targets in self.loaders['forget_train']:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                optimizer.zero_grad(set_to_none=True)
                torch.nn.functional.cross_entropy(probe(inputs), targets).backward()
                optimizer.step()
            curve.append(
                100.0
                * evaluate_task(
                    probe, self.loaders['forget_test'], self.device, test_time_adapt=adapt
                )
            )
        return {'relearn_forget_acc': curve}

    def write_table(self, state):
        labels = {
            'baseline': 'Baseline (original)',
            'retrain': 'Retrain (gold standard)',
            'finetune': r'Fine-tune on D_r',
            'badteacher': 'Bad Teacher',
            'amnesiac': 'Amnesiac',
            'unsir': 'UNSIR',
            'ssd': 'SSD',
            'uniclun': 'UniCLUN †',
            'scrub': 'SCRUB (NeurIPS23)',
            'salun': 'SalUn (ICLR24)',
            'rl': 'RL (random labels)',
            'ga': 'GA (NegGrad)',
            'l1sparse': 'l1-sparse',
            'iu': 'IU (influence)',
            'bs': 'BS (boundary shrink)',
            'be': 'BE (boundary expand)',
            'lwu': 'LwU (ours)',
        }
        lines = [
            f'# {self.dataset_name.upper()} machine unlearning (one seed)',
            '',
            f'Protocol: `{self.protocol["protocol_version"]}`. Hyperparameters are selected on '
            'validation data; this table contains the single final test evaluation.',
            '',
            '| Method | D_f Acc ↓ | D_r Acc ↑ | MIA → 50 | Time (s) ↓ | KL (weights) ↓ |',
            '|---|---:|---:|---:|---:|---:|',
        ]
        for name in self.report_methods:
            record = state['results'].get(name)
            if record is None:
                lines.append(f'| {labels[name]} | *pending* | | | | |')
                continue
            metrics = record['metrics']
            lines.append(
                f'| {labels[name]} | {metrics["forget_acc"]:.2f} | '
                f'{metrics["retain_acc"]:.2f} | {metrics["mia"]:.2f} | '
                f'{metrics["unlearn_time"]:.2f} | {metrics["kl_divergence"]:.4f} |'
            )
        (self.output / 'MACHINE_UNLEARNING_PARTIAL.md').write_text('\n'.join(lines) + '\n')


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--phase', choices=['prepare', 'search', 'final', 'all'], default='all')
    parser.add_argument('--methods', default=None)
    parser.add_argument(
        '--dataset', choices=['cifar10', 'cifar20', 'cifar100', 'tinyimagenet'], default='cifar100'
    )
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--data_dir', default='./data')
    parser.add_argument('--output_dir', default=None)
    parser.add_argument('--cache_dir', default=None)
    parser.add_argument('--forget_class', type=int, default=None)
    parser.add_argument(
        '--forget_mode',
        choices=['class', 'instance', 'subclass'],
        default='class',
        help="'subclass': trains/evaluates on the coarse "
        "superclasses (cifar100: 20 of 5 fine classes; "
        "cifar10: 5 of 2; tinyimagenet: 56 WordNet groups of "
        "2-12) while forget_class selects one fine "
        "class to remove -- its siblings under the same "
        "superclass stay in the retain set",
    )
    parser.add_argument(
        '--num_forget',
        type=int,
        default=500,
        help='Class-balanced training instances deleted in instance mode',
    )
    parser.add_argument(
        '--backbone',
        choices=['cnn', 'resnet18'],
        default=None,
        help='cnn=FPECNN (~2M params, matches the CIFAR-10 default), '
        'resnet18=~11M params (the CIFAR-100 default). '
        'Unset keeps the historical per-dataset default.',
    )
    parser.add_argument(
        '--optimizer',
        choices=['sgd', 'adam', 'muon'],
        default='adam',
        help='Optimizer for the shared source model and the '
        'retrain-oracle training (unlearning-method-internal '
        'optimizers are unchanged, matching each published recipe)',
    )
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--num_workers', type=int, default=3)
    parser.add_argument('--source_lr', type=float, default=1e-3)
    parser.add_argument('--source_weight_decay', type=float, default=0.0)
    parser.add_argument('--source_momentum', type=float, default=0.9)
    parser.add_argument(
        '--source_scheduler', choices=['plateau', 'cosine', 'multistep'], default='plateau'
    )
    parser.add_argument('--source_min_lr', type=float, default=0.0)
    parser.add_argument(
        '--source_lr_milestones',
        default='60,120',
        help='Comma-separated epochs for the multistep scheduler',
    )
    parser.add_argument('--source_lr_gamma', type=float, default=0.2)
    parser.add_argument(
        '--early_stopping_patience',
        type=int,
        default=10,
        help='Stop after this many non-improving validation epochs; '
        '0 disables stopping while still restoring the best epoch',
    )
    parser.add_argument(
        '--min_epochs', type=int, default=0, help='Do not early-stop before this epoch'
    )
    parser.add_argument('--retain_ratio', type=float, default=0.1)
    parser.add_argument(
        '--selection',
        choices=['utility', 'gap'],
        default='utility',
        help="'utility': closest D_f accuracy to Retrain among "
        "candidates within --utility_tolerance points of "
        "Retrain's retain/test accuracy; 'gap': legacy "
        "minimum mean |delta| over D_f/D_r/MIA",
    )
    parser.add_argument(
        '--utility_tolerance',
        type=float,
        default=2.0,
        help='Allowed retain/test accuracy drop (points) vs Retrain',
    )
    parser.add_argument(
        '--relearn_epochs',
        type=int,
        default=5,
        help='Recovery-attack epochs on D_f in the final phase; 0 disables',
    )
    parser.add_argument('--relearn_lr', type=float, default=1e-4)
    parser.add_argument(
        '--fpecnn_hidden_size',
        type=int,
        default=64,
        help="Width of FPECNN's head. 64 is the CIFAR-10 default; larger "
        'label spaces need a wider head',
    )
    parser.add_argument(
        '--protocol_version',
        default=None,
        help='Override the recorded protocol string (also the cache key)',
    )
    parser.add_argument('--candidate_shard_count', type=int, default=1)
    parser.add_argument('--candidate_shard_index', type=int, default=0)
    args = parser.parse_args()

    def csv_values(value, cast):
        return (
            None
            if value is None
            else [cast(item.strip()) for item in value.split(',') if item.strip()]
        )

    args.source_lr_milestones = csv_values(args.source_lr_milestones, int) or []
    if args.candidate_shard_count < 1:
        parser.error('--candidate_shard_count must be positive')
    if not 0 <= args.candidate_shard_index < args.candidate_shard_count:
        parser.error('--candidate_shard_index must be in [0, shard_count)')
    if args.phase in {'all', 'final'} and args.candidate_shard_count != 1:
        parser.error('candidate sharding is valid only with --phase search')
    if args.output_dir is None:
        args.output_dir = (
            './results_unlearning_cifar100_v3_e20'
            if args.dataset == 'cifar100'
            else f'./results_unlearning_{args.dataset}_e{args.epochs}'
        )
    if args.cache_dir is None:
        args.cache_dir = os.path.join(args.output_dir, 'cache')
    if args.forget_mode in ('class', 'subclass') and args.forget_class is None:
        args.forget_class = 0 if args.dataset == 'cifar10' else 42
    if args.forget_mode == 'instance' and args.num_forget < 6:
        parser.error('--num_forget must be at least 6 in instance mode')
    if args.early_stopping_patience < 0:
        parser.error('--early_stopping_patience must be non-negative')
    if not 0 <= args.min_epochs <= args.epochs:
        parser.error('--min_epochs must be in [0, epochs]')
    if args.source_scheduler == 'multistep' and not args.source_lr_milestones:
        parser.error('--source_lr_milestones cannot be empty for multistep')
    if args.methods is None:
        selected = DEFAULT_METHODS + ['lwu']
        args.methods = ','.join(selected)
    methods = [name.strip().lower() for name in args.methods.split(',') if name.strip()]
    unknown = set(methods) - set(DEFAULT_METHODS + OPTIONAL_METHODS)
    if unknown:
        parser.error(f'unknown methods: {sorted(unknown)}')
    args.methods_list = methods
    return args


def main():
    args = parse_args()
    benchmark = Benchmark(args)
    if args.phase == 'prepare':
        print('shared source, retrain reference, and importance thresholds ready', flush=True)
    elif args.phase == 'search':
        benchmark.search(args.methods_list)
    elif args.phase == 'final':
        benchmark.final(args.methods_list)
    else:
        # Report methods with no grid immediately, then complete and report the
        # faster tunable methods one at a time.  This preserves a single final
        # test evaluation per method while keeping the partial table useful.
        direct = [name for name in args.methods_list if name not in benchmark.grids]
        if direct:
            benchmark.final(direct)
        requested = set(args.methods_list)
        for name in SEARCH_PRIORITY:
            if name not in requested:
                continue
            benchmark.search([name])
            benchmark.final([name])


if __name__ == '__main__':
    main()
