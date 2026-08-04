"""
Tests that verify this repository runs and that its metrics match the
definitions given in the paper.

Run with:  pytest -q

These tests use the synthetic RTI fixture and therefore need no downloaded
data. They check that the code paths execute and that the metric functions
compute the stated formulas; they do not and cannot reproduce reported
accuracies, which require the real datasets and GPU time.
"""

import numpy as np
import pytest

from lwu.datasets.continual_datasets import get_dataset
from lwu.evaluation.metrics import AccuracyMatrix, evaluate_task


# --------------------------------------------------------------------------
# Dataset factory
# --------------------------------------------------------------------------

def test_factory_accepts_entry_point_kwargs():
    """The training scripts pass data_dir and scenario; both must be accepted."""
    ds = get_dataset('rti', data_dir='./data/rti', num_tasks=10, scenario='task')
    assert ds.num_tasks == 10
    assert ds.scenario == 'task'


def test_rti_supports_ten_task_protocol():
    """The reported RTI experiments use 10 tasks, so the loader must build 10."""
    ds = get_dataset('rti', root='./data/rti', num_tasks=10)
    assert len(ds.task_classes) == 10
    assert ds.num_classes == 10

    train_loader, _, _ = ds.get_task_loaders(0, batch_size=8, num_workers=0)
    x, y = next(iter(train_loader))
    assert x.shape[0] > 0


def test_cifar100_vit_key_is_registered():
    """reproduce_table2.sh uses this key; it must resolve in the registry."""
    with pytest.raises(Exception) as exc:
        get_dataset('cifar100-vit', root='./data', num_tasks=10, download=False)
    # Any failure is acceptable except "unknown dataset", which would mean the
    # key never resolved at all.
    assert 'Unknown dataset' not in str(exc.value)


# --------------------------------------------------------------------------
# Metrics: the formulas in the paper
# --------------------------------------------------------------------------

def _filled_matrix():
    T = 4
    A = np.array([
        [0.90, 0.00, 0.00, 0.00],
        [0.86, 0.88, 0.00, 0.00],
        [0.84, 0.85, 0.91, 0.00],
        [0.82, 0.83, 0.89, 0.90],
    ])
    pre = {1: 0.10, 2: 0.12, 3: 0.09}

    m = AccuracyMatrix(T)
    for t in range(T):
        m.update(t, {j: A[t, j] for j in range(t + 1)})
    for t, v in pre.items():
        m.record_pre_task_accuracy(t, v)
    return m, A, pre, T


def test_plasticity_matches_paper_formula():
    m, A, pre, T = _filled_matrix()
    expected = np.mean([
        (A[t, t] - pre[t]) / (1 - pre[t]) for t in range(1, T)
    ])
    assert m.get_plasticity() == pytest.approx(expected)


def test_stability_matches_paper_formula():
    m, A, pre, T = _filled_matrix()
    expected = 1 - np.mean([A[t, t] - A[T - 1, t] for t in range(T - 1)])
    assert m.get_stability() == pytest.approx(expected)


def test_stability_equals_one_plus_bwt():
    """S = 1 + BWT holds by construction; this guards against drift."""
    m, _, _, _ = _filled_matrix()
    assert m.get_stability() == pytest.approx(1 + m.get_bwt())


def test_ps_is_harmonic_mean():
    m, _, _, _ = _filled_matrix()
    p, s = m.get_plasticity(), m.get_stability()
    assert m.get_ps() == pytest.approx(2 * p * s / (p + s))


def test_ps_refuses_to_guess_missing_pre_task_accuracies():
    """
    Plasticity needs A[t-1, t]. If a run never recorded it, PS must fail
    loudly rather than silently return a number computed from a different
    definition.
    """
    T = 3
    m = AccuracyMatrix(T)
    for t in range(T):
        m.update(t, {j: 0.8 for j in range(t + 1)})
    with pytest.raises(ValueError, match='pre-task accuracies'):
        m.get_ps()


# --------------------------------------------------------------------------
# Task-IL vs Class-IL evaluation
# --------------------------------------------------------------------------

def test_task_il_masking_restricts_prediction_to_task_classes():
    """
    Task-IL assumes task identity is known at test time, so the argmax must be
    taken over the task's own classes. Without masking, Task-IL and Class-IL
    are the same computation.
    """
    import torch
    from torch.utils.data import DataLoader, TensorDataset

    class FixedLogits(torch.nn.Module):
        """Always most confident about class 3."""
        def forward(self, x):
            logits = torch.zeros(x.shape[0], 4)
            logits[:, 3] = 10.0
            logits[:, 1] = 5.0
            return logits

    x = torch.randn(8, 2)
    y = torch.ones(8, dtype=torch.long)  # true label is class 1
    loader = DataLoader(TensorDataset(x, y), batch_size=4)
    model = FixedLogits()

    # Class-IL: argmax over everything picks class 3 -> all wrong.
    assert evaluate_task(model, loader, device='cpu') == pytest.approx(0.0)

    # Task-IL: this task owns classes {0, 1}, so class 1 wins -> all correct.
    acc = evaluate_task(model, loader, device='cpu',
                        task_id=0, task_classes=[0, 1])
    assert acc == pytest.approx(1.0)
