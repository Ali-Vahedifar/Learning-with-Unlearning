"""
Checks that LwU does what the method section says it does.

These test properties the method section claims, not accuracies: that the four
zones partition the network, that the Zone C update really is orthogonal to
the retain gradient, that displacement stays inside its budget, and that
test-time adaptation does not touch the stored weights.
"""

from copy import deepcopy

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from LwU.zones import ZoneDecomposition
from LwU.teacher_repair import TeacherRepair
from LwU.dominance import DominanceZones
from LwU.context_memory import ContextMemory
from LwU.self_distillation import SelfDistillation
from LwU.fixed_parameter_expansion import (
    ZoneDFixedParameterExpansion,
    ZoneFixedParameterExpansion,
)
from core.models.backbones import FPECNN
from core.evaluation.metrics import evaluate_task


def _toy_model(seed=0, threshold_mode='normalized', tau=0.3):
    torch.manual_seed(seed)
    backbone = nn.Sequential(nn.Linear(8, 16), nn.ReLU(), nn.Linear(16, 4))
    return ZoneDecomposition(
        backbone=backbone,
        num_classes=4,
        tau_r=tau,
        tau_f=tau,
        threshold_mode=threshold_mode,
        device='cpu',
    )


def _loader(n=32, num_classes=4, seed=0):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(n, 8, generator=g)
    y = torch.randint(0, num_classes, (n,), generator=g)
    return DataLoader(TensorDataset(x, y), batch_size=8)


# --------------------------------------------------------------------------
# Eq. 6-7: the SSV score
# --------------------------------------------------------------------------


def test_ssv_is_first_order_term_plus_curvature_term():
    """phi_i = -g_i * theta_i + (1/2) * theta_i^2 * F_ii, under the diagonal
    Fisher approximation stated in Section 2.1.1."""
    model = _toy_model()
    loader = _loader()

    fisher = model.compute_fisher_information(loader)
    ssv = model.compute_ssv(loader, fisher)

    # Recompute the same quantity independently from gradients.
    grads = {n: torch.zeros_like(p) for n, p in model.backbone.named_parameters()}
    total = 0
    crit = nn.CrossEntropyLoss()
    model.backbone.eval()
    for xb, yb in loader:
        model.backbone.zero_grad()
        crit(model.backbone(xb), yb).backward()
        for n, p in model.backbone.named_parameters():
            grads[n] += p.grad.data * xb.size(0)
        total += xb.size(0)
    for n in grads:
        grads[n] /= total

    for n, p in model.backbone.named_parameters():
        expected = -grads[n] * p.data + 0.5 * fisher[n] * p.data**2
        assert torch.allclose(ssv[n], expected, atol=1e-5)


def test_interaction_ssv_matches_explicit_empirical_fisher():
    """The matrix-free F-theta product equals an explicit dense toy Fisher."""
    model = ZoneDecomposition(
        nn.Linear(3, 2, bias=False), num_classes=2, tau_r=0.0, tau_f=0.0, device='cpu'
    )
    loader = DataLoader(
        TensorDataset(torch.tensor([[1.0, -2.0, 0.5], [0.3, 0.7, -1.0]]), torch.tensor([0, 1])),
        batch_size=1,
    )
    actual = model.compute_ssv_with_interaction(loader)['weight'].flatten()

    gradients = []
    for inputs, targets in loader:
        model.backbone.zero_grad(set_to_none=True)
        F.cross_entropy(model.backbone(inputs), targets).backward()
        gradients.append(model.backbone.weight.grad.detach().flatten().clone())
    gradient = torch.stack(gradients).mean(0)
    fisher = sum(torch.outer(value, value) for value in gradients) / len(gradients)
    theta = model.backbone.weight.detach().flatten()
    expected = -gradient * theta + 0.5 * theta * (fisher @ theta)
    assert torch.allclose(actual, expected, atol=1e-6)


@pytest.mark.parametrize(
    'estimator',
    [
        'random_mask',
        'magnitude',
        'fisher_retain',
        'fisher_forget',
        'dual_fisher',
        'ssv_without_interaction',
        'ssv_retain',
        'ssv_forget',
        'dual_ssv',
    ],
)
def test_importance_ablation_estimators_form_exact_partition(estimator):
    method = TeacherRepair(
        nn.Linear(8, 4),
        num_classes=4,
        importance=estimator,
        retain_quantile=0.7,
        forget_quantile=0.7,
        distill_epochs=0,
        device='cpu',
    )
    method.identify_zones(_loader(), _loader(seed=1))
    diagnostics = method.reconstruction_diagnostics()
    assert diagnostics['overlap_or_gap_parameters'] == 0
    assert diagnostics['max_abs_reconstruction_error'] == 0.0


def test_masks_threshold_absolute_ssv():
    """Eq. 8 thresholds |phi|, so a large negative score is important too."""
    model = _toy_model(threshold_mode='absolute', tau=1e-4)
    model.identify_zones(_loader(), _loader(seed=1))

    for name, p in model.backbone.named_parameters():
        in_mask = model.zone_masks['A'][name] | model.zone_masks['C'][name]
        assert torch.equal(in_mask, model.phi_r[name].abs() > model.tau_r)
        forget_mask = model.zone_masks['B'][name] | model.zone_masks['C'][name]
        assert torch.equal(forget_mask, model.phi_f[name].abs() > model.tau_f)


# --------------------------------------------------------------------------
# Eq. 9-12: the partition
# --------------------------------------------------------------------------


def test_four_zones_are_disjoint_and_cover_everything():
    model = _toy_model()
    model.identify_zones(_loader(), _loader(seed=1))

    for name, p in model.backbone.named_parameters():
        A = model.zone_masks['A'][name]
        B = model.zone_masks['B'][name]
        C = model.zone_masks['C'][name]
        D = model.zone_masks['D'][name]

        stacked = torch.stack([A, B, C, D]).long().sum(dim=0)
        assert (stacked == 1).all(), f"{name}: zones overlap or leave gaps"

    diagnostics = model.zone_reconstruction_diagnostics()
    assert diagnostics['overlap_or_gap_parameters'] == 0
    assert diagnostics['max_abs_reconstruction_error'] == 0.0


@pytest.mark.parametrize('importance', ['ssv', 'fisher'])
@pytest.mark.parametrize('zone_c', ['granular', 'orthogonal'])
def test_teacher_repair_ablation_zones_reconstruct_exactly(importance, zone_c):
    torch.manual_seed(0)
    method = TeacherRepair(
        nn.Sequential(nn.Linear(8, 16), nn.ReLU(), nn.Linear(16, 4)),
        num_classes=4,
        importance=importance,
        zone_c=zone_c,
        retain_quantile=0.7,
        forget_quantile=0.7,
        distill_epochs=0,
        device='cpu',
    )
    method.identify_zones(_loader(), _loader(seed=1))
    diagnostics = method.reconstruction_diagnostics()
    assert diagnostics['overlap_or_gap_parameters'] == 0
    assert diagnostics['max_abs_reconstruction_error'] == 0.0
    assert sum(method.get_zone_statistics()['overall'][zone] for zone in 'ABCD') == pytest.approx(
        100.0
    )


def test_looser_forget_quantile_monotonically_expands_forget_active_zones():
    loader = _loader(n=32, num_classes=4, seed=29)
    seed = nn.Linear(8, 4, bias=False)
    strict = SelfDistillation(
        deepcopy(seed),
        num_classes=4,
        forget_quantile=0.99,
        distill_epochs=0,
        context_lr=0.0,
        device='cpu',
    )
    loose = SelfDistillation(
        deepcopy(seed),
        num_classes=4,
        forget_quantile=0.10,
        distill_epochs=0,
        context_lr=0.0,
        device='cpu',
    )
    strict.identify_zones(loader, loader)
    loose.identify_zones(loader, loader)
    strict_forget = sum(
        int((strict.zone_masks['B'][name] | strict.zone_masks['C'][name]).sum())
        for name in strict.zone_masks['B']
    )
    loose_forget = sum(
        int((loose.zone_masks['B'][name] | loose.zone_masks['C'][name]).sum())
        for name in loose.zone_masks['B']
    )
    assert loose_forget > strict_forget


def test_instance_forget_priority_assigns_every_forget_active_weight_to_b():
    loader = _loader(n=32, num_classes=4, seed=31)
    method = SelfDistillation(
        nn.Linear(8, 4, bias=False),
        num_classes=4,
        retain_quantile=0.9,
        forget_quantile=0.1,
        instance_forget_priority=True,
        distill_epochs=0,
        context_lr=0.0,
        device='cpu',
    )
    method.identify_zones(loader, loader)
    for name in method.phi_f:
        forget_active = method.phi_f[name] > method.resolved_tau_f
        assert torch.equal(method.zone_masks['B'][name], forget_active)
        assert not method.zone_masks['C'][name].any()
        stacked = torch.stack([method.zone_masks[zone][name] for zone in 'ABCD']).long().sum(0)
        assert torch.equal(stacked, torch.ones_like(stacked))


def test_teacher_repair_granular_zone_c_matches_piecewise_equation():
    method = TeacherRepair(
        nn.Linear(3, 1, bias=False),
        num_classes=1,
        zone_c='granular',
        distill_epochs=0,
        device='cpu',
    )
    name, parameter = next(method.backbone.named_parameters())
    parameter.data.fill_(2.0)
    method.zone_masks['C'][name].fill_(True)
    method.phi_r = {name: torch.tensor([[1.0, 4.0, 2.0]])}
    method.phi_f = {name: torch.tensor([[4.0, 1.0, 2.0]])}
    method.resolved_tau_r = method.resolved_tau_f = 1.0
    method._granular_zone_c()
    # omega=1: ties follow the first (forget-dominant) equation branch.
    assert torch.equal(parameter, torch.tensor([[0.0, 2.0, 0.0]]))


def test_dominance_adds_labeled_ce_to_zone_a_repair():
    torch.manual_seed(7)
    seed = nn.Linear(8, 4)
    teacher = deepcopy(seed).eval()
    kl_only = TeacherRepair(
        deepcopy(seed), num_classes=4, distill_epochs=1, distill_lr=0.1, device='cpu'
    )
    kl_and_ce = DominanceZones(
        deepcopy(seed), num_classes=4, distill_epochs=1, distill_lr=0.1, ce_weight=1.0, device='cpu'
    )
    for method in (kl_only, kl_and_ce):
        for name in method.zone_masks['A']:
            method.zone_masks['A'][name].fill_(True)

    loader = _loader(seed=11)
    kl_only._masked_distill(loader, teacher, 'A')
    kl_and_ce._masked_distill(loader, teacher, 'A')

    # An identical teacher produces zero KL gradient.  LwU_3 must still move
    # because the labeled retain cross-entropy supplies an additional signal.
    for initial, repaired in zip(seed.parameters(), kl_only.backbone.parameters()):
        assert torch.allclose(initial, repaired, atol=1e-7)
    assert any(
        not torch.allclose(initial, repaired)
        for initial, repaired in zip(seed.parameters(), kl_and_ce.backbone.parameters())
    )


def test_dominance_requires_positive_ce_weight():
    with pytest.raises(ValueError, match='ce_weight'):
        DominanceZones(nn.Linear(8, 4), num_classes=4, ce_weight=0.0, device='cpu')


def test_dominance_dominance_zones_are_exhaustive_and_send_ties_to_c():
    method = DominanceZones(
        nn.Linear(4, 1, bias=False),
        num_classes=1,
        retain_quantile=0.5,
        forget_quantile=0.5,
        distill_epochs=0,
        device='cpu',
    )
    name = next(iter(method.zone_masks['A']))
    scores = iter(
        [
            {name: torch.tensor([[10.0, 1.0, 4.0, 0.0]])},
            {name: torch.tensor([[1.0, 10.0, 4.0, 0.0]])},
        ]
    )
    method._importance = lambda loader: next(scores)
    method.identify_zones(object(), object())

    assert method.dominance_omega == pytest.approx(1.0)
    assert torch.equal(method.zone_masks['A'][name], torch.tensor([[True, False, False, False]]))
    assert torch.equal(method.zone_masks['B'][name], torch.tensor([[False, True, False, False]]))
    assert torch.equal(method.zone_masks['C'][name], torch.tensor([[False, False, True, False]]))
    assert torch.equal(method.zone_masks['D'][name], torch.tensor([[False, False, False, True]]))
    diagnostics = method.reconstruction_diagnostics()
    assert diagnostics['overlap_or_gap_parameters'] == 0
    assert diagnostics['max_abs_reconstruction_error'] == 0.0


def test_dominance_rejects_non_dominance_zone_rule():
    with pytest.raises(ValueError, match='zone_rule'):
        DominanceZones(nn.Linear(8, 4), num_classes=4, zone_rule='threshold', device='cpu')


def test_zone_d_fpe_expands_neurons_at_fixed_effective_budget():
    seed = FPECNN(num_classes=4, flatten_size=16, hidden_size=8)
    method = TeacherRepair(seed, num_classes=4, distill_epochs=0, device='cpu')
    for name in method.zone_masks['D']:
        method.zone_masks['D'][name].fill_(True)
    expanded = ZoneDFixedParameterExpansion(seed, method.zone_masks)
    assert expanded.fc1.out_features == 2 * seed.fc1.out_features
    assert expanded.effective_nonzero_budget() == sum(
        parameter.numel() for parameter in seed.parameters()
    )
    method.backbone = expanded
    method.zone_masks = expanded.translated_zone_masks
    diagnostics = method.zone_reconstruction_diagnostics()
    assert diagnostics['overlap_or_gap_parameters'] == 0
    assert diagnostics['max_abs_reconstruction_error'] == 0.0


def test_zone_b_fpe_uses_b_mask_and_preserves_effective_budget():
    seed = FPECNN(num_classes=4, flatten_size=16, hidden_size=8)
    method = TeacherRepair(seed, num_classes=4, distill_epochs=0, device='cpu')
    for name in method.zone_masks['B']:
        method.zone_masks['D'][name].fill_(False)
        method.zone_masks['B'][name].fill_(True)
    expanded = ZoneFixedParameterExpansion(seed, method.zone_masks, split_zone='B')
    assert expanded.fc1.out_features == 2 * seed.fc1.out_features
    assert expanded.effective_nonzero_budget() == sum(
        parameter.numel() for parameter in seed.parameters()
    )


def test_context_memory_context_is_feature_only_zone_d_fast_state_and_ephemeral():
    torch.manual_seed(3)
    seed = FPECNN(num_classes=4, flatten_size=16, hidden_size=8)
    method = ContextMemory(
        seed,
        num_classes=4,
        distill_epochs=0,
        context_window=2,
        context_lr=1e-2,
        context_temperature=1.0,
        device='cpu',
    )
    for name in method.zone_masks['D']:
        method.zone_masks['A'][name].fill_(True)
        method.zone_masks['D'][name].fill_(False)
    method.zone_masks['A']['fc1.weight'].fill_(False)
    method.zone_masks['A']['fc2.weight'].fill_(False)
    method.zone_masks['D']['fc1.weight'].fill_(True)
    method.zone_masks['D']['fc2.weight'].fill_(True)
    method.forgotten_classes = {0}
    teacher = deepcopy(method.backbone).eval()
    for parameter in teacher.parameters():
        parameter.requires_grad_(False)
    method.__dict__['_context_teacher'] = teacher
    method._reset_context()
    method.eval()

    inputs = torch.randn(6, 3, 32, 32)
    stored = {
        name: parameter.detach().clone() for name, parameter in method.backbone.named_parameters()
    }
    with method.test_stream():
        method(inputs, test_time_adapt=True)
        assert len(method._context) == 1
        keys, values = method._context[0]
        assert keys.ndim == 2 and values.ndim == 2
        assert values[:, 0].max() < 1e-7
        assert any(
            not torch.equal(parameter, stored[name])
            for name, parameter in method.backbone.named_parameters()
            if name in {'fc1.weight', 'fc2.weight'}
        )
        assert all(
            torch.equal(parameter, stored[name])
            for name, parameter in method.backbone.named_parameters()
            if name not in {'fc1.weight', 'fc2.weight'}
        )

    assert not method._context
    for name, parameter in method.backbone.named_parameters():
        assert torch.equal(parameter, stored[name])


def test_context_memory_he_reinitializes_b_then_repairs_only_a_and_b():
    torch.manual_seed(13)
    seed = nn.Linear(8, 4, bias=False)
    teacher = deepcopy(seed).eval()
    method = ContextMemory(
        deepcopy(seed),
        num_classes=4,
        distill_epochs=1,
        distill_lr=0.1,
        context_lr=0.0,
        device='cpu',
    )
    name = 'weight'
    for zone in 'ABCD':
        method.zone_masks[zone][name].fill_(False)
    method.zone_masks['A'][name][:, :2] = True
    method.zone_masks['B'][name][:, 2:4] = True
    method.zone_masks['C'][name][:, 4:6] = True
    method.zone_masks['D'][name][:, 6:] = True

    original = method.backbone.weight.detach().clone()
    method._zero_zone_b()
    initialized = method.backbone.weight.detach().clone()
    assert torch.equal(initialized[:, :2], original[:, :2])
    assert not torch.equal(initialized[:, 2:4], original[:, 2:4])
    assert torch.equal(initialized[:, 4:], original[:, 4:])

    generator = torch.Generator().manual_seed(4)
    inputs = torch.randn(16, 8, generator=generator)
    targets = torch.randint(1, 4, (16,), generator=generator)
    loader = DataLoader(TensorDataset(inputs, targets), batch_size=8)
    method.forgotten_classes = {0}
    masked_logits = method._repair_teacher_logits(teacher, inputs)
    assert F.softmax(masked_logits, dim=1)[:, 0].max() < 1e-7

    method._masked_distill(loader, teacher, method.REPAIR_ZONES)
    repaired = method.backbone.weight.detach()
    assert not torch.equal(repaired[:, :2], initialized[:, :2])
    assert not torch.equal(repaired[:, 2:4], initialized[:, 2:4])
    assert torch.equal(repaired[:, 4:], initialized[:, 4:])


def test_self_distillation_instance_deletion_does_not_mask_any_class_logits():
    seed = nn.Linear(8, 4, bias=False)
    method = SelfDistillation(
        deepcopy(seed),
        num_classes=4,
        distill_epochs=0,
        context_lr=0.0,
        mask_forgotten_classes=False,
        device='cpu',
    )
    loader = _loader(n=16, num_classes=4, seed=19)
    method._infer_forgotten_classes = lambda _: (_ for _ in ()).throw(
        AssertionError('instance deletion must not infer classes to mask')
    )
    method.identify_zones = lambda *_: None
    method.reconstruction_diagnostics = lambda: {}
    method._zero_zone_b = lambda: setattr(method, 'b_reinitialization_stats', {})
    method._orthogonal_conflict_resolution = lambda *_: None
    method._zone_delta_summary = lambda _: {
        f'{zone}_{suffix}': 0 for zone in 'ABCD' for suffix in ('delta_l2', 'count')
    }
    method._retain_objectives = lambda *_: {'kl': 0.0, 'ce': 0.0}
    method._masked_distill = lambda *_args, **_kwargs: None
    method._update_accumulated_mask = lambda: None
    method.get_zone_statistics = lambda: {'overall': {}}
    method.unlearn(loader, loader)
    assert method.forgotten_classes == set()
    logits = method._repair_teacher_logits(deepcopy(seed), torch.randn(3, 8))
    assert torch.isfinite(logits).all()
    assert method.diagnostics['deletion_scope'] == 'instance'


def test_retain_kl_zone_c_updates_only_c_without_ce():
    torch.manual_seed(23)
    seed = nn.Linear(8, 4, bias=False)
    teacher = deepcopy(seed).eval()
    method = ContextMemory(
        deepcopy(seed),
        num_classes=4,
        zone_c='retain_kl',
        distill_epochs=1,
        distill_lr=0.1,
        ce_weight=1.0,
        context_lr=0.0,
        device='cpu',
    )
    name = 'weight'
    for zone in 'ABCD':
        method.zone_masks[zone][name].fill_(False)
    method.zone_masks['A'][name][:, :2] = True
    method.zone_masks['B'][name][:, 2:4] = True
    method.zone_masks['C'][name][:, 4:6] = True
    method.zone_masks['D'][name][:, 6:] = True
    method.forgotten_classes = {0}

    generator = torch.Generator().manual_seed(24)
    inputs = torch.randn(16, 8, generator=generator)
    # Deliberately invalid labels prove that the pure-KL path never reads CE.
    targets = torch.full((16,), 99)
    loader = DataLoader(TensorDataset(inputs, targets), batch_size=8)
    before = method.backbone.weight.detach().clone()
    method._masked_distill(loader, teacher, 'C', include_ce=False)
    after = method.backbone.weight.detach()
    assert torch.equal(after[:, :4], before[:, :4])
    assert not torch.equal(after[:, 4:6], before[:, 4:6])
    assert torch.equal(after[:, 6:], before[:, 6:])


def test_self_distillation_context_teacher_is_exemplar_free_and_updates_only_free_d():
    torch.manual_seed(29)
    seed = FPECNN(num_classes=4, flatten_size=16, hidden_size=8)
    method = SelfDistillation(
        seed,
        num_classes=4,
        distill_epochs=0,
        context_window=2,
        context_lr=1e-2,
        context_temperature=1.0,
        device='cpu',
    )
    for name in method.zone_masks['D']:
        method.zone_masks['A'][name].fill_(True)
        method.zone_masks['D'][name].fill_(False)
    for name in ('fc1.weight', 'fc2.weight'):
        method.zone_masks['A'][name].fill_(False)
        method.zone_masks['D'][name].fill_(True)
        method.accumulated_mask[name].fill_(False)
    # Protect part of fc1 despite current Zone-D membership.
    method.accumulated_mask['fc1.weight'][:, :4] = True
    method._install_context_teacher()
    method.set_context_classes([1, 2])
    method.eval()

    inputs = torch.randn(6, 3, 32, 32)
    stored = {
        name: parameter.detach().clone() for name, parameter in method.backbone.named_parameters()
    }
    with method.test_stream():
        method(inputs, test_time_adapt=True)
        assert len(method._context) == 1
        keys, values = method._context[0]
        assert keys.ndim == 2 and values.ndim == 2
        assert values[:, [0, 3]].max() < 1e-7
        assert torch.equal(method.backbone.fc1.weight[:, :4], stored['fc1.weight'][:, :4])
        assert not torch.equal(method.backbone.fc1.weight[:, 4:], stored['fc1.weight'][:, 4:])
        assert all(
            torch.equal(parameter, stored[name])
            for name, parameter in method.backbone.named_parameters()
            if name not in {'fc1.weight', 'fc2.weight'}
        )

    assert not method._context
    for name, parameter in method.backbone.named_parameters():
        assert torch.equal(parameter, stored[name])


# --------------------------------------------------------------------------
# Eq. 19-23: Zone C
# --------------------------------------------------------------------------


def test_projected_update_is_orthogonal_to_retain_gradient():
    """
    The claim in Section 2.2 is first-order non-interference: the update's
    component along the retain gradient is zero after projection.
    """
    torch.manual_seed(0)
    g_r = torch.randn(64)
    g_f = torch.randn(64)

    dot = (g_f * g_r).sum()
    norm_sq = (g_r * g_r).sum() + 1e-8
    g_perp = g_f - (dot / norm_sq) * g_r

    assert torch.allclose((g_perp * g_r).sum(), torch.tensor(0.0), atol=1e-4)
    # Projection never amplifies the update.
    assert g_perp.norm() <= g_f.norm() + 1e-6


def test_zone_c_displacement_respects_budget():
    """Eq. 23 clips the complete Zone-C vector to its SSV-scaled budget."""
    model = _toy_model()
    retain, forget = _loader(), _loader(seed=1)
    model.identify_zones(retain, forget)

    before = {n: p.data.clone() for n, p in model.backbone.named_parameters()}

    delta_c = 0.05
    model._orthogonal_conflict_resolution(
        retain, forget, nn.CrossEntropyLoss(), eta_c=1.0, n_c=5, delta_c=delta_c
    )

    drift_sq = 0.0
    for name, p in model.backbone.named_parameters():
        mask = model.zone_masks['C'][name]
        drift_sq += ((p.data - before[name]) * mask).square().sum().item()
    assert drift_sq**0.5 <= model.last_zone_c_budget + 1e-6


def test_frozen_zone_a_receives_no_gradient():
    """Zone A is frozen under the cumulative mask during training."""
    model = _toy_model()
    model.identify_zones(_loader(), _loader(seed=1))
    model._update_accumulated_mask()

    xb, yb = next(iter(_loader()))
    model.backbone.zero_grad()
    loss = nn.CrossEntropyLoss()(model.backbone(xb), yb)
    model.masked_backward(loss)

    for name, p in model.backbone.named_parameters():
        if p.grad is not None:
            protected = model.accumulated_mask[name]
            if protected.any():
                assert p.grad.data[protected].abs().max() == 0.0


# --------------------------------------------------------------------------
# Eq. 13-17: test-time update
# --------------------------------------------------------------------------


def test_ttu_does_not_modify_stored_weights():
    """
    Section 2.2 states TTU acts on an ephemeral inference-time copy and the
    stored long-term parameters are restored afterwards.
    """
    model = _toy_model()
    model.identify_zones(_loader(), _loader(seed=1))

    stored = {n: p.data.clone() for n, p in model.backbone.named_parameters()}

    model.original_params = {n: p.data.clone() for n, p in model.backbone.named_parameters()}
    xb, _ = next(iter(_loader()))
    model.test_time_update(xb)
    model.restore_original_params()

    for name, p in model.backbone.named_parameters():
        assert torch.allclose(
            p.data, stored[name], atol=1e-6
        ), f"{name} was changed by test-time adaptation"


def test_ttu_only_moves_zone_a():
    """Eq. 14 masks the entropy gradient to Zone A."""
    model = _toy_model()
    model.identify_zones(_loader(), _loader(seed=1))

    model.original_params = {n: p.data.clone() for n, p in model.backbone.named_parameters()}
    before = {n: p.data.clone() for n, p in model.backbone.named_parameters()}

    xb, _ = next(iter(_loader()))
    model.test_time_update(xb)

    for name, p in model.backbone.named_parameters():
        outside_A = ~model.zone_masks['A'][name]
        if outside_A.any():
            delta = (p.data - before[name])[outside_A].abs().max().item()
            assert delta < 1e-6, f"{name} moved outside Zone A by {delta}"


def test_evaluator_runs_ephemeral_ttu_and_restores_weights():
    """The continual evaluator must not bypass the paper's TTU path."""
    model = _toy_model()
    model.eval()
    model.surprise_threshold = 0.0
    for name in model.zone_masks['A']:
        model.zone_masks['A'][name].fill_(True)
    stored = {name: p.detach().clone() for name, p in model.backbone.named_parameters()}

    assert evaluate_task(model, _loader(), 'cpu', test_time_adapt=True) >= 0.0
    assert not model.original_params
    for name, parameter in model.backbone.named_parameters():
        assert torch.equal(parameter, stored[name])


def test_checkpoint_persists_masks_and_task_counter():
    model = _toy_model()
    model.consolidate_task(_loader())
    state = model.state_dict()
    restored = _toy_model(seed=1)
    restored.load_state_dict(state)
    assert restored.current_task == model.current_task
    for name in model.accumulated_mask:
        assert torch.equal(restored.accumulated_mask[name], model.accumulated_mask[name])


def test_zone_statistics_include_current_partition_and_cumulative_capacity():
    model = _toy_model(tau=0.0)
    model.consolidate_task(_loader())
    stats = model.get_zone_statistics()
    current = stats['overall']
    accumulated = stats['accumulated']
    assert sum(current[zone] for zone in ('A', 'B', 'C', 'D')) == pytest.approx(100.0)
    assert accumulated['protected'] + accumulated['plastic'] == pytest.approx(100.0)
    # CL consolidation has no forget set, hence M_f is empty.
    assert current['B'] == pytest.approx(0.0)
    assert current['C'] == pytest.approx(0.0)
    assert accumulated['protected'] == pytest.approx(current['A'])
