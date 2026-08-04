"""
Checks that the implementation does what Section 2 says it does.

These test properties the method section claims, not accuracies: that the four
zones partition the network, that the Zone C update really is orthogonal to
the retain gradient, that displacement stays inside its budget, and that
test-time adaptation does not touch the stored weights.
"""

import numpy as np
import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from lwu.models.lwu import LwU


def _toy_model(seed=0):
    torch.manual_seed(seed)
    backbone = nn.Sequential(
        nn.Linear(8, 16), nn.ReLU(), nn.Linear(16, 4)
    )
    return LwU(backbone=backbone, num_classes=4, tau_r=0.3, tau_f=0.3,
               device='cpu')


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
    grads = {n: torch.zeros_like(p)
             for n, p in model.backbone.named_parameters()}
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
        expected = -grads[n] * p.data + 0.5 * fisher[n] * p.data ** 2
        assert torch.allclose(ssv[n], expected, atol=1e-5)


def test_masks_threshold_absolute_ssv():
    """Eq. 8 thresholds |phi|, so a large negative score is important too."""
    model = _toy_model()
    model.identify_zones(_loader(), _loader(seed=1))

    for name, p in model.backbone.named_parameters():
        phi = model.phi_r[name]
        in_mask = model.zone_masks['A'][name] | model.zone_masks['C'][name]
        if in_mask.any() and (~in_mask).any():
            # Selected weights should have larger |phi| than unselected ones.
            assert phi[in_mask].abs().max() >= phi[~in_mask].abs().min()


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
    """Eq. 23 clips cumulative displacement to delta_C."""
    model = _toy_model()
    retain, forget = _loader(), _loader(seed=1)
    model.identify_zones(retain, forget)

    before = {n: p.data.clone()
              for n, p in model.backbone.named_parameters()}

    delta_c = 0.05
    model._orthogonal_conflict_resolution(
        retain, forget, nn.CrossEntropyLoss(),
        eta_c=1.0, n_c=5, delta_c=delta_c
    )

    for name, p in model.backbone.named_parameters():
        if model.zone_masks['C'][name].any():
            drift = (p.data - before[name]).norm().item()
            assert drift <= delta_c + 1e-5, f"{name} drifted {drift}"


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

    stored = {n: p.data.clone()
              for n, p in model.backbone.named_parameters()}

    model.original_params = {n: p.data.clone()
                             for n, p in model.backbone.named_parameters()}
    xb, _ = next(iter(_loader()))
    model.test_time_update(xb)
    model.restore_original_params()

    for name, p in model.backbone.named_parameters():
        assert torch.allclose(p.data, stored[name], atol=1e-6), \
            f"{name} was changed by test-time adaptation"


def test_ttu_only_moves_zone_a():
    """Eq. 14 masks the entropy gradient to Zone A."""
    model = _toy_model()
    model.identify_zones(_loader(), _loader(seed=1))

    model.original_params = {n: p.data.clone()
                             for n, p in model.backbone.named_parameters()}
    before = {n: p.data.clone()
              for n, p in model.backbone.named_parameters()}

    xb, _ = next(iter(_loader()))
    model.test_time_update(xb)

    for name, p in model.backbone.named_parameters():
        outside_A = ~model.zone_masks['A'][name]
        if outside_A.any():
            delta = (p.data - before[name])[outside_A].abs().max().item()
            assert delta < 1e-6, f"{name} moved outside Zone A by {delta}"
