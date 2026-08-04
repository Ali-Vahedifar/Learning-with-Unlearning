"""
Validation of the closed-form SSV against reference Shapley values.

WHAT THIS IS FOR
----------------
Section 2.1.1 derives a closed form for the Shapley value of a synapse
(Eq. 6), consisting of a first-order term, a self-curvature term, and a
cooperative interaction sum over off-diagonal curvature. At ResNet and LoRA
scale the interaction sum is dropped: the full Hessian is O(|N|^2), which for
ResNet-18 is ~1.2e14 entries, so the deployed estimator uses the diagonal
Fisher approximation and the interaction sum vanishes.

This harness tests the closed form where it *can* be computed exactly. On a
network small enough for the full Hessian to fit in memory, it compares:

    (a) the diagonal closed form      phi_i = -g_i t_i + 1/2 t_i^2 H_ii
    (b) the full closed form          (a) + 1/2 t_i sum_{j != i} H_ij t_j

against Monte-Carlo Shapley values estimated directly from the definition
(Eq. 5), by sampling permutations and accumulating marginal contributions.

The comparison is by Spearman rank correlation, because the zone masks depend
only on the ranking of |phi|, not on its scale.

HOW THE REFERENCE IS COMPUTED
-----------------------------
The Shapley value of parameter i is its average marginal contribution over all
coalitions. Sampling uniformly random permutations and, for each, walking the
parameters in order while recording U(S u {i}) - U(S), gives an unbiased
estimator of that average. "Removing" a parameter means setting it to zero,
which is the same removal operation the derivation in Appendix uses.

Utility is U(S) = -loss(theta_S), so a parameter whose presence lowers the
loss receives a positive value.

USAGE
-----
    python validation/ssv_exact_validation.py                  # default run
    python validation/ssv_exact_validation.py --permutations 200
    python validation/ssv_exact_validation.py --seeds 0 1 2

Runs on CPU. The default configuration takes a few minutes.
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn


# --------------------------------------------------------------------------
# Model and data
# --------------------------------------------------------------------------

class TinyNet(nn.Module):
    """
    Small MLP whose parameter count is low enough for the full Hessian to be
    formed explicitly. With the default widths this is 1,024 parameters, so
    the Hessian has 1,048,576 entries (4.2 MB in fp32).
    """

    def __init__(self, in_dim=16, hidden=32, out_dim=16, bias=False):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden, bias=bias)
        self.fc2 = nn.Linear(hidden, out_dim, bias=bias)

    def forward(self, x):
        return self.fc2(torch.tanh(self.fc1(x)))


def make_data(n=256, in_dim=16, out_dim=16, seed=0):
    """A fixed synthetic classification task with learnable structure."""
    g = torch.Generator().manual_seed(seed)
    w = torch.randn(in_dim, out_dim, generator=g)
    x = torch.randn(n, in_dim, generator=g)
    y = (x @ w).argmax(dim=1)
    return x, y


def flat_params(model):
    return torch.cat([p.detach().reshape(-1) for p in model.parameters()])


def set_flat_params(model, flat):
    i = 0
    for p in model.parameters():
        n = p.numel()
        p.data.copy_(flat[i:i + n].view_as(p))
        i += n


# --------------------------------------------------------------------------
# Utility function
# --------------------------------------------------------------------------

def make_utility(model, x, y):
    """
    U(S) = -loss evaluated with parameters outside S set to zero.

    Returns a function taking a boolean coalition mask over the flat
    parameter vector.
    """
    theta = flat_params(model).clone()
    criterion = nn.CrossEntropyLoss()

    def utility(mask):
        masked = theta * torch.as_tensor(mask, dtype=theta.dtype)
        set_flat_params(model, masked)
        with torch.no_grad():
            loss = criterion(model(x), y)
        return -loss.item()

    return utility, theta


# --------------------------------------------------------------------------
# Reference: Monte-Carlo Shapley values
# --------------------------------------------------------------------------

def monte_carlo_shapley(utility, n_params, n_permutations, seed=0, verbose=True):
    """
    Unbiased estimate of the Shapley value of every parameter.

    For each sampled permutation, parameters are added one at a time and the
    increase in utility is credited to the parameter that was added. Averaging
    over permutations converges to Eq. 5.

    Cost is n_permutations * n_params utility evaluations.
    """
    rng = np.random.default_rng(seed)
    phi = np.zeros(n_params, dtype=np.float64)

    empty = np.zeros(n_params, dtype=bool)
    u_empty = utility(empty)

    t0 = time.time()
    for p_idx in range(n_permutations):
        order = rng.permutation(n_params)
        mask = np.zeros(n_params, dtype=bool)
        u_prev = u_empty

        for i in order:
            mask[i] = True
            u_curr = utility(mask)
            phi[i] += (u_curr - u_prev)
            u_prev = u_curr

        if verbose and (p_idx + 1) % 5 == 0:
            elapsed = time.time() - t0
            rate = (p_idx + 1) / elapsed
            remaining = (n_permutations - p_idx - 1) / rate
            print(f"    permutation {p_idx + 1}/{n_permutations} "
                  f"({elapsed:.0f}s elapsed, ~{remaining:.0f}s left)")

    return phi / n_permutations


# --------------------------------------------------------------------------
# Closed forms
# --------------------------------------------------------------------------

def gradient_and_hessian(model, x, y):
    """Full gradient and full Hessian of the loss at the current parameters."""
    criterion = nn.CrossEntropyLoss()
    params = list(model.parameters())

    loss = criterion(model(x), y)
    grads = torch.autograd.grad(loss, params, create_graph=True)
    flat_grad = torch.cat([g.reshape(-1) for g in grads])

    n = flat_grad.numel()
    hessian = torch.zeros(n, n)
    for i in range(n):
        row = torch.autograd.grad(flat_grad[i], params, retain_graph=True)
        hessian[i] = torch.cat([r.reshape(-1) for r in row]).detach()

    # Symmetrise to remove numerical asymmetry from the row-wise construction.
    hessian = 0.5 * (hessian + hessian.T)
    return flat_grad.detach(), hessian


def ssv_diagonal(grad, hessian, theta):
    """phi_i = -g_i t_i + 1/2 t_i^2 H_ii   (the deployed estimator's form)"""
    return -grad * theta + 0.5 * theta.pow(2) * torch.diagonal(hessian)


def ssv_full(grad, hessian, theta):
    """
    Diagonal form plus the cooperative interaction sum:

        + 1/2 t_i sum_{j != i} H_ij t_j

    with path weights w_ij = 1 under the near-convergence assumption.
    """
    diag = ssv_diagonal(grad, hessian, theta)

    h_off = hessian.clone()
    h_off.fill_diagonal_(0.0)
    interaction = 0.5 * theta * (h_off @ theta)

    return diag + interaction


# --------------------------------------------------------------------------
# Correlation
# --------------------------------------------------------------------------

def spearman(a, b):
    """Spearman rank correlation without a scipy dependency."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)

    def rank(v):
        order = v.argsort()
        r = np.empty_like(order, dtype=np.float64)
        r[order] = np.arange(len(v), dtype=np.float64)
        # average ties
        _, inv, counts = np.unique(v, return_inverse=True, return_counts=True)
        if (counts > 1).any():
            sums = np.zeros(len(counts))
            np.add.at(sums, inv, r)
            r = (sums / counts)[inv]
        return r

    ra, rb = rank(a), rank(b)
    ra -= ra.mean()
    rb -= rb.mean()
    denom = np.sqrt((ra ** 2).sum() * (rb ** 2).sum())
    return float((ra * rb).sum() / denom) if denom > 0 else 0.0


# --------------------------------------------------------------------------
# Experiment
# --------------------------------------------------------------------------

def run_seed(seed, n_permutations, pretrain_steps, verbose=True):
    torch.manual_seed(seed)
    np.random.seed(seed)

    model = TinyNet()
    x, y = make_data(seed=seed)

    n_params = sum(p.numel() for p in model.parameters())

    # Train briefly so the evaluation point is a trained network rather than a
    # random one; the closed form assumes proximity to a converged solution.
    opt = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()
    for _ in range(pretrain_steps):
        opt.zero_grad()
        loss = criterion(model(x), y)
        loss.backward()
        opt.step()

    if verbose:
        print(f"  seed {seed}: {n_params} parameters, "
              f"Hessian {n_params}x{n_params} = {n_params ** 2:,} entries "
              f"({n_params ** 2 * 4 / 1e6:.1f} MB fp32), "
              f"final loss {loss.item():.4f}")

    grad, hessian = gradient_and_hessian(model, x, y)
    theta = flat_params(model).clone()

    phi_diag = ssv_diagonal(grad, hessian, theta).numpy()
    phi_full = ssv_full(grad, hessian, theta).numpy()

    utility, _ = make_utility(model, x, y)
    if verbose:
        print(f"  seed {seed}: Monte-Carlo Shapley, {n_permutations} "
              f"permutations x {n_params} parameters = "
              f"{n_permutations * n_params:,} utility evaluations")
    phi_mc = monte_carlo_shapley(utility, n_params, n_permutations,
                                 seed=seed, verbose=verbose)

    # Restore parameters after the masking done during utility evaluation.
    set_flat_params(model, theta)

    return {
        'seed': seed,
        'n_params': int(n_params),
        'hessian_entries': int(n_params ** 2),
        'rho_diagonal': spearman(np.abs(phi_diag), np.abs(phi_mc)),
        'rho_full': spearman(np.abs(phi_full), np.abs(phi_mc)),
        'rho_diagonal_signed': spearman(phi_diag, phi_mc),
        'rho_full_signed': spearman(phi_full, phi_mc),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--permutations', type=int, default=40,
                    help='Monte-Carlo permutations per seed')
    ap.add_argument('--seeds', type=int, nargs='+', default=[0, 1, 2])
    ap.add_argument('--pretrain-steps', type=int, default=300)
    ap.add_argument('--out', type=str, default='validation/table9_results.json')
    args = ap.parse_args()

    print("=" * 70)
    print("SSV closed form vs Monte-Carlo Shapley reference")
    print("=" * 70)

    results = []
    for seed in args.seeds:
        results.append(run_seed(seed, args.permutations, args.pretrain_steps))
        r = results[-1]
        print(f"  seed {seed}: rho(diagonal) = {r['rho_diagonal']:.3f}, "
              f"rho(full) = {r['rho_full']:.3f}")
        print()

    diag = [r['rho_diagonal'] for r in results]
    full = [r['rho_full'] for r in results]

    print("=" * 70)
    print(f"{'estimator':<40} {'Spearman rho':>16}")
    print("-" * 70)
    print(f"{'diagonal only (deployed form)':<40} "
          f"{np.mean(diag):>10.3f} +/- {np.std(diag):.3f}")
    print(f"{'with cooperative interactions':<40} "
          f"{np.mean(full):>10.3f} +/- {np.std(full):.3f}")
    print("-" * 70)
    print(f"{'difference':<40} {np.mean(full) - np.mean(diag):>10.3f}")
    print("=" * 70)

    summary = {
        'config': vars(args),
        'per_seed': results,
        'rho_diagonal_mean': float(np.mean(diag)),
        'rho_diagonal_std': float(np.std(diag)),
        'rho_full_mean': float(np.mean(full)),
        'rho_full_std': float(np.std(full)),
    }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2))
    print(f"\nWritten to {out}")


if __name__ == '__main__':
    main()
