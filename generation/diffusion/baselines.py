"""Generative unlearning baselines on the conditional DDPM.

Fine-tune and NegGrad are the generic ones; SalUn and the density-matched random
mask share one objective and one update budget and differ only in which half of
the weights may move, which is what isolates the saliency criterion.
"""

import copy

import torch
import torch.nn.functional as F

from diffusion.ddpm import CondUNet, Diffusion


def load_source(ck, dev='cuda', use_ema=True):
    d = torch.load(ck, map_location=dev, weights_only=False)
    m = CondUNet(base=d['base']).to(dev)
    m.load_state_dict(d['ema'] if use_ema and 'ema' in d else d['model'])
    return m


def batches(x, y, bs, shuffle=True):
    idx = (
        torch.randperm(len(x), device=x.device)
        if shuffle
        else torch.arange(len(x), device=x.device)
    )
    for i in range(0, len(idx), bs):
        j = idx[i : i + bs]
        yield x[j], y[j]


def cycle(x, y, bs=128):
    while True:
        yield from batches(x, y, bs)


def finetune(model, xr, yr, diff, *, lr=1e-4, epochs=4, bs=128, dev='cuda', log=print):
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.0)
    for ep in range(epochs):
        model.train()
        tot = nb = 0
        for xb, yb in batches(xr, yr, bs):
            t = torch.randint(0, diff.T, (len(xb),), device=dev)
            n = torch.randn_like(xb)
            loss = F.mse_loss(model(diff.q_sample(xb, t, n), t, yb), n)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tot += float(loss)
            nb += 1
        log(f'  finetune ep{ep}: {tot/max(nb,1):.4f}')
    return model


def neggrad(model, xr, yr, xf, yf, diff, *, lr=5e-5, epochs=4, bs=128, dev='cuda', log=print):
    """Ascent on the forget denoising loss, descent on retain."""
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.0)
    for ep in range(epochs):
        model.train()
        tot = nb = 0
        fb = list(batches(xf, yf, bs))
        for i, (xb, yb) in enumerate(batches(xr, yr, bs)):
            t = torch.randint(0, diff.T, (len(xb),), device=dev)
            n = torch.randn_like(xb)
            loss = F.mse_loss(model(diff.q_sample(xb, t, n), t, yb), n)
            if fb:
                xf_, yf_ = fb[i % len(fb)]
                tf = torch.randint(0, diff.T, (len(xf_),), device=dev)
                nf = torch.randn_like(xf_)
                loss = loss - F.mse_loss(model(diff.q_sample(xf_, tf, nf), tf, yf_), nf)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tot += float(loss)
            nb += 1
        log(f'  neggrad ep{ep}: {tot/max(nb,1):.4f}')
    return model


def saliency_masks(source, xf, yf, dev='cuda', mask_seed=4242, fraction=0.5):
    """SalUn's forget-gradient mask and a density-matched random control.

    The random mask is the same mask permuted, so both keep exactly the same
    number of weights; any difference between the two rows is the criterion,
    not the update budget.
    """
    model = copy.deepcopy(source).eval()
    acc = {n: torch.zeros_like(p) for n, p in model.named_parameters()}
    diff = Diffusion(device=dev)
    torch.manual_seed(42)
    for x, y in batches(xf, yf, 128):
        t = torch.randint(0, diff.T, (len(x),), device=dev)
        e = torch.randn_like(x)
        model.zero_grad(set_to_none=True)
        F.mse_loss(model(diff.q_sample(x, t, e), t, y), e).backward()
        for n, p in model.named_parameters():
            if p.grad is not None:
                acc[n] += p.grad
    flat = torch.cat([v.abs().flatten() for v in acc.values()])
    mask = torch.zeros_like(flat)
    mask[torch.argsort(flat, descending=True)[: int(len(flat) * fraction)]] = 1
    g = torch.Generator(device=dev).manual_seed(mask_seed)
    random = mask[torch.randperm(len(mask), generator=g, device=dev)]
    out = {kind: {} for kind in ('salun', 'random')}
    start = 0
    for n, p in model.named_parameters():
        for kind, m in (('salun', mask), ('random', random)):
            out[kind][n] = m[start : start + p.numel()].reshape_as(p).clone()
        start += p.numel()
    assert sum(int(v.sum()) for v in out['salun'].values()) == sum(
        int(v.sum()) for v in out['random'].values()
    )
    model.zero_grad(set_to_none=True)
    return out


def train_masked(
    source,
    mask,
    xr,
    yr,
    xf,
    yf,
    *,
    steps=1000,
    lr=1e-4,
    bs=128,
    retain_weight=1e-3,
    num_classes=10,
    dev='cuda',
    log=print,
):
    """SalUn / random-mask training: relabel the forget condition to (c+1) mod C
    and keep denoising the retained classes, with updates restricted to `mask`."""
    torch.manual_seed(42)
    model = copy.deepcopy(source).train()
    diff = Diffusion(device=dev)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    rb, fb = cycle(xr, yr, bs), cycle(xf, yf, bs)
    history = []
    for step in range(steps):
        x, y = next(rb)
        t = torch.randint(0, diff.T, (len(x),), device=dev)
        e = torch.randn_like(x)
        # Source training used CFG dropout; preserve it on the retained term.
        y = y.clone()
        y[torch.rand(len(y), device=dev) < 0.1] = model.num_classes
        retain = (model(diff.q_sample(x, t, e), t, y) - e).square().flatten(1).sum(1).mean()
        x, y = next(fb)
        t = torch.randint(0, diff.T, (len(x),), device=dev)
        e = torch.randn_like(x)
        xt = diff.q_sample(x, t, e)
        with torch.no_grad():
            target = model(xt, t, (y + 1) % num_classes)
        forget = F.mse_loss(model(xt, t, y), target)
        loss = forget + retain_weight * retain
        assert torch.isfinite(loss)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        for n, p in model.named_parameters():
            if p.grad is not None:
                p.grad.mul_(mask[n])
        opt.step()
        if step % 100 == 0 or step + 1 == steps:
            history.append(
                dict(
                    step=step,
                    retain_sum_mse=float(retain.detach()),
                    forget_mse=float(forget.detach()),
                )
            )
            log(history[-1])
    return model.eval(), history
