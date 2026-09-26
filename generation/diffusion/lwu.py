"""LwU on the conditional DDPM: zone decomposition, then a teacher-redirection
update confined to the plastic zone.

The classifier method's four zones carry over unchanged in spirit, with the
denoising gradient replacing the classification gradient as the importance
signal: mean squared minibatch denoising gradients over 128 forget and 128
retain examples, a global quantile threshold per side, and the same scale-free
dominance rule (A retain-dominant, B forget-dominant, C conflict, D plastic).

Only Zone D is trainable here: the forgotten condition is redirected to the
frozen source's prediction under class (c+1) mod 10 while the retained
conditions are anchored to the frozen source's own predictions, and every
coordinate stays inside an L-infinity trust region around the source. There is
no Zone-B reset and no test-time memory, so the generative transfer is the
zone machinery alone.
"""

import copy
import json
import math
import time

import torch
import torch.nn.functional as F

from diffusion.baselines import cycle
from diffusion.ddpm import Diffusion

CONFIG = dict(
    seed=42,
    steps=1000,
    lr=1e-4,
    batch=128,
    importance_examples=128,
    importance_microbatch=32,
    quantile=0.99,
    dominance_margin=1.25,
    bound=0.1,
    retain_weight=1.0,
    reset=False,
    memory=False,
    target_rule='(forgotten_class + 1) % 10',
)


def zones(source, x, y, forget_class, work, dev='cuda', log=print):
    """Four-zone partition of the UNet parameters; cached in `work/zones.pt`."""
    path = work / 'zones.pt'
    if path.exists():
        return torch.load(path, map_location=dev, weights_only=True)
    model = copy.deepcopy(source).eval().requires_grad_(True)
    diff = Diffusion(device=dev)
    scores = {}
    torch.manual_seed(CONFIG['seed'])
    for group, selected in [('forget', y == forget_class), ('retain', y != forget_class)]:
        indices = torch.where(selected)[0][: CONFIG['importance_examples']]
        score = {n: torch.zeros_like(p) for n, p in model.named_parameters()}
        micro = CONFIG['importance_microbatch']
        for ix in indices.split(micro):
            t = torch.randint(0, diff.T, (len(ix),), device=dev)
            noise = torch.randn_like(x[ix])
            model.zero_grad(set_to_none=True)
            F.mse_loss(model(diff.q_sample(x[ix], t, noise), t, y[ix]), noise).backward()
            for n, p in model.named_parameters():
                if p.grad is not None:
                    score[n].add_(p.grad.square(), alpha=micro / CONFIG['importance_examples'])
        scores[group] = score
    model.zero_grad(set_to_none=True)

    thresholds = {}
    for group, score in scores.items():
        flat = torch.cat([v.flatten() for v in score.values()])
        assert torch.isfinite(flat).all()
        k = math.ceil(CONFIG['quantile'] * flat.numel())
        thresholds[group] = max(float(torch.kthvalue(flat, k).values), 1e-30)
        del flat
    masks = {}
    counts = {z: 0 for z in 'ABCD'}
    margin = CONFIG['dominance_margin']
    for n, p in model.named_parameters():
        f = scores['forget'][n] / thresholds['forget']
        r = scores['retain'][n] / thresholds['retain']
        a = (r > 1) & (r > margin * f)
        b = (f > 1) & (f > margin * r)
        salient = (r > 1) | (f > 1)
        c = salient & ~a & ~b
        d = ~salient
        assert torch.all(a.to(torch.int8) + b + c + d == 1)
        mask = torch.full_like(p, 3, dtype=torch.uint8)
        mask[a] = 0
        mask[b] = 1
        mask[c] = 2
        masks[n] = mask
        for z, v in zip('ABCD', (a, b, c, d)):
            counts[z] += int(v.sum())
    assert counts['D'] > 0, 'empty plastic zone: nothing would be trainable'
    (work / 'zone_audit.json').write_text(
        json.dumps(
            dict(
                counts=counts,
                thresholds=thresholds,
                estimator='mean squared minibatch denoising gradient',
            ),
            indent=2,
        )
    )
    log(dict(stage='zones', forgotten_class=forget_class, counts=counts))
    torch.save({n: v.cpu() for n, v in masks.items()}, path)
    del scores
    return masks


def unlearn(source, x, y, forget_class, work, dev='cuda', log=print):
    """One class deletion. Returns the unlearned model and writes `lwu.pt`."""
    masks = zones(source, x, y, forget_class, work, dev=dev, log=log)
    active = {n: v == 3 for n, v in masks.items()}  # Zone D only
    del masks
    teacher = copy.deepcopy(source).eval().requires_grad_(False)
    model = copy.deepcopy(source).train().requires_grad_(True)
    torch.manual_seed(CONFIG['seed'])
    diff = Diffusion(device=dev)
    rb = cycle(x[y != forget_class], y[y != forget_class], CONFIG['batch'])
    fb = cycle(x[y == forget_class], y[y == forget_class], CONFIG['batch'])
    opt = torch.optim.Adam(model.parameters(), lr=CONFIG['lr'])
    history = []
    started = time.time()
    for step in range(CONFIG['steps']):
        xr, yr = next(rb)
        xf, yf = next(fb)
        tr = torch.randint(0, diff.T, (len(xr),), device=dev)
        tf = torch.randint(0, diff.T, (len(xf),), device=dev)
        xr = diff.q_sample(xr, tr, torch.randn_like(xr))
        xf = diff.q_sample(xf, tf, torch.randn_like(xf))
        with torch.no_grad():
            good = teacher(xf, tf, (yf + 1) % 10)
            retain = teacher(xr, tr, yr)
        lf = F.mse_loss(model(xf, tf, yf), good)
        lr = F.mse_loss(model(xr, tr, yr), retain)
        loss = lf + CONFIG['retain_weight'] * lr
        assert torch.isfinite(loss)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        for n, p in model.named_parameters():
            if p.grad is not None:
                p.grad.mul_(active[n])
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
        opt.step()
        # Frozen coordinates restored exactly; active ones held in the trust region.
        with torch.no_grad():
            for n, p in model.named_parameters():
                anchor = teacher.get_parameter(n)
                p.copy_(
                    anchor
                    + torch.where(
                        active[n],
                        (p - anchor).clamp(-CONFIG['bound'], CONFIG['bound']),
                        torch.zeros_like(p),
                    )
                )
        if step % 100 == 0 or step + 1 == CONFIG['steps']:
            row = dict(
                step=step + 1,
                forget_loss=float(lf.detach()),
                retain_loss=float(lr.detach()),
                seconds=time.time() - started,
            )
            history.append(row)
            (work / 'training.json').write_text(json.dumps(history, indent=2))
            log(dict(stage='training', forgotten_class=forget_class, **row))

    frozen_max = 0.0
    displacement = 0.0
    with torch.no_grad():
        for n, p in model.named_parameters():
            assert torch.isfinite(p).all()
            delta = p - teacher.get_parameter(n)
            frozen_max = max(frozen_max, float(delta.masked_fill(active[n], 0).abs().max()))
            assert float(delta.abs().max()) <= CONFIG['bound'] + 1e-6
            displacement += float(delta.square().sum())
    assert frozen_max == 0 and all(p.grad is None for p in teacher.parameters())
    torch.save(
        dict(
            base=model.stem.out_channels,
            model=model.state_dict(),
            config=CONFIG,
            forgotten_class=forget_class,
            history=history,
            displacement_l2=displacement**0.5,
            frozen_max=frozen_max,
        ),
        work / 'lwu.pt',
    )
    return model.eval()
