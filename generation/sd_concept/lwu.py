"""LwU on Stable Diffusion v1.4: zone decomposition, then a teacher-redirection
update confined to the plastic zone.

The zones are the classifier method's, with the denoising gradient as the
importance signal: mean squared minibatch gradients over 128 forgotten and 128
retained latents, a global 99th-percentile threshold per side, and the same
scale-free dominance rule (A retain-dominant, B forget-dominant, C conflict,
D plastic).

Only Zone D trains: under the forgotten prompt the student is pulled onto the
frozen source's prediction for the *clothed* prompt, while the retained
condition is anchored to the frozen source's own prediction, every coordinate
held inside an L-infinity trust region. No Zone-B reset, no test-time memory --
the transfer is the zone machinery alone.
"""

import gc
import math
import time

import torch
import torch.nn.functional as F

from sd_concept import engine as b

CONFIG = dict(
    seed=42,
    steps=340,
    lr=1e-5,
    batch=8,
    microbatch=2,
    importance_examples=128,
    quantile=0.99,
    dominance_margin=1.25,
    retain_weight=1.0,
    max_coordinate_displacement=0.0034,
    reset=False,
    memory=False,
    objective='forget clothed-teacher MSE + retain source-teacher MSE',
    forget_prompt=b.FORGET,
    target_prompt=b.RETAIN,
    source=b.SOURCE,
    revision=b.REVISION,
)


def zones(e, data, root=None):
    """Four-zone partition of the UNet; cached in $LWU_WORK/generation/sd/zones.pt."""
    path = b.WORK / 'zones.pt'
    if path.exists():
        return torch.load(path, map_location='cuda', weights_only=True)
    b.seed(CONFIG['seed'])
    model = e.unet
    model.train()
    model.enable_gradient_checkpointing()
    scores = {}
    started = time.perf_counter()
    for group, prompt in (('forget', b.FORGET), ('retain', b.RETAIN)):
        importance = {n: torch.zeros_like(p) for n, p in model.named_parameters()}
        emb = e.embed([prompt] * 2)
        for start in range(0, CONFIG['importance_examples'], 2):
            model.zero_grad(set_to_none=True)
            x, t, noise = b.batch(e, data, group, slice(start, start + 2))
            F.mse_loss(b.predict(model, x, t, emb), noise).backward()
            with torch.no_grad():
                for n, p in model.named_parameters():
                    if p.grad is not None:
                        importance[n].add_(
                            p.grad.square(), alpha=1 / (CONFIG['importance_examples'] // 2)
                        )
            if root and start % 16 == 0:
                b.status(
                    root,
                    'importance',
                    group=group,
                    examples=start + 2,
                    total=CONFIG['importance_examples'],
                )
        scores[group] = importance
    model.zero_grad(set_to_none=True)

    thresholds = {}
    for group, values in scores.items():
        flat = torch.cat([v.flatten() for v in values.values()])
        assert torch.isfinite(flat).all()
        k = math.ceil(CONFIG['quantile'] * flat.numel())
        thresholds[group] = max(float(torch.kthvalue(flat, k).values), 1e-30)
        del flat
    masks = {}
    counts = [0] * 4
    for n, p in model.named_parameters():
        r = scores['retain'][n] / thresholds['retain']
        f = scores['forget'][n] / thresholds['forget']
        a = (r > 1) & (r > CONFIG['dominance_margin'] * f)
        bb = (f > 1) & (f > CONFIG['dominance_margin'] * r)
        salient = (r > 1) | (f > 1)
        c = salient & ~a & ~bb
        d = ~salient
        assert torch.all(a.to(torch.int8) + bb + c + d == 1)
        mask = torch.full_like(p, 3, dtype=torch.uint8)
        mask[a] = 0
        mask[bb] = 1
        mask[c] = 2
        masks[n] = mask
        for i, z in enumerate((a, bb, c, d)):
            counts[i] += int(z.sum())
    assert all(count > 0 for count in counts), 'empty zone: stop before training'
    b.dump(
        b.WORK / 'zone_audit.json',
        dict(
            counts=dict(zip('ABCD', counts)),
            total=sum(counts),
            thresholds=thresholds,
            importance_seconds=time.perf_counter() - started,
            estimator='mean squared minibatch denoising gradient',
        ),
    )
    torch.save({n: m.cpu() for n, m in masks.items()}, path)
    model.disable_gradient_checkpointing()
    del scores
    gc.collect()
    torch.cuda.empty_cache()
    return masks


@torch.no_grad()
def project(model, teacher, active, bound):
    """Restore frozen coordinates exactly; hold active ones in the trust region."""
    for name, p in model.named_parameters():
        anchor = teacher.get_parameter(name)
        p.copy_(anchor + torch.where(active[name], (p - anchor).clamp(-bound, bound), 0.0))


def train(e, data, tseed=42, root=None):
    """One LwU deletion of the nudity concept. Returns the checkpoint path."""
    final = b.WORK / f'lwu_s{tseed}.pt'
    if final.exists():
        return final
    masks = zones(e, data, root)
    e.unet = b.unet(trainable=True)
    model = e.unet
    active = {n: (m == 3) for n, m in masks.items()}  # Zone D only
    del masks
    b.seed(tseed)
    teacher = b.unet()
    model.train()
    model.enable_gradient_checkpointing()
    opt = torch.optim.Adam(model.parameters(), lr=CONFIG['lr'])
    ef, er = e.embed([b.FORGET] * 2), e.embed([b.RETAIN] * 2)
    n = b.CONFIG['training_images_per_set']
    order = torch.randperm(n, device='cuda')
    history = []
    started = time.perf_counter()
    for step in range(CONFIG['steps']):
        opt.zero_grad(set_to_none=True)
        losses = [0.0, 0.0]
        for micro in range(4):
            base = step * 8 + micro * 2
            ix = order[torch.arange(base, base + 2, device='cuda') % n]
            xf, tf, _ = b.batch(e, data, 'forget', ix)
            xr, tr, _ = b.batch(e, data, 'retain', ix)
            with torch.no_grad():
                good = b.predict(teacher, xf, tf, er)
                retain = b.predict(teacher, xr, tr, er)
            lf = F.mse_loss(b.predict(model, xf, tf, ef), good)
            lr = F.mse_loss(b.predict(model, xr, tr, er), retain)
            loss = lf + CONFIG['retain_weight'] * lr
            if not torch.isfinite(loss):
                raise RuntimeError('nonfinite loss')
            (loss / 4).backward()
            losses[0] += float(lf.detach()) / 4
            losses[1] += float(lr.detach()) / 4
        for name, p in model.named_parameters():
            if p.grad is not None:
                p.grad.mul_(active[name])
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
        opt.step()
        project(model, teacher, active, CONFIG['max_coordinate_displacement'])
        history.append(
            dict(
                step=step + 1,
                forget_loss=losses[0],
                retain_loss=losses[1],
                gradient_norm=float(norm),
                seconds=time.perf_counter() - started,
            )
        )
        if root and (step % 10 == 0 or step + 1 == CONFIG['steps']):
            b.status(root, 'training', seed=tseed, total=CONFIG['steps'], **history[-1])

    audit = dict(frozen_max=0.0, active_max=0.0, displacement_l2=0.0)
    with torch.no_grad():
        for name, p in model.named_parameters():
            assert torch.isfinite(p).all(), name
            delta = p - teacher.get_parameter(name)
            audit['frozen_max'] = max(
                audit['frozen_max'], float(delta.masked_fill(active[name], 0).abs().max())
            )
            audit['active_max'] = max(audit['active_max'], float(delta.abs().max()))
            audit['displacement_l2'] += float(delta.square().sum())
    audit['displacement_l2'] **= 0.5
    assert audit['frozen_max'] == 0
    assert audit['active_max'] <= CONFIG['max_coordinate_displacement'] + 1e-6
    assert all(p.grad is None for p in teacher.parameters())
    model.zero_grad(set_to_none=True)
    model.disable_gradient_checkpointing()
    torch.save(
        dict(
            model={n: p.cpu() for n, p in model.state_dict().items()},
            config=CONFIG,
            seed=tseed,
            history=history,
            audit=audit,
        ),
        final,
    )
    if root:
        b.dump(
            root / 'training.json',
            dict(
                **audit,
                seed=tseed,
                steps=CONFIG['steps'],
                seconds=time.perf_counter() - started,
                teacher_gradients=False,
            ),
        )
    del teacher, active, opt
    gc.collect()
    torch.cuda.empty_cache()
    return final
