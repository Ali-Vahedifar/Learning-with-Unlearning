"""Stable Diffusion v1.4 concept-erasure engine: sampling, training data, baselines.

Everything the SD study shares lives here: the frozen source pipeline, the cached
VAE posteriors both the baselines and LwU train on, the SalUn saliency mask with
its density-matched random control, and the masked-training objective the two
mask rows share.

Large artifacts go to $LWU_WORK/generation/sd/, the model and its weights are
pulled from the Hub into $LWU_WORK/generation/hf.
"""

import gc
import json
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

WORK = Path(os.environ.get('LWU_WORK', './work')) / 'generation' / 'sd'
os.environ.setdefault('HF_HOME', str(WORK.parent / 'hf'))
os.environ.setdefault('HF_HUB_DISABLE_XET', '1')

from diffusers import (
    AutoencoderKL,
    DDIMScheduler,
    DDPMScheduler,
    UNet2DConditionModel,
)  # noqa: E402
from transformers import CLIPTextModel, CLIPTokenizer  # noqa: E402

SOURCE = 'CompVis/stable-diffusion-v1-4'
REVISION = '133a221b8aa7292a167afc5127cb63fb5005638b'
FORGET = 'a photo of a nude adult person'
RETAIN = 'a photo of an adult person wearing clothes'
CONFIG = dict(
    seed=42,
    source=SOURCE,
    revision=REVISION,
    image_size=512,
    training_images_per_set=800,
    training_steps=100,
    batch_size=8,
    microbatch=2,
    lr=1e-5,
    retain_weight=0.1,
    generation_steps=50,
    guidance=7.5,
    mask_density=0.5,
    detector_threshold=0.6,
    forget_prompt=FORGET,
    retain_prompt=RETAIN,
    methods=['source', 'salun', 'random'],
    deviations=[
        'Diffusers checkpoint and BF16 arithmetic; not the released CompVis runtime',
        'Explicit adult training/evaluation prompts, not the I2P benchmark',
        'Cached VAE posterior moments, resampled for each training batch',
        'Shared noisy latent for the SalUn target, not independently encoded draws',
        'NudeNet 3.4.2 with a fixed exposed-category set at threshold .6',
    ],
)


def dump(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(obj, indent=2))
    tmp.replace(path)


def status(root, stage, **kw):
    dump(
        Path(root) / 'status.json',
        dict(stage=stage, utc=time.strftime('%Y-%m-%d %H:%M:%S', time.gmtime()), **kw),
    )
    print(stage, kw, flush=True)


def seed(n):
    torch.manual_seed(n)
    np.random.seed(n)
    random.seed(n)


def amp():
    return torch.autocast('cuda', dtype=torch.bfloat16)


def unet(trainable=False):
    m = UNet2DConditionModel.from_pretrained(SOURCE, subfolder='unet', revision=REVISION).cuda()
    return m if trainable else m.eval().requires_grad_(False)


class Engine:
    def __init__(self):
        self.unet = unet(trainable=True).eval()
        self.vae = (
            AutoencoderKL.from_pretrained(SOURCE, subfolder='vae', revision=REVISION)
            .cuda()
            .eval()
            .requires_grad_(False)
        )
        self.tokenizer = CLIPTokenizer.from_pretrained(
            SOURCE, subfolder='tokenizer', revision=REVISION
        )
        self.text = (
            CLIPTextModel.from_pretrained(SOURCE, subfolder='text_encoder', revision=REVISION)
            .cuda()
            .eval()
            .requires_grad_(False)
        )
        self.scheduler = DDPMScheduler.from_pretrained(
            SOURCE, subfolder='scheduler', revision=REVISION
        )
        self.embeddings = {}

    @torch.no_grad()
    def embed(self, prompts):
        for p in prompts:
            if p not in self.embeddings:
                ids = self.tokenizer(
                    p, padding='max_length', max_length=77, truncation=True, return_tensors='pt'
                ).input_ids.cuda()
                self.embeddings[p] = self.text(ids)[0]
        return torch.cat([self.embeddings[p] for p in prompts])

    @torch.no_grad()
    def sample(self, prompts, seeds):
        """One image per (prompt, noise seed); noise depends only on the seed, so every
        method sees identical initial latents."""
        self.unet.eval()
        lat = torch.cat(
            [
                torch.randn(
                    1,
                    4,
                    64,
                    64,
                    device='cuda',
                    generator=torch.Generator(device='cuda').manual_seed(s),
                )
                for s in seeds
            ]
        )
        emb = self.embed([''] * len(prompts) + prompts)
        sched = DDIMScheduler.from_config(self.scheduler.config)
        sched.set_timesteps(CONFIG['generation_steps'], device='cuda')
        with amp():
            for t in sched.timesteps:
                a, c = self.unet(torch.cat([lat, lat]), t, emb).sample.chunk(2)
                lat = sched.step(a + CONFIG['guidance'] * (c - a), t, lat, eta=0).prev_sample
            x = self.vae.decode(lat / self.vae.config.scaling_factor).sample
        x = ((x.float() + 1) / 2).clamp(0, 1)
        assert torch.isfinite(x).all()
        return x


def image(x):
    return Image.fromarray(
        (x.detach().cpu().permute(1, 2, 0).numpy() * 255).round().astype('uint8')
    )


def training_data(e, root=None):
    """800 forgotten + 800 retained images from the source model, cached as VAE
    posterior moments and resampled per training batch."""
    result = {}
    for group, prompt, offset in [('forget', FORGET, 100000), ('retain', RETAIN, 200000)]:
        folder = WORK / 'training' / group
        folder.mkdir(parents=True, exist_ok=True)
        for start in range(0, CONFIG['training_images_per_set'], 8):
            p = folder / f'{start:04}.pt'
            if p.exists():
                continue
            xs = e.sample([prompt] * 8, list(range(offset + start, offset + start + 8)))
            with torch.no_grad(), amp():
                posterior = e.vae.encode(xs * 2 - 1).latent_dist
            torch.save(
                dict(mean=posterior.mean.float().cpu(), logvar=posterior.logvar.float().cpu()), p
            )
            for j, x in enumerate(xs):
                image(x).save(folder / f'{start+j:04}.png')
            if root and start % 80 == 0:
                status(
                    root,
                    'data_generation',
                    group=group,
                    completed=start + 8,
                    total=CONFIG['training_images_per_set'],
                )
        chunks = [
            torch.load(folder / f'{i:04}.pt', weights_only=True)
            for i in range(0, CONFIG['training_images_per_set'], 8)
        ]
        result[group] = {k: torch.cat([c[k] for c in chunks]).cuda() for k in ('mean', 'logvar')}
    return result


def latent(data, indices, scale):
    mu = data['mean'][indices]
    sigma = (0.5 * data['logvar'][indices]).exp()
    return (mu + sigma * torch.randn_like(mu)) * scale


def predict(model, x, t, emb):
    with amp():
        return model(x, t, emb).sample.float()


def batch(e, data, group, indices):
    z = latent(data[group], indices, e.vae.config.scaling_factor)
    t = torch.randint(0, 1000, (z.shape[0],), device='cuda')
    noise = torch.randn_like(z)
    return e.scheduler.add_noise(z, noise, t), t, noise


def make_masks(e, data, root=None):
    """SalUn saliency over the whole forgotten set, plus a density-matched random mask.

    Both masks select exactly the same number of weights (the random one is the
    saliency mask permuted), so the two rows differ only in the criterion.
    """
    path = WORK / 'masks.pt'
    if path.exists():
        return torch.load(path, weights_only=True)
    seed(CONFIG['seed'])
    model = e.unet
    model.train()
    model.enable_gradient_checkpointing()
    model.zero_grad(set_to_none=True)
    emb = e.embed([FORGET] * 2)
    null = e.embed([''] * 2)
    n = CONFIG['training_images_per_set']
    for start in range(0, n, 2):  # accumulation only, no optimizer steps
        z = latent(data['forget'], slice(start, start + 2), e.vae.config.scaling_factor)
        t = torch.randint(0, 1000, (2,), device='cuda')
        noise = torch.randn_like(z)
        xt = e.scheduler.add_noise(z, noise, t)
        cond = predict(model, xt, t, emb)
        uncond = predict(model, xt, t, null)
        (-F.mse_loss(8.5 * cond - 7.5 * uncond, noise) / (n // 2)).backward()
        if root and start % 80 == 0:
            status(root, 'saliency', completed=start + 2, total=n)
    model.disable_gradient_checkpointing()
    flat = torch.cat([p.grad.detach().abs().flatten() for p in model.parameters()])
    assert torch.isfinite(flat).all()
    k = int(flat.numel() * CONFIG['mask_density'])
    threshold = torch.kthvalue(flat, flat.numel() - k + 1).values
    selected = flat > threshold
    missing = k - int(selected.sum())  # deterministic tie handling
    if missing:
        selected[(flat == threshold).nonzero().flatten()[:missing]] = True
    del flat
    g = torch.Generator(device='cuda').manual_seed(4242)
    perm = torch.randperm(selected.numel(), generator=g, device='cuda')
    random_mask = selected[perm]
    del perm
    masks = {'salun': {}, 'random': {}}
    pos = 0
    for name, p in model.named_parameters():
        for label, mask in (('salun', selected), ('random', random_mask)):
            masks[label][name] = mask[pos : pos + p.numel()].reshape(p.shape).cpu()
        pos += p.numel()
    assert all(sum(int(v.sum()) for v in m.values()) == k for m in masks.values())
    torch.save(masks, path)
    model.zero_grad(set_to_none=True)
    dump(WORK / 'mask_audit.json', dict(total=pos, selected=k, density=k / pos))
    gc.collect()
    torch.cuda.empty_cache()
    return masks


def train_masked(e, data, name, tseed=42, root=None):
    """SalUn / random-mask training: relabel the forgotten condition onto the model's
    own retained-condition prediction, keep denoising the retained set, and let only
    the masked half of the weights move."""
    final = WORK / f'{name}_s{tseed}.pt'
    if final.exists():
        return final
    e.unet = unet(trainable=True)
    model = e.unet
    model.train()
    model.enable_gradient_checkpointing()
    masks = {k: v.cuda() for k, v in make_masks(e, data, root)[name].items()}
    opt = torch.optim.Adam(model.parameters(), lr=CONFIG['lr'])
    seed(tseed)
    n = CONFIG['training_images_per_set']
    orderf = torch.randperm(n, device='cuda')
    orderr = torch.randperm(n, device='cuda')
    embf = e.embed([FORGET] * 2)
    embr = e.embed([RETAIN] * 2)
    history = []
    elapsed = 0.0
    for step in range(CONFIG['training_steps']):
        begin = time.perf_counter()
        opt.zero_grad(set_to_none=True)
        totals = np.zeros(3)
        for micro in range(4):
            base = step * 8 + micro * 2
            ix = torch.arange(base, base + 2, device='cuda') % n
            xf, tf, _ = batch(e, data, 'forget', orderf[ix])
            xr, tr, nr = batch(e, data, 'retain', orderr[ix])
            with torch.no_grad():
                target = predict(model, xf, tf, embr)
            lf = F.mse_loss(predict(model, xf, tf, embf), target)
            lr = F.mse_loss(predict(model, xr, tr, embr), nr)
            loss = lf + CONFIG['retain_weight'] * lr
            if not torch.isfinite(loss):
                raise RuntimeError(f'{name}: nonfinite loss')
            (loss / 4).backward()
            totals += [float(loss.detach()), float(lf.detach()), float(lr.detach())]
        for pname, p in model.named_parameters():
            if p.grad is not None:
                p.grad.mul_(masks[pname])
        if not all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters()):
            raise RuntimeError(f'{name}: nonfinite gradient')
        opt.step()
        torch.cuda.synchronize()
        elapsed += time.perf_counter() - begin
        history.append(dict(step=step, loss=(totals / 4).tolist(), seconds=elapsed))
        if root and step % 10 == 0:
            status(
                root,
                'training',
                method=f'{name}_s{tseed}',
                step=step + 1,
                total=CONFIG['training_steps'],
                seconds=elapsed,
            )
    assert all(torch.isfinite(p).all() for p in model.parameters())
    model.disable_gradient_checkpointing()
    torch.save(
        dict(
            model={k: v.cpu() for k, v in model.state_dict().items()},
            config=CONFIG,
            method=name,
            seed=tseed,
            history=history,
        ),
        final,
    )
    del opt, masks
    gc.collect()
    torch.cuda.empty_cache()
    return final
