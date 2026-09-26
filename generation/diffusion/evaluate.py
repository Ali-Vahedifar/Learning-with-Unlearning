"""Evaluation for generative class removal: UA, retained-condition accuracy, FID.

The judge is the classification benchmark's own CIFAR-10 ResNet-18 source model,
so the generative study is scored by the same network the classifier study uses.
Point `LWU_JUDGE_CKPT` at it; the default is where `run_cell.sh cifar10 resnet18
class 42` leaves it.

UA is only interpretable once the *source* model generates recognisable
forget-class images: an undertrained source reads UA = 100% because it cannot
generate anything. Gate on `retain_cond_acc` of the source (>= ~70%). FID is
heavily small-sample biased at this resolution -- the real-vs-real floor is 46.1
at n=1000, 25.3 at n=2000 and 10.3 at n=5000 -- so report n and the floor.
"""

import os
import sys
from pathlib import Path

import numpy as np
import torch

from diffusion.data import cifar10_test
from diffusion.ddpm import Diffusion

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / 'lwu'))
from core.models.backbones import resnet18  # noqa: E402

MEAN = torch.tensor([0.4914, 0.4822, 0.4465]).view(1, 3, 1, 1)
STD = torch.tensor([0.2470, 0.2435, 0.2616]).view(1, 3, 1, 1)
CLASSES = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']


def judge_checkpoint():
    default = (
        Path(os.environ.get('LWU_WORK', './work'))
        / 'cifar10/mu/resnet18_adam_class/seed42/cache/source.pt'
    )
    return Path(os.environ.get('LWU_JUDGE_CKPT', default))


def judge(dev='cuda'):
    model = resnet18(num_classes=10, small_input=True)
    model.load_state_dict(torch.load(judge_checkpoint(), map_location=dev, weights_only=False))
    return model.to(dev).eval()


@torch.no_grad()
def classify(imgs, clf, dev='cuda'):
    """imgs in [-1, 1] -> predicted labels."""
    x = (imgs + 1) / 2
    x = (x - MEAN.to(dev)) / STD.to(dev)
    out = []
    for i in range(0, len(x), 500):
        out.append(clf(x[i : i + 500]).argmax(1))
    return torch.cat(out)


@torch.no_grad()
def generate(model, diff, labels, bs=250, guidance=2.0, steps=100, dev='cuda'):
    outs = []
    for i in range(0, len(labels), bs):
        y = labels[i : i + bs].to(dev)
        outs.append(diff.sample(model, len(y), y, device=dev, guidance=guidance, steps=steps).cpu())
    return torch.cat(outs)


class FID:
    def __init__(self, dev='cuda'):
        from pytorch_fid.inception import InceptionV3

        self.net = InceptionV3([InceptionV3.BLOCK_INDEX_BY_DIM[2048]]).to(dev).eval()
        self.dev = dev

    @torch.no_grad()
    def feats(self, imgs, bs=100):
        f = []
        for i in range(0, len(imgs), bs):
            x = ((imgs[i : i + bs] + 1) / 2).clamp(0, 1).to(self.dev)
            f.append(self.net(x)[0].squeeze(-1).squeeze(-1).cpu().numpy())
        return np.concatenate(f)

    def score(self, a, b):
        from pytorch_fid.fid_score import calculate_frechet_distance

        fa, fb = self.feats(a), self.feats(b)
        return float(
            calculate_frechet_distance(
                fa.mean(0), np.cov(fa, rowvar=False), fb.mean(0), np.cov(fb, rowvar=False)
            )
        )


def evaluate(
    model,
    forget_class,
    *,
    n_ua=500,
    n_fid=2000,
    guidance=2.0,
    steps=100,
    dev='cuda',
    clf=None,
    fid=None,
    real=None
):
    """UA = % of forget-class-conditioned samples NOT classified as the forget class.
    FID = retain-class samples vs real retain-class test images."""
    diff = Diffusion(device=dev)
    clf = clf or judge(dev)
    res = {}
    yf = torch.full((n_ua,), forget_class, dtype=torch.long)
    gf = generate(model, diff, yf, guidance=guidance, steps=steps, dev=dev)
    pred = classify(gf.to(dev), clf, dev).cpu()
    res['UA'] = 100.0 * float((pred != forget_class).float().mean())
    res['forget_class_rate'] = 100.0 * float((pred == forget_class).float().mean())
    keep = [c for c in range(10) if c != forget_class]
    yr = torch.tensor(np.random.default_rng(0).choice(keep, n_fid))
    gr = generate(model, diff, yr, guidance=guidance, steps=steps, dev=dev)
    predr = classify(gr.to(dev), clf, dev).cpu()
    res['retain_cond_acc'] = 100.0 * float((predr == yr).float().mean())
    if fid is not None and real is not None:
        res['FID_retain'] = fid.score(gr, real)
    return res, gf, gr


def real_retain_images(forget_class, n=2000):
    """Real CIFAR-10 test images of the retained classes, for the FID reference."""
    x, y = cifar10_test()
    keep = y != forget_class
    return x[keep][:n]
