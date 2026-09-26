"""CPU checks for the generative studies: run with `python -m pytest -q tests` from generation/."""

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from diffusion.baselines import saliency_masks, train_masked  # noqa: E402
from diffusion.ddpm import CondUNet, Diffusion  # noqa: E402


@pytest.fixture(scope='module')
def tiny():
    torch.manual_seed(0)
    model = CondUNet(base=8, cemb=16)
    x = torch.randn(8, 3, 32, 32)
    y = torch.tensor([0, 0, 0, 0, 1, 2, 3, 4])
    return model, x, y


def test_conditional_unet_and_forward_process(tiny):
    model, x, y = tiny
    diff = Diffusion(device='cpu')
    t = torch.randint(0, diff.T, (len(x),))
    xt = diff.q_sample(x, t, torch.randn_like(x))
    assert xt.shape == x.shape and torch.isfinite(xt).all()
    out = model(xt, t, y)
    assert out.shape == x.shape and torch.isfinite(out).all()
    # The null token is a real embedding row, not an out-of-range index.
    assert torch.isfinite(model(xt, t, torch.full_like(y, model.num_classes))).all()


def test_saliency_and_random_masks_have_identical_density(tiny):
    model, x, y = tiny
    masks = saliency_masks(model, x[y == 0], y[y == 0], dev='cpu')
    counts = {k: sum(int(v.sum()) for v in m.values()) for k, m in masks.items()}
    total = sum(p.numel() for p in model.parameters())
    assert counts['salun'] == counts['random'] == total // 2
    # Same budget, different coordinates: the criterion is what differs between the rows.
    assert any(not torch.equal(masks['salun'][n], masks['random'][n]) for n in masks['salun'])


def test_masked_training_moves_only_masked_weights(tiny):
    model, x, y = tiny
    mask = {n: (torch.rand_like(p) < 0.5).float() for n, p in model.named_parameters()}
    trained, history = train_masked(
        model,
        mask,
        x[y != 0],
        y[y != 0],
        x[y == 0],
        y[y == 0],
        steps=3,
        bs=4,
        dev='cpu',
        log=lambda row: None,
    )
    moved = frozen = 0
    for n, p in trained.named_parameters():
        delta = (p - model.get_parameter(n)).abs()
        moved += int((delta > 0).sum())
        frozen += int((delta * (1 - mask[n]) > 0).sum())
    assert frozen == 0 and moved > 0 and history


def test_heldout_prompt_protocol():
    pytest.importorskip('diffusers')
    from sd_concept.heldout import DRAWS, cases

    cs = cases()
    assert len({c['prompt'] for c in cs}) == 90 and len({c['seed'] for c in cs}) == 90 * DRAWS
    groups = {
        g: len({c['pid'] for c in cs if c['group'] == g})
        for g in ('forget', 'retain_clothed', 'retain_other')
    }
    assert groups == {'forget': 30, 'retain_clothed': 30, 'retain_other': 30}
    assert all('context' in c for c in cs if c['group'] == 'forget')
