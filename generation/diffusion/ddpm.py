"""Class-conditional DDPM on CIFAR-10: model, forward process, DDIM sampler.

Small UNet (~10.4M parameters, 256 bottleneck channels) with classifier-free
guidance; the extra embedding row is the null token. This is a local
architecture, not a reproduction of a released DDPM checkpoint.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def timestep_embedding(t, dim):
    half = dim // 2
    freqs = torch.exp(-math.log(10000) * torch.arange(half, device=t.device) / half)
    a = t.float()[:, None] * freqs[None]
    return torch.cat([torch.cos(a), torch.sin(a)], dim=-1)


class Block(nn.Module):
    def __init__(self, cin, cout, cemb):
        super().__init__()
        self.n1 = nn.GroupNorm(8, cin)
        self.c1 = nn.Conv2d(cin, cout, 3, padding=1)
        self.emb = nn.Linear(cemb, cout)
        self.n2 = nn.GroupNorm(8, cout)
        self.c2 = nn.Conv2d(cout, cout, 3, padding=1)
        self.skip = nn.Conv2d(cin, cout, 1) if cin != cout else nn.Identity()

    def forward(self, x, emb):
        h = self.c1(F.silu(self.n1(x)))
        h = h + self.emb(F.silu(emb))[:, :, None, None]
        h = self.c2(F.silu(self.n2(h)))
        return h + self.skip(x)


class CondUNet(nn.Module):
    """Small class-conditional UNet.  base*4 channels at the bottleneck."""

    def __init__(self, base=64, num_classes=10, cemb=256, dropout_cond=0.1):
        super().__init__()
        self.cemb, self.num_classes, self.dropout_cond = cemb, num_classes, dropout_cond
        self.temb = nn.Sequential(nn.Linear(cemb, cemb), nn.SiLU(), nn.Linear(cemb, cemb))
        self.cembed = nn.Embedding(num_classes + 1, cemb)  # last index = null token (CFG)
        c1, c2, c3 = base, base * 2, base * 4
        self.stem = nn.Conv2d(3, c1, 3, padding=1)
        self.d1a, self.d1b = Block(c1, c1, cemb), Block(c1, c1, cemb)
        self.down1 = nn.Conv2d(c1, c1, 3, stride=2, padding=1)
        self.d2a, self.d2b = Block(c1, c2, cemb), Block(c2, c2, cemb)
        self.down2 = nn.Conv2d(c2, c2, 3, stride=2, padding=1)
        self.d3a, self.d3b = Block(c2, c3, cemb), Block(c3, c3, cemb)
        self.mid1, self.mid2 = Block(c3, c3, cemb), Block(c3, c3, cemb)
        self.u3a, self.u3b = Block(c3 + c3, c3, cemb), Block(c3, c3, cemb)
        self.up2 = nn.Upsample(scale_factor=2, mode='nearest')
        self.u2a, self.u2b = Block(c3 + c2, c2, cemb), Block(c2, c2, cemb)
        self.up1 = nn.Upsample(scale_factor=2, mode='nearest')
        self.u1a, self.u1b = Block(c2 + c1, c1, cemb), Block(c1, c1, cemb)
        self.u0 = Block(c1 + c1, c1, cemb)
        self.outn = nn.GroupNorm(8, c1)
        self.outc = nn.Conv2d(c1, 3, 3, padding=1)
        self.mid_channels = c3

    def forward(self, x, t, y):
        emb = self.temb(timestep_embedding(t, self.cemb)) + self.cembed(y)
        h0 = self.stem(x)
        h1 = self.d1b(self.d1a(h0, emb), emb)
        h2 = self.d2b(self.d2a(self.down1(h1), emb), emb)
        h3 = self.d3b(self.d3a(self.down2(h2), emb), emb)
        m = self.mid2(self.mid1(h3, emb), emb)
        u = self.u3b(self.u3a(torch.cat([m, h3], 1), emb), emb)  # 8x8
        u = self.u2b(self.u2a(torch.cat([self.up2(u), h2], 1), emb), emb)  # 16x16
        u = self.u1b(self.u1a(torch.cat([self.up1(u), h1], 1), emb), emb)  # 32x32
        u = self.u0(torch.cat([u, h0], 1), emb)
        return self.outc(F.silu(self.outn(u)))


class Diffusion:
    def __init__(self, T=1000, device='cuda'):
        self.T = T
        b = torch.linspace(1e-4, 0.02, T, device=device)
        self.beta = b
        self.alpha = 1.0 - b
        self.abar = torch.cumprod(self.alpha, 0)

    def q_sample(self, x0, t, noise):
        a = self.abar[t][:, None, None, None]
        return a.sqrt() * x0 + (1 - a).sqrt() * noise

    @torch.no_grad()
    def sample(self, model, n, y, device='cuda', guidance=2.0, steps=100):
        """DDIM sampling with classifier-free guidance."""
        model.eval()
        x = torch.randn(n, 3, 32, 32, device=device)
        null = torch.full_like(y, model.num_classes)
        ts = torch.linspace(self.T - 1, 0, steps, device=device).long()
        for i, tc in enumerate(ts):
            tb = torch.full((n,), tc, device=device, dtype=torch.long)
            ec = model(x, tb, y)
            if guidance != 1.0:
                eu = model(x, tb, null)
                e = eu + guidance * (ec - eu)
            else:
                e = ec
            ab = self.abar[tc]
            x0 = ((x - (1 - ab).sqrt() * e) / ab.sqrt()).clamp(-1, 1)
            if i < len(ts) - 1:
                ab_p = self.abar[ts[i + 1]]
                x = ab_p.sqrt() * x0 + (1 - ab_p).sqrt() * e
            else:
                x = x0
        return x
