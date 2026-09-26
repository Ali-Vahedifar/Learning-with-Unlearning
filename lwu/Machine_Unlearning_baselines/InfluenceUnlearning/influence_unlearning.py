"""IU: influence unlearning via a Woodbury-Fisher inverse-Hessian step.

One Newton-style update theta <- theta + alpha * H^-1 (g_f - g_r), where the
inverse Hessian is approximated by the WoodFisher recursion over single retain
examples.  No training loop: the whole method is two gradient passes plus the
recursion, which is why its cost profile differs from every other baseline.
"""

import torch
import torch.nn.functional as F
from torch.autograd import grad
from torch.utils.data import DataLoader


def _flat_grad(model, loss):
    params = [p for p in model.parameters() if p.requires_grad]
    return torch.cat([g.reshape(-1) for g in grad(loss, params)])


class InfluenceUnlearning:
    def __init__(self, model, device='cuda', alpha=1.0, damping=1000.0, max_samples=1000):
        self.model = model
        self.device = device
        self.alpha = alpha
        self.damping = damping
        self.max_samples = max_samples

    def _mean_gradient(self, loader):
        total = 0
        accumulated = None
        for inputs, targets in loader:
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            self.model.zero_grad(set_to_none=True)
            batch = inputs.size(0)
            flat = _flat_grad(self.model, F.cross_entropy(self.model(inputs), targets)) * batch
            accumulated = flat if accumulated is None else accumulated + flat
            total += batch
        if accumulated is None:
            raise ValueError('cannot take a gradient over an empty loader')
        return accumulated, total

    def _woodfisher(self, loader, vector):
        """WoodFisher: rank-one updates of H^-1 v over single retain examples."""
        k_vec = vector.clone()
        o_vec = None
        for index, (inputs, targets) in enumerate(loader):
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            self.model.zero_grad(set_to_none=True)
            sample_grad = _flat_grad(self.model, F.cross_entropy(self.model(inputs), targets))
            with torch.no_grad():
                if o_vec is None:
                    o_vec = sample_grad.clone()
                else:
                    tmp = torch.dot(o_vec, sample_grad)
                    k_vec -= (torch.dot(k_vec, sample_grad) / (self.damping + tmp)) * o_vec
                    o_vec -= (tmp / (self.damping + tmp)) * o_vec
            if index > self.max_samples:
                break
        return k_vec

    def unlearn(self, forget_loader, retain_loader):
        self.model.eval()
        forget_grad, forget_n = self._mean_gradient(forget_loader)
        retain_grad, retain_n = self._mean_gradient(retain_loader)
        retain_grad *= forget_n / ((forget_n + retain_n) * retain_n)
        forget_grad /= forget_n + retain_n

        single = DataLoader(retain_loader.dataset, batch_size=1, shuffle=False)
        perturbation = self._woodfisher(single, forget_grad - retain_grad)

        with torch.no_grad():
            offset = 0
            for param in self.model.parameters():
                if not param.requires_grad:
                    continue
                length = param.numel()
                param.add_((self.alpha * perturbation[offset : offset + length]).view_as(param))
                offset += length
        self.model.zero_grad(set_to_none=True)
        return self.model
