import torch

from Machine_Unlearning_baselines.common import diagonal_fisher


class SSD:
    def __init__(self, model, device='cuda', alpha=10.0, dampening_constant=1.0, full_fisher=None):
        self.model = model
        self.device = device
        self.alpha = alpha
        self.dampening_constant = dampening_constant
        self.full_fisher = full_fisher

    def prepare(self, full_loader):
        self.full_fisher = diagonal_fisher(self.model, full_loader, self.device)
        return self.full_fisher

    @torch.no_grad()
    def _dampen(self, forget_fisher):
        if self.full_fisher is None:
            raise RuntimeError('SSD requires stored full-data Fisher; call prepare(full_loader)')
        for name, parameter in self.model.named_parameters():
            if not parameter.requires_grad:
                continue
            full = self.full_fisher[name]
            forget = forget_fisher[name]
            selected = forget > self.alpha * full
            beta = torch.minimum(
                self.dampening_constant * full / forget.clamp_min(1e-12), torch.ones_like(forget)
            )
            parameter.mul_(torch.where(selected, beta, torch.ones_like(beta)))

    def unlearn(self, forget_loader, retain_loader=None, full_loader=None):
        del retain_loader
        if self.full_fisher is None:
            if full_loader is None:
                raise RuntimeError('SSD needs full_loader once or a stored full_fisher')
            self.prepare(full_loader)
        self._dampen(diagonal_fisher(self.model, forget_loader, self.device))
        return self.model
