"""BS: boundary shrink (Chen et al., CVPR 2023).

Every forget example is relabelled with the class an FGSM step pushes it into
-- its nearest neighbouring decision region -- and the model is fine-tuned on
those labels, which contracts the forget class's region instead of erasing it.

The perturbation is taken in normalised input space (the loaders normalise),
so there is no [0, 1] clamp/discretisation step here; `bound` is therefore in
units of normalised pixels, and 0.1 reproduces the paper's setting on CIFAR.
"""

from copy import deepcopy

import torch
import torch.nn.functional as F


class BoundaryShrink:
    def __init__(
        self, model, device='cuda', epochs=5, lr=1e-4, bound=0.1, momentum=0.9, weight_decay=5e-4
    ):
        self.model = model
        self.device = device
        self.epochs = epochs
        self.lr = lr
        self.bound = bound
        self.momentum = momentum
        self.weight_decay = weight_decay

    def _adversarial_labels(self, reference, inputs, targets):
        """Label each input with the class an FGSM step of size `bound` lands in."""
        reference.zero_grad(set_to_none=True)
        adversarial = inputs.detach().clone().requires_grad_(True)
        F.cross_entropy(reference(adversarial), targets).backward()
        with torch.no_grad():
            moved = adversarial + self.bound * adversarial.grad.detach().sign()
            labels = reference(moved).argmax(dim=1)
        reference.zero_grad(set_to_none=True)
        return labels.detach()

    def unlearn(self, forget_loader, retain_loader=None):
        reference = deepcopy(self.model).to(self.device).eval()
        optimizer = torch.optim.SGD(
            self.model.parameters(),
            lr=self.lr,
            momentum=self.momentum,
            weight_decay=self.weight_decay,
        )
        for _ in range(self.epochs):
            for inputs, targets in forget_loader:
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                labels = self._adversarial_labels(reference, inputs, targets)
                self.model.train()
                optimizer.zero_grad(set_to_none=True)
                F.cross_entropy(self.model(inputs), labels).backward()
                optimizer.step()
        return self.model
