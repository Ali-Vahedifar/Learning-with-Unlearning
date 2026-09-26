"""GA: gradient ascent on the forget set (NegGrad).

The cheapest unlearning baseline and the least controlled: nothing anchors the
retained data, so the learning rate alone decides whether the model forgets or
collapses.  It is included as the lower bound every other method must beat.
"""

import torch
import torch.nn.functional as F


class GradientAscent:
    def __init__(self, model, device='cuda', epochs=5, lr=1e-4, momentum=0.9, weight_decay=5e-4):
        self.model = model
        self.device = device
        self.epochs = epochs
        self.lr = lr
        self.momentum = momentum
        self.weight_decay = weight_decay

    def unlearn(self, forget_loader, retain_loader=None):
        optimizer = torch.optim.SGD(
            self.model.parameters(),
            lr=self.lr,
            momentum=self.momentum,
            weight_decay=self.weight_decay,
        )
        self.model.train()
        for _ in range(self.epochs):
            for inputs, targets in forget_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                optimizer.zero_grad(set_to_none=True)
                (-F.cross_entropy(self.model(inputs), targets)).backward()
                optimizer.step()
        return self.model
