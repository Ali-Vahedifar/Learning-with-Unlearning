"""RL: random-labelling unlearning (the SalUn objective without the mask).

Forget examples are given uniformly random labels while retained examples keep
theirs, so the model is pushed to a random decision on D_f rather than merely
away from the true one.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from Machine_Unlearning_baselines.common import paired_batches


class RandomLabel:
    def __init__(
        self,
        model,
        device='cuda',
        epochs=5,
        lr=1e-3,
        alpha=1.0,
        momentum=0.9,
        weight_decay=5e-4,
        num_classes=None,
        seed=0,
    ):
        self.model = model
        self.device = device
        self.epochs = epochs
        self.lr = lr
        self.alpha = alpha
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.num_classes = num_classes
        self.seed = seed

    def _infer_num_classes(self):
        if self.num_classes is not None:
            return self.num_classes
        last = None
        for module in self.model.modules():
            if isinstance(module, nn.Linear):
                last = module
        if last is None:
            raise ValueError('cannot infer num_classes; pass it explicitly')
        return last.out_features

    def unlearn(self, forget_loader, retain_loader):
        num_classes = self._infer_num_classes()
        if num_classes < 2:
            raise ValueError('random labelling needs at least two classes')
        generator = torch.Generator().manual_seed(self.seed)
        optimizer = torch.optim.SGD(
            self.model.parameters(),
            lr=self.lr,
            momentum=self.momentum,
            weight_decay=self.weight_decay,
        )
        self.model.train()
        for _ in range(self.epochs):
            for (forget_x, forget_y), (retain_x, retain_y) in paired_batches(
                forget_loader, retain_loader
            ):
                forget_x = forget_x.to(self.device)
                retain_x = retain_x.to(self.device)
                retain_y = retain_y.to(self.device)
                random_y = torch.randint(0, num_classes, forget_y.shape, generator=generator).to(
                    self.device
                )
                optimizer.zero_grad(set_to_none=True)
                loss = F.cross_entropy(
                    self.model(forget_x), random_y
                ) + self.alpha * F.cross_entropy(self.model(retain_x), retain_y)
                loss.backward()
                optimizer.step()
        return self.model
