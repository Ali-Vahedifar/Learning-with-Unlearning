"""l1-sparse: fine-tuning on D_r with an L1 penalty that decays to zero.

Sparsity is applied while the model is still being repaired and then released
(`no_l1_epochs`), so the final epochs are plain fine-tuning; this is the
schedule of the official implementation.
"""

import torch
import torch.nn.functional as F


class L1Sparse:
    def __init__(
        self,
        model,
        device='cuda',
        epochs=10,
        lr=1e-3,
        alpha=1e-4,
        no_l1_epochs=0,
        momentum=0.9,
        weight_decay=5e-4,
    ):
        self.model = model
        self.device = device
        self.epochs = epochs
        self.lr = lr
        self.alpha = alpha
        self.no_l1_epochs = no_l1_epochs
        self.momentum = momentum
        self.weight_decay = weight_decay

    def _l1(self):
        return torch.linalg.norm(torch.cat([p.view(-1) for p in self.model.parameters()]), ord=1)

    def unlearn(self, forget_loader, retain_loader):
        optimizer = torch.optim.SGD(
            self.model.parameters(),
            lr=self.lr,
            momentum=self.momentum,
            weight_decay=self.weight_decay,
        )
        decay_epochs = max(1, self.epochs - self.no_l1_epochs)
        self.model.train()
        for epoch in range(self.epochs):
            current_alpha = self.alpha * (1 - epoch / decay_epochs) if epoch < decay_epochs else 0.0
            for inputs, targets in retain_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                optimizer.zero_grad(set_to_none=True)
                loss = F.cross_entropy(self.model(inputs), targets)
                if current_alpha:
                    loss = loss + current_alpha * self._l1()
                loss.backward()
                optimizer.step()
        return self.model
