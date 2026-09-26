import torch
import torch.nn.functional as F

from Machine_Unlearning_baselines.common import clone_frozen, kd_kl


class SCRUB:
    def __init__(
        self,
        model,
        device='cuda',
        epochs=5,
        msteps=2,
        lr=5e-4,
        temperature=4.0,
        alpha=1.0,
        gamma=1.0,
        momentum=0.9,
        weight_decay=5e-4,
        max_grad_norm=None,
    ):
        if msteps > epochs:
            raise ValueError('msteps cannot exceed epochs')
        if max_grad_norm is not None and max_grad_norm <= 0:
            raise ValueError('max_grad_norm must be positive')
        self.max_grad_norm = max_grad_norm
        self.model = model
        self.device = device
        self.epochs = epochs
        self.msteps = msteps
        self.lr = lr
        self.temperature = temperature
        self.alpha = alpha
        self.gamma = gamma
        self.momentum = momentum
        self.weight_decay = weight_decay

    def unlearn(self, forget_loader, retain_loader):
        teacher = clone_frozen(self.model)
        optimizer = torch.optim.SGD(
            self.model.parameters(),
            lr=self.lr,
            momentum=self.momentum,
            weight_decay=self.weight_decay,
        )
        for epoch in range(self.epochs):
            if epoch < self.msteps:
                self.model.train()
                for inputs, _ in forget_loader:
                    inputs = inputs.to(self.device)
                    with torch.no_grad():
                        target_logits = teacher(inputs)
                    optimizer.zero_grad(set_to_none=True)
                    loss = -kd_kl(self.model(inputs), target_logits, self.temperature)
                    loss.backward()
                    if self.max_grad_norm is not None:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                    optimizer.step()

            self.model.train()
            for inputs, targets in retain_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                with torch.no_grad():
                    target_logits = teacher(inputs)
                optimizer.zero_grad(set_to_none=True)
                outputs = self.model(inputs)
                loss = self.gamma * F.cross_entropy(outputs, targets) + self.alpha * kd_kl(
                    outputs, target_logits, self.temperature
                )
                loss.backward()
                optimizer.step()
        return self.model
