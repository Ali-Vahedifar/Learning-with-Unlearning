import torch
import torch.nn as nn
import torch.nn.functional as F

from Machine_Unlearning_baselines.common import paired_batches


class SalUn:
    def __init__(
        self,
        model,
        device='cuda',
        epochs=5,
        lr=1e-3,
        sparsity=0.5,
        alpha=1.0,
        momentum=0.9,
        weight_decay=5e-4,
        num_classes=None,
        seed=0,
    ):
        if not 0.0 < sparsity <= 1.0:
            raise ValueError('sparsity must lie in (0, 1]')
        self.model = model
        self.device = device
        self.epochs = epochs
        self.lr = lr
        self.sparsity = sparsity
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

    def saliency_mask(self, forget_loader):
        grads = {
            name: torch.zeros_like(param)
            for name, param in self.model.named_parameters()
            if param.requires_grad
        }
        was_training = self.model.training
        self.model.eval()
        total = 0
        for inputs, targets in forget_loader:
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            self.model.zero_grad(set_to_none=True)
            F.cross_entropy(self.model(inputs), targets).backward()
            count = inputs.size(0)
            for name, param in self.model.named_parameters():
                if param.requires_grad and param.grad is not None:
                    grads[name].add_(param.grad.detach(), alpha=count)
            total += count
        if total == 0:
            raise ValueError('cannot build a saliency mask from an empty forget set')
        for value in grads.values():
            value.div_(total)

        flat = torch.cat([value.abs().flatten() for value in grads.values()])
        keep = max(1, int(round(self.sparsity * flat.numel())))
        threshold = torch.topk(flat, keep, largest=True).values.min()
        mask = {name: (value.abs() >= threshold) for name, value in grads.items()}

        self.model.zero_grad(set_to_none=True)
        if was_training:
            self.model.train()
        return mask

    def unlearn(self, forget_loader, retain_loader):
        mask = self.saliency_mask(forget_loader)
        num_classes = self._infer_num_classes()
        if num_classes < 2:
            raise ValueError('random labelling needs at least two classes')
        generator = torch.Generator().manual_seed(self.seed)
        optimizer = torch.optim.SGD(
            self.model.parameters(), lr=self.lr, momentum=self.momentum, weight_decay=0.0
        )
        self.model.train()
        for _ in range(self.epochs):
            for (forget_x, forget_y), (retain_x, retain_y) in paired_batches(
                forget_loader, retain_loader
            ):
                forget_x = forget_x.to(self.device)
                forget_y = forget_y.to(self.device)
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
                for name, param in self.model.named_parameters():
                    if param.grad is None or name not in mask:
                        continue
                    if self.weight_decay:
                        param.grad.add_(param.detach(), alpha=self.weight_decay)
                    param.grad.mul_(mask[name])
                optimizer.step()
        return self.model
