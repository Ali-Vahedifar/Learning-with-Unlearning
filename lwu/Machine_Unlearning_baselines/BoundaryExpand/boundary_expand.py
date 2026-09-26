"""BE: boundary expanding (Chen et al., CVPR 2023).

A shadow class is appended to the classifier, the forget examples are trained
into it, and the extra unit is then removed: the forget mass is routed into
capacity that no longer exists at inference.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def _last_linear(model):
    name, layer = None, None
    for candidate, module in model.named_modules():
        if isinstance(module, nn.Linear):
            name, layer = candidate, module
    if layer is None:
        raise ValueError('boundary expanding needs a Linear classifier')
    return name, layer


def _set_module(model, name, replacement):
    parts = name.split('.')
    parent = model
    for part in parts[:-1]:
        parent = getattr(parent, part)
    setattr(parent, parts[-1], replacement)


class BoundaryExpand:
    def __init__(self, model, device='cuda', epochs=5, lr=1e-4, momentum=0.9, weight_decay=5e-4):
        self.model = model
        self.device = device
        self.epochs = epochs
        self.lr = lr
        self.momentum = momentum
        self.weight_decay = weight_decay

    def _resize_head(self, delta):
        """Grow (+1) or shrink (-1) the classifier by one output unit."""
        name, layer = _last_linear(self.model)
        has_bias = layer.bias is not None
        widened = nn.Linear(
            layer.in_features,
            layer.out_features + delta,
            bias=has_bias,
            device=layer.weight.device,
            dtype=layer.weight.dtype,
        )
        with torch.no_grad():
            keep = min(layer.out_features, widened.out_features)
            widened.weight[:keep] = layer.weight[:keep]
            if has_bias:
                widened.bias[:keep] = layer.bias[:keep]
        _set_module(self.model, name, widened)
        return widened

    def unlearn(self, forget_loader, retain_loader=None):
        shadow_class = _last_linear(self.model)[1].out_features
        self._resize_head(+1)
        optimizer = torch.optim.SGD(
            self.model.parameters(),
            lr=self.lr,
            momentum=self.momentum,
            weight_decay=self.weight_decay,
        )
        self.model.train()
        for _ in range(self.epochs):
            for inputs, _ in forget_loader:
                inputs = inputs.to(self.device)
                labels = torch.full(
                    (inputs.size(0),), shadow_class, dtype=torch.long, device=self.device
                )
                optimizer.zero_grad(set_to_none=True)
                F.cross_entropy(self.model(inputs), labels).backward()
                optimizer.step()
        self._resize_head(-1)
        return self.model
