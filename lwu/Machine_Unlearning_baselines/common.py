from copy import deepcopy
from itertools import cycle
from typing import Dict, Iterator, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def reset_model(model: nn.Module) -> nn.Module:
    for module in model.modules():
        if hasattr(module, 'reset_parameters'):
            module.reset_parameters()
    return model


def train_classifier(
    model: nn.Module, loader, device='cuda', epochs=1, lr=1e-3, optimizer='adam'
) -> nn.Module:
    if optimizer == 'sgd':
        opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4)
    else:
        opt = torch.optim.Adam(model.parameters(), lr=lr)
    model.train()
    for _ in range(epochs):
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)
            opt.zero_grad(set_to_none=True)
            F.cross_entropy(model(inputs), targets).backward()
            opt.step()
    return model


def diagonal_fisher(model: nn.Module, loader, device='cuda') -> Dict[str, torch.Tensor]:
    fisher = {n: torch.zeros_like(p) for n, p in model.named_parameters() if p.requires_grad}
    was_training = model.training
    model.eval()
    total = 0
    for inputs, targets in loader:
        inputs, targets = inputs.to(device), targets.to(device)
        model.zero_grad(set_to_none=True)
        F.cross_entropy(model(inputs), targets).backward()
        batch_n = inputs.size(0)
        for name, param in model.named_parameters():
            if param.requires_grad and param.grad is not None:
                fisher[name].add_(param.grad.detach().square(), alpha=batch_n)
        total += batch_n
    if total == 0:
        raise ValueError('cannot estimate Fisher information from an empty loader')
    for value in fisher.values():
        value.div_(total)
    model.zero_grad(set_to_none=True)
    model.train(was_training)
    return fisher


def paired_batches(first, second) -> Iterator[Tuple]:
    if len(first) == 0 or len(second) == 0:
        raise ValueError('both loaders must be non-empty')
    if len(first) >= len(second):
        for left, right in zip(first, cycle(second)):
            yield left, right
    else:
        for right, left in zip(second, cycle(first)):
            yield left, right


def kd_kl(student_logits, teacher_logits, temperature=1.0):
    return (
        F.kl_div(
            F.log_softmax(student_logits / temperature, dim=1),
            F.softmax(teacher_logits / temperature, dim=1),
            reduction='batchmean',
        )
        * temperature**2
    )


def features(model: nn.Module, inputs: torch.Tensor) -> torch.Tensor:
    if hasattr(model, 'get_features'):
        return model.get_features(inputs)
    return model(inputs)


def clone_frozen(model: nn.Module) -> nn.Module:
    clone = deepcopy(model).eval()
    for parameter in clone.parameters():
        parameter.requires_grad_(False)
    return clone
