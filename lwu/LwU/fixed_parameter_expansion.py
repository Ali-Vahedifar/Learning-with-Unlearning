"""Zone-D-aware Fixed-Parameter Expansion for the LwU classifier head.

The implementation follows Kong et al.'s expand-then-contract construction:
each hidden neuron is replaced with sparse children, input edges are assigned
disjointly, and magnitude balancing removes the extra outgoing edges so the
effective nonzero-weight count does not increase.
"""

from copy import deepcopy
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from core.models.backbones import FPECNN


def _topk_mask(values, keep):
    flat = values.abs().flatten()
    result = torch.zeros_like(flat, dtype=torch.bool)
    if keep > 0:
        result[torch.topk(flat, keep, sorted=False).indices] = True
    return result.view_as(values)


def _balanced_masks(fc1_weight, fc2_weight, expansion_factor):
    hidden_size = fc1_weight.shape[0]
    num_classes = fc2_weight.shape[0]
    excess = (expansion_factor - 1) * hidden_size * num_classes
    drop_fc1 = min(fc1_weight.numel(), math.ceil(excess / 2))
    drop_fc2_expanded = excess - drop_fc1
    keep_fc1 = _topk_mask(fc1_weight, fc1_weight.numel() - drop_fc1)
    expanded_fc2 = fc2_weight.repeat_interleave(expansion_factor, dim=1)
    keep_fc2 = _topk_mask(expanded_fc2, expanded_fc2.numel() - drop_fc2_expanded)
    return (keep_fc1.repeat_interleave(expansion_factor, dim=0), keep_fc2)


class ZoneFixedParameterExpansion(nn.Module):
    """Split hidden neurons while isolating one requested parameter zone.

    ``split_zone='D'`` reproduces the original construction. ContextMemory uses
    ``split_zone='B'`` so forget-dominant connections live on separate child
    neurons before their inherited values are cleared.
    """

    def __init__(self, seed: FPECNN, zone_masks, expansion_factor: int = 2, split_zone: str = 'D'):
        super().__init__()
        if not isinstance(seed, FPECNN):
            raise TypeError('zone-aware FPE requires the FPECNN two-layer head')
        if expansion_factor != 2:
            raise ValueError('the zone/non-zone split currently requires factor=2')
        if split_zone not in 'ABCD':
            raise ValueError("split_zone must be one of 'A', 'B', 'C', or 'D'")
        self.flatten_size = seed.flatten_size
        self.hidden_size = seed.hidden_size
        self.num_classes = seed.num_classes
        self.expansion_factor = expansion_factor
        self.conv_layers = deepcopy(seed.conv_layers)
        self.relu = nn.ReLU()

        original_fc1 = seed.fc1.weight.detach().clone()
        original_fc2 = seed.fc2.weight.detach().clone()
        device = original_fc1.device
        self.fc1 = nn.Linear(self.flatten_size, expansion_factor * self.hidden_size, bias=False).to(
            device
        )
        self.fc2 = nn.Linear(expansion_factor * self.hidden_size, self.num_classes, bias=False).to(
            device
        )
        self.fc1.weight.data.copy_(original_fc1.repeat_interleave(expansion_factor, dim=0))
        self.fc2.weight.data.copy_(original_fc2.repeat_interleave(expansion_factor, dim=1))

        # Child 0 receives non-target edges; child 1 receives target-zone
        # edges. Degenerate rows are split round-robin so both children exist
        # without duplicating a connection.
        plastic = zone_masks[split_zone]['fc1.weight']
        disjoint = torch.zeros_like(self.fc1.weight, dtype=torch.bool)
        parity = torch.arange(self.flatten_size, device=device).remainder(2).bool()
        for neuron in range(self.hidden_size):
            row = plastic[neuron]
            if row.any() and (~row).any():
                child_one = row
            else:
                child_one = parity
            disjoint[2 * neuron] = ~child_one
            disjoint[2 * neuron + 1] = child_one

        balance_fc1, balance_fc2 = _balanced_masks(original_fc1, original_fc2, expansion_factor)
        self.register_buffer('sparsity_mask_fc1', disjoint)
        self.register_buffer('sparsity_mask_fc1_balance', balance_fc1)
        self.register_buffer('sparsity_mask_fc2_balance', balance_fc2)
        self.translated_zone_masks = self._translate_masks(zone_masks)

    def _translate_masks(self, masks):
        translated = {zone: {} for zone in 'ABCD'}
        effective_fc1 = self.sparsity_mask_fc1 & self.sparsity_mask_fc1_balance
        effective_fc2 = self.sparsity_mask_fc2_balance
        for zone in 'ABCD':
            for name, mask in masks[zone].items():
                if name == 'fc1.weight':
                    value = mask.repeat_interleave(2, dim=0) & effective_fc1
                elif name == 'fc2.weight':
                    value = mask.repeat_interleave(2, dim=1) & effective_fc2
                else:
                    value = mask.clone()
                translated[zone][name] = value
        # Structurally inactive dense slots are represented as plastic so the
        # four masks still partition every stored tensor; forward masks keep
        # those slots inactive and prevent a hidden parameter-budget increase.
        translated['D']['fc1.weight'] |= ~effective_fc1
        translated['D']['fc2.weight'] |= ~effective_fc2
        return translated

    def forward(self, inputs):
        features = self.conv_layers(inputs).flatten(1)
        return self.forward_from_features(features)

    def forward_from_features(self, features):
        first = self.fc1.weight * self.sparsity_mask_fc1 * self.sparsity_mask_fc1_balance
        second = self.fc2.weight * self.sparsity_mask_fc2_balance
        return F.linear(self.relu(F.linear(features, first)), second)

    def get_features(self, inputs):
        return self.conv_layers(inputs).flatten(1)

    def effective_parameter_tensors(self):
        for name, parameter in self.named_parameters():
            if name == 'fc1.weight':
                yield parameter[self.sparsity_mask_fc1 & self.sparsity_mask_fc1_balance]
            elif name == 'fc2.weight':
                yield parameter[self.sparsity_mask_fc2_balance]
            else:
                yield parameter.flatten()

    def effective_nonzero_budget(self):
        return sum(tensor.numel() for tensor in self.effective_parameter_tensors())


class ZoneDFixedParameterExpansion(ZoneFixedParameterExpansion):
    """Backward-compatible name for the original Zone-D split."""

    def __init__(self, seed: FPECNN, zone_masks, expansion_factor: int = 2):
        super().__init__(seed, zone_masks, expansion_factor=expansion_factor, split_zone='D')
