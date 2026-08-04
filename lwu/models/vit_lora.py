"""
ViT-B/16 + LoRA backbone for the LwU framework.

This module provides the parameter-efficient instantiation of LwU described in
Section 3 of the paper: the pretrained ViT backbone is frozen and the four-zone
decomposition is applied exclusively to the low-rank adapter matrices A and B.

Design note
-----------
No parallel zone implementation is required. `LwU.compute_ssv`,
`LwU.identify_zones`, `LwU.apply_zone_operations` and `LwU.test_time_update`
all iterate over `backbone.named_parameters()` filtered by `param.requires_grad`.
Freezing every pretrained weight and leaving only the LoRA matrices trainable
therefore restricts dual-SSV scoring, zone construction, Zone B reinitialisation,
Zone C orthogonal projection and TTU to the adapters automatically, using
exactly the same code path as the ResNet experiments.

Usage
-----
    from lwu.models import get_backbone, LwU

    backbone = get_backbone('vit_b16_lora', num_classes=100, lora_rank=8)
    model = LwU(backbone=backbone, num_classes=100, tau_r=0.49, tau_f=0.35)
"""

import math
from typing import List, Optional

import torch
import torch.nn as nn


# --------------------------------------------------------------------------- #
# LoRA adapter
# --------------------------------------------------------------------------- #

class LoRALinear(nn.Module):
    """
    Low-rank adapted linear layer.

        h = W_pretrained @ x + (alpha / r) * B @ A @ x

    `W_pretrained` is frozen. Only `lora_A` and `lora_B` receive gradients and
    are therefore the only parameters visible to the LwU zone machinery.

    Following Hu et al. (2022), `lora_A` uses Kaiming initialisation and
    `lora_B` is zero-initialised, so the adapted layer is exactly equivalent to
    the frozen layer at initialisation.
    """

    def __init__(
        self,
        base_layer: nn.Linear,
        rank: int = 8,
        alpha: float = 16.0,
        dropout: float = 0.0
    ):
        super().__init__()

        if rank <= 0:
            raise ValueError(f"LoRA rank must be positive, got {rank}")

        self.in_features = base_layer.in_features
        self.out_features = base_layer.out_features
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank

        # Frozen pretrained projection
        self.base_layer = base_layer
        for p in self.base_layer.parameters():
            p.requires_grad = False

        # Trainable low-rank factors: A in R^{r x d_in}, B in R^{d_out x r}
        self.lora_A = nn.Parameter(torch.empty(rank, self.in_features))
        self.lora_B = nn.Parameter(torch.zeros(self.out_features, rank))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

        self.lora_dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = self.base_layer(x)
        lora_out = self.lora_dropout(x) @ self.lora_A.T @ self.lora_B.T
        return base_out + self.scaling * lora_out

    def extra_repr(self) -> str:
        return (f"in_features={self.in_features}, out_features={self.out_features}, "
                f"rank={self.rank}, alpha={self.alpha}")


# --------------------------------------------------------------------------- #
# Injection
# --------------------------------------------------------------------------- #

DEFAULT_TARGET_MODULES = ('qkv', 'proj', 'fc1', 'fc2')


def inject_lora(
    model: nn.Module,
    rank: int = 8,
    alpha: float = 16.0,
    dropout: float = 0.0,
    target_modules: Optional[List[str]] = None
) -> int:
    """
    Replace every `nn.Linear` whose attribute name matches `target_modules`
    with a `LoRALinear` wrapper. Returns the number of layers adapted.
    """
    if target_modules is None:
        target_modules = list(DEFAULT_TARGET_MODULES)

    replaced = 0
    for module in model.modules():
        for child_name, child in list(module.named_children()):
            if child_name in target_modules and isinstance(child, nn.Linear):
                setattr(
                    module,
                    child_name,
                    LoRALinear(child, rank=rank, alpha=alpha, dropout=dropout)
                )
                replaced += 1
    return replaced


def freeze_non_lora(model: nn.Module, train_head: bool = True) -> None:
    """
    Freeze every parameter except the LoRA factors (and optionally the
    classification head, which must remain trainable to accommodate new classes
    under class-incremental evaluation).

    After this call, `param.requires_grad` is True only for parameters the
    LwU zone decomposition should operate on.
    """
    for name, param in model.named_parameters():
        is_lora = ('lora_A' in name) or ('lora_B' in name)
        is_head = train_head and ('head' in name or 'classifier' in name)
        param.requires_grad = bool(is_lora or is_head)


def lora_parameter_names(model: nn.Module) -> List[str]:
    """Names of all LoRA parameters, in `named_parameters()` order."""
    return [n for n, _ in model.named_parameters()
            if 'lora_A' in n or 'lora_B' in n]


def count_trainable(model: nn.Module) -> dict:
    """Parameter accounting, useful for reporting adapter overhead."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    lora = sum(p.numel() for n, p in model.named_parameters()
               if 'lora_A' in n or 'lora_B' in n)
    return {
        'total': total,
        'trainable': trainable,
        'lora': lora,
        'trainable_fraction': trainable / total if total else 0.0
    }


# --------------------------------------------------------------------------- #
# Backbone construction
# --------------------------------------------------------------------------- #

class ViTLoRABackbone(nn.Module):
    """
    ViT-B/16 pretrained on ImageNet-21K with LoRA adapters injected.

    Exposes the same call signature as the ResNet backbones so it can be passed
    directly to `LwU(backbone=...)`.
    """

    def __init__(
        self,
        num_classes: int,
        lora_rank: int = 8,
        lora_alpha: float = 16.0,
        lora_dropout: float = 0.0,
        pretrained: bool = True,
        model_name: str = 'vit_base_patch16_224.augreg_in21k',
        target_modules: Optional[List[str]] = None
    ):
        super().__init__()

        try:
            import timm
        except ImportError as exc:
            raise ImportError(
                "timm is required for the ViT-B/16 backbone. "
                "Install it with: pip install timm>=0.9.0"
            ) from exc

        self.vit = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=num_classes
        )

        self.num_adapted_layers = inject_lora(
            self.vit,
            rank=lora_rank,
            alpha=lora_alpha,
            dropout=lora_dropout,
            target_modules=target_modules
        )
        if self.num_adapted_layers == 0:
            raise RuntimeError(
                "No layers were adapted. Check `target_modules` against the "
                f"module names of '{model_name}'."
            )

        freeze_non_lora(self.vit, train_head=True)

        self.num_classes = num_classes
        self.lora_rank = lora_rank

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.vit(x)

    def expand_head(self, num_new_classes: int) -> None:
        """
        Widen the classification head for class-incremental learning, copying
        existing weights so previously learned logits are preserved.
        """
        old_head = self.vit.head
        in_features = old_head.in_features
        old_out = old_head.out_features

        new_head = nn.Linear(in_features, old_out + num_new_classes)
        with torch.no_grad():
            new_head.weight[:old_out] = old_head.weight
            new_head.bias[:old_out] = old_head.bias

        self.vit.head = new_head.to(old_head.weight.device)
        self.num_classes = old_out + num_new_classes
        for p in self.vit.head.parameters():
            p.requires_grad = True

    def parameter_summary(self) -> dict:
        return count_trainable(self.vit)


def vit_b16_lora(num_classes: int = 100, **kwargs) -> ViTLoRABackbone:
    """Factory matching the naming convention of `resnet18` / `resnet34`."""
    return ViTLoRABackbone(num_classes=num_classes, **kwargs)
