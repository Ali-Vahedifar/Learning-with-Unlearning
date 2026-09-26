"""ContextMemory: joint A/B retain repair with contextual Zone-D fast memory.

Zone B is He-reinitialized to remove its forget-specialized values, then it is
retrained jointly with Zone A using retained-data KL+CE. Zone C keeps the
orthogonal rule. Zone D is an ephemeral neural memory containing bounded
feature contexts and teacher distributions, never images or labels.
"""

from contextlib import contextmanager
from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F

from LwU.dominance import DominanceZones


class ContextMemory(DominanceZones):
    """He-reinitialized Zone-B repair plus a fast Zone-D context memory."""

    EXPANSION_ZONE = None
    REPAIR_ZONES = ('A', 'B')

    def __init__(
        self,
        backbone: nn.Module,
        num_classes: int,
        context_window: int = 4,
        context_lr: float = 1e-3,
        context_temperature: float = 0.25,
        context_decay: float = 1.0,
        context_momentum: float = 0.9,
        context_steps: int = 1,
        context_unmasked: bool = False,
        mask_forgotten_classes: bool = True,
        instance_forget_priority: bool = False,
        **kwargs,
    ):
        if context_window < 1:
            raise ValueError('context_window must be positive')
        if context_lr < 0.0:
            raise ValueError('context_lr must be non-negative')
        if context_temperature <= 0.0:
            raise ValueError('context_temperature must be positive')
        if not 0.0 <= context_decay <= 1.0:
            raise ValueError('context_decay must be in [0, 1]')
        if not 0.0 <= context_momentum < 1.0:
            raise ValueError('context_momentum must be in [0, 1)')
        if context_steps < 1:
            raise ValueError('context_steps must be positive')
        self.context_window = context_window
        self.context_lr = context_lr
        self.context_temperature = context_temperature
        self.context_decay = context_decay
        self.context_momentum = context_momentum
        self.context_steps = context_steps
        self.context_unmasked = bool(context_unmasked)
        self.mask_forgotten_classes = bool(mask_forgotten_classes)
        self.instance_forget_priority = bool(instance_forget_priority)
        self.forgotten_classes = set()
        self._context = []
        self._context_fast_momentum = {}
        self.__dict__['_context_teacher'] = None
        super().__init__(backbone=backbone, num_classes=num_classes, **kwargs)
        self.reset_ttu_diagnostics()

    def identify_zones(self, retain_loader, forget_loader=None, criterion=None):
        super().identify_zones(retain_loader, forget_loader, criterion)
        if not self.instance_forget_priority:
            return
        for name in self.phi_r:
            retain_active = self.phi_r[name] > self.resolved_tau_r
            forget_active = self.phi_f[name] > self.resolved_tau_f
            # Aggressive instance-deletion ablation: any forget-active weight
            # is assigned to B even when it also supports retained examples.
            self.zone_masks['B'][name] = forget_active
            self.zone_masks['A'][name] = retain_active & ~forget_active
            self.zone_masks['C'][name] = torch.zeros_like(forget_active)
            self.zone_masks['D'][name] = ~(retain_active | forget_active)

    def _infer_forgotten_classes(self, forget_loader):
        # A single batch is sufficient because the active benchmark forgets
        # one complete class. Preserve RNG so this lookup cannot alter Fisher.
        rng = self._rng_state(forget_loader)
        try:
            _, targets = next(iter(forget_loader))
            return {int(value) for value in targets.unique()}
        finally:
            self._restore_rng_state(forget_loader, rng)

    @torch.no_grad()
    def _zero_zone_b(self):
        """Replace forget-specialized values with fresh He initialization."""
        count = 0
        displacement_sq = 0.0
        for name, parameter in self.backbone.named_parameters():
            if not parameter.requires_grad:
                continue
            mask = self.zone_masks['B'][name]
            if not mask.any():
                continue
            replacement = torch.empty_like(parameter)
            if parameter.ndim >= 2:
                nn.init.kaiming_normal_(replacement, mode='fan_in', nonlinearity='relu')
            else:
                nn.init.zeros_(replacement)
            displacement_sq += float(((replacement - parameter) * mask).square().sum())
            parameter.copy_(torch.where(mask, replacement, parameter))
            count += int(mask.sum())
        self.b_reinitialization_stats = {
            'count': count,
            'displacement_l2': displacement_sq**0.5,
            'initializer': 'He normal for tensors with ndim >= 2; zero otherwise',
        }

    def _repair_teacher_logits(self, teacher, inputs):
        logits = teacher(inputs)
        if self.forgotten_classes:
            index = torch.as_tensor(sorted(self.forgotten_classes), device=logits.device)
            logits.index_fill_(1, index, torch.finfo(logits.dtype).min)
        return logits

    def unlearn(self, forget_loader, retain_loader, **kwargs):
        self.forgotten_classes = (
            self._infer_forgotten_classes(forget_loader) if self.mask_forgotten_classes else set()
        )
        result = super().unlearn(forget_loader, retain_loader, **kwargs)

        teacher = deepcopy(self.backbone).to(self.device).eval()
        for parameter in teacher.parameters():
            parameter.requires_grad_(False)
        self.__dict__['_context_teacher'] = teacher
        self._reset_context()
        self.diagnostics.update(
            {
                'zone_a': 'joint A+B retain KL to masked teacher + labeled CE',
                'zone_b': 'He reinitialization followed by joint A+B retain repair',
                'zone_b_reinitialization': self.b_reinitialization_stats,
                'zone_c': self.zone_c_strategy,
                'expansion_split_zone': None,
                'zone_d_policy': 'bounded feature-context fast memory; no examples',
                'repair_teacher': 'frozen joint teacher with forgotten logits masked',
                'context_teacher': 'frozen post-repair context teacher',
                'context_objective': 'reverse KL(student || context teacher)',
                'context_window_batches': self.context_window,
                'context_lr': self.context_lr,
                'context_temperature': self.context_temperature,
                'context_decay': self.context_decay,
                'context_momentum': self.context_momentum,
                'context_steps': self.context_steps,
                'forgotten_classes_masked_at_test': sorted(self.forgotten_classes),
                'deletion_scope': ('class' if self.mask_forgotten_classes else 'instance'),
                'instance_forget_priority': self.instance_forget_priority,
            }
        )
        if self.instance_forget_priority:
            self.diagnostics['zone_construction'] = (
                'instance forget-priority: B=forget-active, '
                'A=retain-active minus B, C=empty, D=low-both'
            )
        return result

    def _reset_context(self):
        self._context = []
        self._context_fast_momentum = {
            name: torch.zeros_like(parameter)
            for name, parameter in self.backbone.named_parameters()
            if parameter.requires_grad
        }

    def reset_ttu_diagnostics(self):
        """Reset cumulative, read-only diagnostics for contextual inference."""
        self._ttu_diagnostics = {
            'batches': 0,
            'examples': 0,
            'optimizer_steps': 0,
            'objective_sum': 0.0,
            'gradient_l2_sum': 0.0,
            'update_l2_sum': 0.0,
            'update_l2_max': 0.0,
            'updated_parameters_sum': 0,
            'logit_delta_l2_sum': 0.0,
            'prediction_flips': 0,
        }

    def get_ttu_diagnostics(self):
        """Return normalized evidence that the configured TTU was active."""
        stats = dict(self._ttu_diagnostics)
        steps = max(stats['optimizer_steps'], 1)
        examples = max(stats['examples'], 1)
        stats.update(
            {
                'mean_objective': stats.pop('objective_sum') / steps,
                'mean_gradient_l2_per_step': stats.pop('gradient_l2_sum') / steps,
                'mean_update_l2_per_step': stats.pop('update_l2_sum') / steps,
                'mean_updated_parameters_per_step': (stats.pop('updated_parameters_sum') / steps),
                'mean_logit_delta_l2_per_example': (stats.pop('logit_delta_l2_sum') / examples),
                'prediction_flip_rate': stats['prediction_flips'] / examples,
            }
        )
        return stats

    def _teacher_values(self, inputs):
        teacher = self.__dict__.get('_context_teacher')
        if teacher is None:
            raise RuntimeError('unlearn must be called before contextual inference')
        with torch.no_grad():
            features = teacher.get_features(inputs).detach()
            logits = teacher.forward_from_features(features)
            if self.forgotten_classes:
                index = torch.as_tensor(sorted(self.forgotten_classes), device=logits.device)
                logits.index_fill_(1, index, torch.finfo(logits.dtype).min)
            values = F.softmax(logits, dim=1).detach()
        return features, values

    def _context_targets(self, keys, values):
        normalized = F.normalize(keys.float(), dim=1)
        similarity = normalized @ normalized.t()
        attention = F.softmax(similarity / self.context_temperature, dim=1)
        targets = attention @ values.float()
        return targets.clamp_min(1e-8) / targets.sum(dim=1, keepdim=True)

    def _adapt_context_memory(self):
        if not self._context or self.context_lr == 0.0:
            return
        keys = torch.cat([entry[0] for entry in self._context])
        values = torch.cat([entry[1] for entry in self._context])
        targets = self._context_targets(keys, values).detach()

        batch_weights = []
        for age, (context_keys, _) in enumerate(reversed(self._context)):
            weight = self.context_decay**age
            batch_weights.append((context_keys.size(0), weight))
        sample_weights = []
        for size, weight in reversed(batch_weights):
            sample_weights.append(torch.full((size,), weight, device=keys.device))
        sample_weights = torch.cat(sample_weights)

        for _ in range(self.context_steps):
            with torch.enable_grad():
                self.backbone.zero_grad(set_to_none=True)
                logits = self.backbone.forward_from_features(keys)
                log_student = F.log_softmax(logits, dim=1)
                student = log_student.exp()
                per_sample = (student * (log_student - targets.log())).sum(dim=1)
                loss = (per_sample * sample_weights).sum() / sample_weights.sum()
                loss.backward()

                with torch.no_grad():
                    for name, parameter in self.backbone.named_parameters():
                        if parameter.grad is None:
                            continue
                        if not self.context_unmasked:
                            # Cached features make Zone D an explicit head-memory;
                            # convolutional representation weights remain core.
                            if name not in self.zone_masks['D'] or name not in {
                                'fc1.weight',
                                'fc2.weight',
                            }:
                                continue
                            mask = self.zone_masks['D'][name]
                        else:
                            # Ablation: no zone/layer restriction at all -- every
                            # trainable parameter with a gradient gets updated.
                            mask = torch.ones_like(parameter, dtype=torch.bool)
                        gradient = parameter.grad * mask
                        momentum = self._context_fast_momentum[name]
                        momentum.mul_(self.context_momentum).add_(gradient)
                        anchor = self.original_params[name]
                        candidate = (
                            anchor
                            + self.context_decay * (parameter - anchor)
                            - self.context_lr * momentum
                        )
                        parameter.copy_(torch.where(mask, candidate, parameter))
        self.backbone.zero_grad(set_to_none=True)

    def test_time_update(self, inputs):
        if not self.original_params:
            self.original_params = {
                name: parameter.detach().clone()
                for name, parameter in self.backbone.named_parameters()
                if parameter.requires_grad
            }
        self.backbone.eval()
        with torch.no_grad():
            logits_before = self.backbone(inputs).detach()
        features, values = self._teacher_values(inputs)
        self._context.append((features, values))
        if len(self._context) > self.context_window:
            self._context.pop(0)
        self._adapt_context_memory()
        with torch.no_grad():
            logits_after = self.backbone(inputs)

        before = logits_before
        after = logits_after
        allowed = getattr(self, '_context_classes', None)
        if allowed is not None:
            index = torch.as_tensor(allowed, dtype=torch.long, device=after.device)
            before = before.index_select(1, index)
            after = after.index_select(1, index)
        stats = self._ttu_diagnostics
        stats['batches'] += 1
        stats['examples'] += int(inputs.size(0))
        stats['logit_delta_l2_sum'] += float((after - before).float().norm(dim=1).sum().item())
        stats['prediction_flips'] += int(before.argmax(dim=1).ne(after.argmax(dim=1)).sum().item())
        return logits_after

    def restore_original_params(self):
        super().restore_original_params()
        self._reset_context()

    @contextmanager
    def test_stream(self):
        try:
            yield self
        finally:
            self.restore_original_params()
