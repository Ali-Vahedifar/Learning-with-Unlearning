"""Teacher-guided zone repair for post-hoc machine unlearning.

This stage keeps the four-zone decomposition explicit while separating the two
questions under study: how importance is estimated (SSV or diagonal Fisher),
and how conflict parameters in Zone C are treated (granular rescaling or the
orthogonal zone-C update).
"""

from copy import deepcopy
import random
import time
from typing import Dict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from Machine_Unlearning_baselines.common import kd_kl
from LwU.zones import ZoneDecomposition
from LwU.fixed_parameter_expansion import ZoneFixedParameterExpansion


class TeacherRepair(ZoneDecomposition):
    """Teacher/student repair on top of the zone decomposition.

    Operations are deliberately ordered B -> C -> A.  Zone-A distillation
    would otherwise start from a student identical to the joint teacher and
    therefore have exactly zero KL gradient.
    """

    VALID_IMPORTANCE = {
        'ssv',
        'fisher',
        'random_mask',
        'magnitude',
        'fisher_retain',
        'fisher_forget',
        'dual_fisher',
        'ssv_without_interaction',
        'ssv_retain',
        'ssv_forget',
        'dual_ssv',
    }
    VALID_ZONE_C = {'granular', 'orthogonal', 'retain_kl'}
    EXPANSION_ZONE = 'D'
    REPAIR_ZONES = ('A',)

    def __init__(
        self,
        backbone: nn.Module,
        num_classes: int,
        importance: str = 'ssv',
        zone_c: str = 'granular',
        retain_quantile: float = 0.9,
        forget_quantile: float = 0.9,
        distill_epochs: int = 3,
        distill_lr: float = 1e-3,
        temperature: float = 4.0,
        ce_weight: float = 0.0,
        per_instance_forget_fisher: bool = False,
        device: str = 'cuda',
    ):
        if importance not in self.VALID_IMPORTANCE:
            raise ValueError(f'importance must be one of {sorted(self.VALID_IMPORTANCE)}')
        if zone_c not in self.VALID_ZONE_C:
            raise ValueError(f'zone_c must be one of {sorted(self.VALID_ZONE_C)}')
        for value, name in (
            (retain_quantile, 'retain_quantile'),
            (forget_quantile, 'forget_quantile'),
        ):
            if not 0.0 <= value <= 1.0:
                raise ValueError(f'{name} must be in [0, 1]')
        if ce_weight < 0.0:
            raise ValueError('ce_weight must be non-negative')
        super().__init__(
            backbone=backbone,
            num_classes=num_classes,
            tau_r=0.0,
            tau_f=0.0,
            threshold_mode='absolute',
            device=device,
        )
        self.importance_kind = importance
        self.per_instance_forget_fisher = per_instance_forget_fisher
        self.zone_c_strategy = zone_c
        self.retain_quantile = retain_quantile
        self.forget_quantile = forget_quantile
        self.distill_epochs = distill_epochs
        self.distill_lr = distill_lr
        self.temperature = temperature
        self.ce_weight = ce_weight
        self.resolved_tau_r = 0.0
        self.resolved_tau_f = 0.0
        self.diagnostic_time = 0.0
        self.diagnostics = {}
        self._random_score_call = 0

    def _random_importance(self) -> Dict[str, torch.Tensor]:
        """Draw a reproducible mask score without perturbing training RNG."""
        scores = {}
        for parameter_index, (name, parameter) in enumerate(self.backbone.named_parameters()):
            if not parameter.requires_grad:
                continue
            generator = torch.Generator(device=parameter.device)
            generator.manual_seed(104729 + 1009 * self._random_score_call + parameter_index)
            scores[name] = torch.rand(
                parameter.shape, dtype=parameter.dtype, device=parameter.device, generator=generator
            )
        self._random_score_call += 1
        return scores

    def _importance(self, loader, family=None) -> Dict[str, torch.Tensor]:
        family = family or self.importance_kind
        if family == 'random_mask':
            return self._random_importance()
        if family == 'magnitude':
            return {
                name: parameter.detach().abs().clone()
                for name, parameter in self.backbone.named_parameters()
                if parameter.requires_grad
            }
        if family == 'fisher':
            return self.compute_fisher_information(loader)
        if family == 'ssv_interaction':
            return {
                name: score.abs()
                for name, score in self.compute_ssv_with_interaction(loader).items()
            }
        return {name: score.abs() for name, score in self.compute_importance(loader).items()}

    def _importance_preserving_rng(self, loader, family):
        """Keep estimator traversal from changing later training randomness."""
        state = self._rng_state(loader)
        try:
            # Keep the historical one-argument hook usable by tests and by
            # downstream subclasses that override ``_importance(loader)``.
            if family == self.importance_kind:
                return self._importance(loader)
            return self._importance(loader, family)
        finally:
            self._restore_rng_state(loader, state)

    @staticmethod
    def _per_instance_loader(loader):
        """Rebuild ``loader`` at batch_size=1 for an unbiased per-example Fisher.

        ``compute_fisher_information`` weights each batch's squared
        batch-mean gradient by the batch size, which is a biased estimate of
        the true per-example Fisher diagonal for batch_size > 1 (it misses
        cross-example gradient interference within the batch). Instance-level
        forgetting has few enough forget examples that scoring them one at a
        time is affordable, and removes that bias where it matters most: the
        set the zone thresholds are keyed on.
        """
        return DataLoader(loader.dataset, batch_size=1, shuffle=False, num_workers=0)

    @staticmethod
    def _zeros_like(scores: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        return {name: torch.zeros_like(value) for name, value in scores.items()}

    @staticmethod
    def _global_quantile(scores: Dict[str, torch.Tensor], quantile: float) -> float:
        flat = torch.cat([value.detach().flatten() for value in scores.values()])
        return float(torch.quantile(flat.float(), quantile).item())

    def identify_zones(self, retain_loader, forget_loader=None, criterion=None):
        del criterion
        estimator = self.importance_kind
        # ``ssv`` and ``fisher`` retain the repository's historical dual-
        # partition behavior.  The longer names below are explicit ablations.
        aliases = {'fisher': 'dual_fisher'}
        estimator = aliases.get(estimator, estimator)

        if estimator in {'random_mask', 'magnitude'}:
            self.phi_r = self._importance(retain_loader, estimator)
            self.phi_f = (
                self._importance(forget_loader, estimator)
                if forget_loader is not None
                else self._zeros_like(self.phi_r)
            )
        elif estimator in {'fisher_retain', 'fisher_forget', 'dual_fisher'}:
            retain_scores = self._importance_preserving_rng(retain_loader, 'fisher')
            forget_fisher_loader = (
                self._per_instance_loader(forget_loader)
                if self.per_instance_forget_fisher and forget_loader is not None
                else forget_loader
            )
            forget_scores = (
                self._importance_preserving_rng(forget_fisher_loader, 'fisher')
                if forget_loader is not None
                else self._zeros_like(retain_scores)
            )
            self.phi_r = (
                self._zeros_like(retain_scores) if estimator == 'fisher_forget' else retain_scores
            )
            self.phi_f = (
                self._zeros_like(forget_scores) if estimator == 'fisher_retain' else forget_scores
            )
        else:
            interaction = estimator in {'ssv_retain', 'ssv_forget', 'dual_ssv'}
            family = 'ssv_interaction' if interaction else 'ssv'
            retain_scores = self._importance_preserving_rng(retain_loader, family)
            forget_scores = (
                self._importance_preserving_rng(forget_loader, family)
                if forget_loader is not None
                else self._zeros_like(retain_scores)
            )
            self.phi_r = (
                self._zeros_like(retain_scores) if estimator == 'ssv_forget' else retain_scores
            )
            self.phi_f = (
                self._zeros_like(forget_scores) if estimator == 'ssv_retain' else forget_scores
            )
        self.resolved_tau_r = self._global_quantile(self.phi_r, self.retain_quantile)
        self.resolved_tau_f = self._global_quantile(self.phi_f, self.forget_quantile)

        for name in self.phi_r:
            retain = self.phi_r[name] > self.resolved_tau_r
            forget = self.phi_f[name] > self.resolved_tau_f
            self.zone_masks['A'][name] = retain & ~forget
            self.zone_masks['B'][name] = forget & ~retain
            self.zone_masks['C'][name] = retain & forget
            self.zone_masks['D'][name] = ~retain & ~forget

    @torch.no_grad()
    def reconstruction_diagnostics(self) -> Dict[str, float]:
        """Prove that the masks form an exact parameter partition."""
        return self.zone_reconstruction_diagnostics()

    @torch.no_grad()
    def _zero_zone_b(self):
        for name, parameter in self.backbone.named_parameters():
            if parameter.requires_grad:
                parameter.masked_fill_(self.zone_masks['B'][name], 0.0)

    @torch.no_grad()
    def _granular_zone_c(self):
        tau_r = max(self.resolved_tau_r, 1e-30)
        tau_f = max(self.resolved_tau_f, 1e-30)
        omega = max(tau_f / tau_r, tau_r / tau_f)
        for name, parameter in self.backbone.named_parameters():
            if not parameter.requires_grad:
                continue
            mask = self.zone_masks['C'][name]
            if not mask.any():
                continue
            retain = self.phi_r[name]
            forget = self.phi_f[name]
            forget_dominant = mask & forget.ge(omega * retain)
            retain_dominant = mask & retain.ge(omega * forget)
            balanced = mask & ~forget_dominant & ~retain_dominant
            rho = torch.maximum(retain, forget) / (retain + forget).clamp_min(1e-30)
            parameter.masked_fill_(forget_dominant, 0.0)
            parameter.copy_(torch.where(balanced, rho * parameter, parameter))

    def _repair_teacher_logits(self, teacher, inputs):
        return teacher(inputs)

    def _masked_distill(self, loader, teacher, zone, include_ce=True):
        if self.distill_epochs <= 0:
            return
        zones = (zone,) if isinstance(zone, str) else tuple(zone)
        if not zones or any(value not in 'ABCD' for value in zones):
            raise ValueError('repair zones must be drawn from A, B, C, and D')
        teacher.eval()
        self.backbone.eval()  # freeze BatchNorm buffers as well as non-zone weights
        optimizer = torch.optim.SGD(self.backbone.parameters(), lr=self.distill_lr)
        for _ in range(self.distill_epochs):
            for inputs, targets in loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                with torch.no_grad():
                    teacher_logits = self._repair_teacher_logits(teacher, inputs)
                optimizer.zero_grad(set_to_none=True)
                student_logits = self.backbone(inputs)
                loss = kd_kl(student_logits, teacher_logits, self.temperature)
                if include_ce and self.ce_weight:
                    loss = loss + self.ce_weight * F.cross_entropy(student_logits, targets)
                loss.backward()
                for name, parameter in self.backbone.named_parameters():
                    if parameter.grad is not None:
                        mask = torch.zeros_like(parameter, dtype=torch.bool)
                        for repair_zone in zones:
                            mask |= self.zone_masks[repair_zone][name]
                        parameter.grad.mul_(mask)
                optimizer.step()
        self.backbone.zero_grad(set_to_none=True)

    @torch.no_grad()
    def _retain_objectives(self, loader, teacher) -> Dict[str, float]:
        teacher.eval()
        self.backbone.eval()
        kl_total = 0.0
        ce_total = 0.0
        samples = 0
        for inputs, targets in loader:
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            batch = inputs.size(0)
            student_logits = self.backbone(inputs)
            kl_total += float(kd_kl(student_logits, teacher(inputs), self.temperature)) * batch
            ce_total += float(F.cross_entropy(student_logits, targets)) * batch
            samples += batch
        return {
            'kl': kl_total / max(samples, 1),
            'ce': ce_total / max(samples, 1),
        }

    @staticmethod
    def _rng_state(loader):
        """Capture every RNG source touched by a shuffled worker loader."""
        loader_generator = getattr(loader, 'generator', None)
        sampler_generator = getattr(getattr(loader, 'sampler', None), 'generator', None)
        return {
            'python': random.getstate(),
            'numpy': np.random.get_state(),
            'torch': torch.get_rng_state(),
            'cuda': (torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None),
            'loader': (loader_generator.get_state() if loader_generator is not None else None),
            'sampler': (sampler_generator.get_state() if sampler_generator is not None else None),
        }

    @staticmethod
    def _restore_rng_state(loader, state):
        random.setstate(state['python'])
        np.random.set_state(state['numpy'])
        torch.set_rng_state(state['torch'])
        if state['cuda'] is not None:
            torch.cuda.set_rng_state_all(state['cuda'])
        loader_generator = getattr(loader, 'generator', None)
        if loader_generator is not None and state['loader'] is not None:
            loader_generator.set_state(state['loader'])
        sampler_generator = getattr(getattr(loader, 'sampler', None), 'generator', None)
        if sampler_generator is not None and state['sampler'] is not None:
            sampler_generator.set_state(state['sampler'])

    @torch.no_grad()
    def _zone_delta_summary(self, before) -> Dict[str, float]:
        result = {}
        for zone in 'ABCD':
            delta_sq = 0.0
            count = 0
            for name, parameter in self.backbone.named_parameters():
                if not parameter.requires_grad:
                    continue
                mask = self.zone_masks[zone][name]
                delta_sq += float(((parameter - before[name]) * mask).square().sum())
                count += int(mask.sum())
            result[f'{zone}_delta_l2'] = delta_sq**0.5
            result[f'{zone}_count'] = count
        return result

    def unlearn(
        self,
        forget_loader,
        retain_loader,
        eta_c: float = 0.01,
        n_c: int = 5,
        delta_c: float = 0.1,
    ):
        joint_teacher = deepcopy(self.backbone).to(self.device).eval()
        for parameter in joint_teacher.parameters():
            parameter.requires_grad_(False)

        self.identify_zones(retain_loader, forget_loader)
        before = {
            name: parameter.detach().clone()
            for name, parameter in self.backbone.named_parameters()
            if parameter.requires_grad
        }
        reconstruction = self.reconstruction_diagnostics()

        self._zero_zone_b()
        if self.zone_c_strategy == 'granular':
            self._granular_zone_c()
        elif self.zone_c_strategy == 'orthogonal':
            self._orthogonal_conflict_resolution(
                retain_loader, forget_loader, nn.CrossEntropyLoss(), eta_c, n_c, delta_c
            )
        else:
            # SCRUB-style retain-function repair restricted to the shared
            # parameters.  The dynamic teacher hook lets ContextMemory/SelfDistillation mask the
            # forgotten class, while include_ce=False makes this a pure KL
            # ablation against orthogonal forget ascent.
            self._masked_distill(retain_loader, joint_teacher, 'C', include_ce=False)

        pre_expansion_delta = self._zone_delta_summary(before)
        base_budget = sum(parameter.numel() for parameter in self.backbone.parameters())
        if self.EXPANSION_ZONE is None:
            expanded = self.backbone
        else:
            expanded = ZoneFixedParameterExpansion(
                self.backbone, self.zone_masks, expansion_factor=2, split_zone=self.EXPANSION_ZONE
            ).to(self.device)
            self.backbone = expanded
            self.zone_masks = expanded.translated_zone_masks
        self.accumulated_mask = {
            name: torch.zeros_like(parameter, dtype=torch.bool)
            for name, parameter in self.backbone.named_parameters()
            if parameter.requires_grad
        }
        self.released_c_masks = {
            name: torch.zeros_like(parameter, dtype=torch.bool)
            for name, parameter in self.backbone.named_parameters()
            if parameter.requires_grad
        }
        self.momentum_buffer = {
            name: torch.zeros_like(parameter)
            for name, parameter in self.backbone.named_parameters()
            if parameter.requires_grad
        }

        # SCRUB-style retain minimization is applied after the destructive B/C
        # operations and neuron expansion, when its KL gradient is non-zero.
        # Measure on identical minibatches/augmentations without changing the
        # RNG stream seen by distillation.  This keeps instrumentation from
        # changing the selected method's result.
        pre_distill_rng = self._rng_state(retain_loader)
        if str(self.device).startswith('cuda'):
            torch.cuda.synchronize(torch.device(self.device))
        diagnostic_started = time.perf_counter()
        objectives_before = self._retain_objectives(retain_loader, joint_teacher)
        if str(self.device).startswith('cuda'):
            torch.cuda.synchronize(torch.device(self.device))
        self.diagnostic_time = time.perf_counter() - diagnostic_started
        self._restore_rng_state(retain_loader, pre_distill_rng)
        expanded_before_distill = {
            name: parameter.detach().clone()
            for name, parameter in self.backbone.named_parameters()
            if parameter.requires_grad
        }
        self._masked_distill(retain_loader, joint_teacher, self.REPAIR_ZONES)
        post_distill_rng = self._rng_state(retain_loader)
        self._restore_rng_state(retain_loader, pre_distill_rng)
        if str(self.device).startswith('cuda'):
            torch.cuda.synchronize(torch.device(self.device))
        diagnostic_started = time.perf_counter()
        objectives_after = self._retain_objectives(retain_loader, joint_teacher)
        if str(self.device).startswith('cuda'):
            torch.cuda.synchronize(torch.device(self.device))
        self.diagnostic_time += time.perf_counter() - diagnostic_started
        self._restore_rng_state(retain_loader, post_distill_rng)
        zone_a_distill_delta_sq = 0.0
        for name, parameter in self.backbone.named_parameters():
            if parameter.requires_grad:
                delta = (parameter - expanded_before_distill[name]) * self.zone_masks['A'][name]
                zone_a_distill_delta_sq += float(delta.square().sum())
        self._update_accumulated_mask()
        expanded_reconstruction = self.reconstruction_diagnostics()
        self.diagnostics = {
            'importance': self.importance_kind,
            'zone_a': (
                'retain KL to frozen joint teacher'
                if not self.ce_weight
                else 'retain KL to frozen joint teacher + labeled CE'
            ),
            'zone_a_ce_weight': self.ce_weight,
            'repair_zones': list(self.REPAIR_ZONES),
            'zone_b': 'hard_zero',
            'zone_c': self.zone_c_strategy,
            'retain_quantile': self.retain_quantile,
            'forget_quantile': self.forget_quantile,
            'tau_r': self.resolved_tau_r,
            'tau_f': self.resolved_tau_f,
            'reconstruction': reconstruction,
            'expanded_reconstruction': expanded_reconstruction,
            'zones_pct': self.get_zone_statistics()['overall'],
            'expansion_split_zone': self.EXPANSION_ZONE,
            'zone_d_policy': (
                'zone-D-aware fixed-parameter neuron expansion'
                if self.EXPANSION_ZONE == 'D'
                else 'contextual fast-weight memory'
            ),
            'base_parameter_budget': base_budget,
            'expanded_effective_parameter_budget': (
                expanded.effective_nonzero_budget()
                if hasattr(expanded, 'effective_nonzero_budget')
                else sum(parameter.numel() for parameter in expanded.parameters())
            ),
            'stored_parameter_count_after_expansion': sum(
                parameter.numel() for parameter in expanded.parameters()
            ),
            'retain_teacher_kl_before_zone_a': objectives_before['kl'],
            'retain_teacher_kl_after_zone_a': objectives_after['kl'],
            'retain_ce_before_zone_a': objectives_before['ce'],
            'retain_ce_after_zone_a': objectives_after['ce'],
            'diagnostic_time_excluded': self.diagnostic_time,
            'zone_a_distill_delta_l2': zone_a_distill_delta_sq**0.5,
            **{f'pre_expansion_{key}': value for key, value in pre_expansion_delta.items()},
        }
        return self.backbone
