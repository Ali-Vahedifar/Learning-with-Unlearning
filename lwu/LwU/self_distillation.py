"""SelfDistillation: exemplar-free context-privileged Zone-D fast adaptation.

The persistent A/B/C parameters remain slow state.  During an evaluation
stream, a bounded cache stores only detached penultimate features and frozen
teacher distributions from the current stream.  Similarity retrieval turns
that cache into a context-privileged teacher, and reverse KL updates only
unprotected Zone-D classifier parameters.  No input image or label is stored.
"""

from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F

from LwU.context_memory import ContextMemory


class SelfDistillation(ContextMemory):
    """ContextMemory with a continual-learning-compatible contextual Zone D.

    Two optional unlearning-side additions, ported from the continual-learning
    variant of this method:

    - ``sdft_lambda`` (self-distillation on the plastic zone): after the usual
      Zone A+B repair, run one more masked-distill pass restricted to Zone C,
      distilling from the frozen post-repair snapshot this class already
      builds for its context memory. ``sdft_lambda`` scales the distillation
      learning rate for that pass -- for a lone KL+CE objective this is
      exactly equivalent to scaling the loss itself.
    - ``weight_align``: after repair (and SDFT, if enabled), rescale each
      retained class's classifier-head row to the mean row norm those same
      classes had *before* unlearning touched the model. This is the
      unlearning-side analogue of continual learning's weight-align: it
      corrects head-scale drift the repair procedure introduces, rather than
      the CIL new-vs-old-class drift the original targets.
    """

    def __init__(self, *args, weight_align: bool = False, sdft_lambda: float = 0.0, **kwargs):
        self.weight_align = bool(weight_align)
        self.sdft_lambda = float(sdft_lambda)
        self._context_classes = None
        super().__init__(*args, **kwargs)

    def _find_head(self):
        head = None
        for module in self.backbone.modules():
            if isinstance(module, nn.Linear):
                head = module
        return head

    def unlearn(self, forget_loader, retain_loader, **kwargs):
        head = self._find_head()
        pre_norms = head.weight.detach().norm(dim=1).clone() if head is not None else None

        result = super().unlearn(forget_loader, retain_loader, **kwargs)

        if self.sdft_lambda > 0.0 and self.distill_epochs > 0:
            teacher = self.__dict__.get('_context_teacher')
            base_lr = self.distill_lr
            self.distill_lr = base_lr * self.sdft_lambda
            self._masked_distill(retain_loader, teacher, ('C',), include_ce=True)
            self.distill_lr = base_lr
            self.diagnostics['sdft_lambda'] = self.sdft_lambda
            self.diagnostics['sdft_zone'] = 'C'
            self._install_context_teacher()  # refresh: C moved after distill

        if self.weight_align and pre_norms is not None and head is not None:
            retained = [c for c in range(pre_norms.numel()) if c not in self.forgotten_classes]
            if retained:
                with torch.no_grad():
                    target = pre_norms[retained].mean()
                    current = head.weight.detach().norm(dim=1)
                    for c in retained:
                        if current[c] > 0:
                            head.weight[c].mul_(target / current[c])
            self.diagnostics['weight_align'] = True

        return result

    def _install_context_teacher(self):
        teacher = deepcopy(self.backbone).to(self.device).eval()
        for parameter in teacher.parameters():
            parameter.requires_grad_(False)
        self.__dict__['_context_teacher'] = teacher
        self._reset_context()

    def consolidate_task(self, task_loader, forget_loader=None, criterion=None):
        """Consolidate slow weights, then snapshot the contextual teacher."""
        super().consolidate_task(task_loader, forget_loader=forget_loader, criterion=criterion)
        self._install_context_teacher()

    def set_context_classes(self, classes):
        """Supply known Task-IL classes; ``None`` means Class-IL."""
        self._context_classes = None if classes is None else tuple(sorted(int(v) for v in classes))

    def _ensure_context_teacher(self):
        if self.__dict__.get('_context_teacher') is None:
            # Checkpoint restoration does not serialize a duplicate network.
            # Reconstructing it from the restored slow state is exact.
            self._install_context_teacher()

    def _teacher_values(self, inputs):
        self._ensure_context_teacher()
        teacher = self.__dict__['_context_teacher']
        with torch.no_grad():
            features = teacher.get_features(inputs).detach()
            logits = teacher.forward_from_features(features)

            allowed = self._context_classes
            if allowed is not None:
                active = torch.zeros(logits.size(1), dtype=torch.bool, device=logits.device)
                active[list(allowed)] = True
                logits.masked_fill_(~active.unsqueeze(0), torch.finfo(logits.dtype).min)
            if self.forgotten_classes:
                index = torch.as_tensor(sorted(self.forgotten_classes), device=logits.device)
                logits.index_fill_(1, index, torch.finfo(logits.dtype).min)
            values = F.softmax(logits, dim=1).detach()
        return features, values

    def _adapt_context_memory(self):
        if not self._context or self.context_lr == 0.0:
            return
        keys = torch.cat([entry[0] for entry in self._context])
        values = torch.cat([entry[1] for entry in self._context])
        targets = self._context_targets(keys, values).detach()

        weighted_batches = []
        for age, (context_keys, _) in enumerate(reversed(self._context)):
            weighted_batches.append((context_keys.size(0), self.context_decay**age))
        sample_weights = [
            torch.full((size,), weight, device=keys.device)
            for size, weight in reversed(weighted_batches)
        ]
        sample_weights = torch.cat(sample_weights)

        for _ in range(self.context_steps):
            with torch.enable_grad():
                self.backbone.zero_grad(set_to_none=True)
                logits = self.backbone.forward_from_features(keys)
                log_student = F.log_softmax(logits, dim=1)
                student = log_student.exp()
                reverse_kl = (student * (log_student - targets.log())).sum(dim=1)
                loss = (reverse_kl * sample_weights).sum() / sample_weights.sum()
                loss.backward()

                with torch.no_grad():
                    gradient_sq = 0.0
                    update_sq = 0.0
                    updated_parameters = 0
                    for name, parameter in self.backbone.named_parameters():
                        if parameter.grad is None:
                            continue
                        if self.context_unmasked:
                            # Cached features only give gradients to the head;
                            # this ablation updates every such parameter.
                            mask = torch.ones_like(parameter, dtype=torch.bool)
                        else:
                            mask = self.zone_masks['D'][name]
                            if name in self.accumulated_mask:
                                mask = mask & ~self.accumulated_mask[name]
                        if not mask.any():
                            continue
                        gradient = parameter.grad * mask
                        gradient_sq += float(gradient.float().square().sum())
                        momentum = self._context_fast_momentum[name]
                        momentum.mul_(self.context_momentum).add_(gradient)
                        anchor = self.original_params[name]
                        candidate = (
                            anchor
                            + self.context_decay * (parameter - anchor)
                            - self.context_lr * momentum
                        )
                        delta = torch.where(
                            mask, candidate - parameter, torch.zeros_like(parameter)
                        )
                        update_sq += float(delta.float().square().sum())
                        updated_parameters += int(mask.sum())
                        parameter.copy_(torch.where(mask, candidate, parameter))
                    stats = self._ttu_diagnostics
                    update_l2 = update_sq**0.5
                    stats['optimizer_steps'] += 1
                    stats['objective_sum'] += float(loss.detach())
                    stats['gradient_l2_sum'] += gradient_sq**0.5
                    stats['update_l2_sum'] += update_l2
                    stats['update_l2_max'] = max(stats['update_l2_max'], update_l2)
                    stats['updated_parameters_sum'] += updated_parameters
        self.backbone.zero_grad(set_to_none=True)

    def restore_original_params(self):
        super().restore_original_params()
        self._context_classes = None
