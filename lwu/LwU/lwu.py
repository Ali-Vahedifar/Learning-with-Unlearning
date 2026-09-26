"""LwU: SelfDistillation slow unlearning plus causal persistent neural memory.

The privileged/repair teachers are training-time objects only. Inference uses
the student and a fixed-size associative fast-weight matrix. Each batch is
predicted before it is written, so it can help only future batches. The memory
stores no images or feature vectors: observations are immediately compressed
into its weights and discarded.
"""

from contextlib import contextmanager

import torch
import torch.nn.functional as F

from LwU.self_distillation import SelfDistillation


class LwU(SelfDistillation):
    """Teacher-free causal test-time memory on top of SelfDistillation."""

    def __init__(
        self,
        *args,
        memory_lr: float = 0.05,
        memory_scale: float = 0.5,
        memory_decay: float = 0.995,
        memory_momentum: float = 0.9,
        memory_confidence: float = 0.8,
        memory_max_norm: float = 5.0,
        persist_memory_across_streams: bool = False,
        **kwargs,
    ):
        if memory_lr < 0.0:
            raise ValueError('memory_lr must be non-negative')
        if memory_scale < 0.0:
            raise ValueError('memory_scale must be non-negative')
        if not 0.0 <= memory_decay <= 1.0:
            raise ValueError('memory_decay must be in [0, 1]')
        if not 0.0 <= memory_momentum < 1.0:
            raise ValueError('memory_momentum must be in [0, 1)')
        if not 0.0 <= memory_confidence <= 1.0:
            raise ValueError('memory_confidence must be in [0, 1]')
        if memory_max_norm <= 0.0:
            raise ValueError('memory_max_norm must be positive')
        self.memory_lr = float(memory_lr)
        self.memory_scale = float(memory_scale)
        self.memory_decay = float(memory_decay)
        self.memory_momentum = float(memory_momentum)
        self.memory_confidence = float(memory_confidence)
        self.memory_max_norm = float(memory_max_norm)
        self.persist_memory_across_streams = bool(persist_memory_across_streams)
        self._memory_weight = None
        self._memory_velocity = None
        super().__init__(*args, **kwargs)
        self.reset_memory_diagnostics()

    def reset_memory_diagnostics(self):
        self._memory_diagnostics = {
            'batches_predicted': 0,
            'examples_predicted': 0,
            'batches_written': 0,
            'examples_written': 0,
            'prediction_flips_from_memory': 0,
            'memory_logit_l2_sum': 0.0,
            'memory_update_l2_sum': 0.0,
            'memory_update_l2_max': 0.0,
            'memory_weight_l2_max': 0.0,
            'memory_parameters_peak': 0,
        }

    def get_memory_diagnostics(self):
        stats = dict(self._memory_diagnostics)
        examples = max(stats['examples_predicted'], 1)
        writes = max(stats['batches_written'], 1)
        stats.update(
            {
                'write_rate': stats['examples_written'] / examples,
                'prediction_flip_rate': (stats['prediction_flips_from_memory'] / examples),
                'mean_memory_logit_l2_per_example': (stats.pop('memory_logit_l2_sum') / examples),
                'mean_memory_update_l2_per_written_batch': (
                    stats.pop('memory_update_l2_sum') / writes
                ),
                'memory_weight_l2_current': (
                    0.0
                    if self._memory_weight is None
                    else float(self._memory_weight.float().norm())
                ),
                'memory_parameters_current': (
                    0 if self._memory_weight is None else int(self._memory_weight.numel())
                ),
            }
        )
        return stats

    def reset_fast_memory(self):
        """Erase compressed online state without changing slow parameters."""
        self._memory_weight = None
        self._memory_velocity = None

    def set_persistent_memory(self, enabled: bool, reset: bool = True):
        """Carry memory across loaders/tasks in a chronological online run."""
        self.persist_memory_across_streams = bool(enabled)
        if reset:
            self.reset_fast_memory()

    def _ensure_fast_memory(self, feature_dim, device):
        expected = (self.num_classes, feature_dim)
        if (
            self._memory_weight is None
            or tuple(self._memory_weight.shape) != expected
            or self._memory_weight.device != device
        ):
            self._memory_weight = torch.zeros(expected, device=device, dtype=torch.float32)
            self._memory_velocity = torch.zeros_like(self._memory_weight)
            self._memory_diagnostics['memory_parameters_peak'] = max(
                self._memory_diagnostics['memory_parameters_peak'], int(self._memory_weight.numel())
            )

    def _active_class_mask(self, device):
        active = torch.ones(self.num_classes, dtype=torch.bool, device=device)
        allowed = self._context_classes
        if allowed is not None:
            active.zero_()
            active[list(allowed)] = True
        if self.forgotten_classes:
            forgotten = torch.as_tensor(
                sorted(self.forgotten_classes), dtype=torch.long, device=device
            )
            active[forgotten] = False
        return active

    @torch.no_grad()
    def _write_fast_memory(self, keys, base_logits):
        if self.memory_lr == 0.0:
            return
        active = self._active_class_mask(base_logits.device)
        if not bool(active.any()):
            return

        target_logits = base_logits.float().clone()
        target_logits[:, ~active] = torch.finfo(target_logits.dtype).min
        probabilities = F.softmax(target_logits, dim=1)
        confidence, prediction = probabilities.max(dim=1)
        selected = confidence.ge(self.memory_confidence)
        if not bool(selected.any()):
            return

        write_keys = keys[selected]
        prediction = prediction[selected]
        active_count = int(active.sum())
        target = torch.zeros(
            write_keys.size(0), self.num_classes, device=write_keys.device, dtype=torch.float32
        )
        target[:, active] = -1.0 / active_count
        target.scatter_(1, prediction.unsqueeze(1), 1.0 - 1.0 / active_count)

        memory_prediction = write_keys @ self._memory_weight.t()
        error = memory_prediction - target
        gradient = error.t() @ write_keys / write_keys.size(0)
        self._memory_velocity.mul_(self.memory_momentum).add_(gradient)
        previous = self._memory_weight.clone()
        self._memory_weight.mul_(self.memory_decay).add_(
            self._memory_velocity, alpha=-self.memory_lr
        )

        row_norm = self._memory_weight.norm(dim=1, keepdim=True)
        scale = (self.memory_max_norm / row_norm.clamp_min(self.memory_max_norm)).clamp_max(1.0)
        self._memory_weight.mul_(scale)
        if self.forgotten_classes:
            forgotten = torch.as_tensor(
                sorted(self.forgotten_classes), dtype=torch.long, device=self._memory_weight.device
            )
            self._memory_weight.index_fill_(0, forgotten, 0.0)
            self._memory_velocity.index_fill_(0, forgotten, 0.0)

        update_l2 = float((self._memory_weight - previous).norm())
        stats = self._memory_diagnostics
        stats['batches_written'] += 1
        stats['examples_written'] += int(write_keys.size(0))
        stats['memory_update_l2_sum'] += update_l2
        stats['memory_update_l2_max'] = max(stats['memory_update_l2_max'], update_l2)
        stats['memory_weight_l2_max'] = max(
            stats['memory_weight_l2_max'], float(self._memory_weight.float().norm())
        )

    def test_time_update(self, inputs):
        """Predict from M_(t-1), then compress the batch into M_t."""
        self.backbone.eval()
        with torch.no_grad():
            features = self.backbone.get_features(inputs).detach()
            if features.ndim != 2:
                features = features.flatten(1)
            keys = F.normalize(features.float(), dim=1)
            base_logits = self.backbone.forward_from_features(features)
            self._ensure_fast_memory(keys.size(1), keys.device)
            memory_logits = keys @ self._memory_weight.t()
            output = base_logits + self.memory_scale * memory_logits.to(base_logits.dtype)

            stats = self._memory_diagnostics
            stats['batches_predicted'] += 1
            stats['examples_predicted'] += int(inputs.size(0))
            stats['memory_logit_l2_sum'] += float(memory_logits.norm(dim=1).sum())
            stats['prediction_flips_from_memory'] += int(
                base_logits.argmax(1).ne(output.argmax(1)).sum()
            )

            # ``output`` was computed before this write, so the current batch
            # can influence future predictions only.
            self._write_fast_memory(keys, base_logits)
            return output

    def unlearn(self, forget_loader, retain_loader, **kwargs):
        result = super().unlearn(forget_loader, retain_loader, **kwargs)
        # SelfDistillation temporarily needs its snapshot for repair. LwU discards it
        # before inference and uses only student + compressed fast memory.
        self.__dict__['_context_teacher'] = None
        self._reset_context()
        self.reset_fast_memory()
        self.reset_memory_diagnostics()
        self.diagnostics.update(
            {
                'inference_teacher': None,
                'test_time_memory': 'causal fixed-size associative fast weights',
                'memory_stores_examples': False,
                'memory_lr': self.memory_lr,
                'memory_scale': self.memory_scale,
                'memory_decay': self.memory_decay,
                'memory_momentum': self.memory_momentum,
                'memory_confidence': self.memory_confidence,
                'memory_max_norm': self.memory_max_norm,
                'persist_memory_across_streams': self.persist_memory_across_streams,
            }
        )
        return result

    def consolidate_task(self, task_loader, forget_loader=None, criterion=None):
        super().consolidate_task(task_loader, forget_loader=forget_loader, criterion=criterion)
        self.__dict__['_context_teacher'] = None
        self._reset_context()

    def restore_original_params(self):
        super().restore_original_params()
        if not self.persist_memory_across_streams:
            self.reset_fast_memory()

    @contextmanager
    def test_stream(self):
        if not self.persist_memory_across_streams:
            self.reset_fast_memory()
        try:
            yield self
        finally:
            if not self.persist_memory_across_streams:
                self.reset_fast_memory()
