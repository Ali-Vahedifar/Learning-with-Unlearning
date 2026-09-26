"""LwU run as a continual learner on the GTEP loop.

The learner is LwU's zone stage, ``ZoneDecomposition`` in ``../../lwu/LwU/zones.py``
-- the same code the machine-unlearning benchmark builds LwU on. The later LwU
stages are unlearning-time steps (zone-B reset, repair, self-distillation,
causal memory) and have no forget set to act on here. This adapter only
connects it to the shared loop in cl_base, so LwU trains with the same
optimiser, early stopping and multi-head layout as every other method:

* CL-only mode: there is no forget set, so phi_f = 0 and only Zone A
  (|SSV| > tau_r) forms. tau_f cannot change a result and stays fixed.
* Zone A of every finished task accumulates and receives no gradient.
* After each task its SSV is computed on that task's training data while it is
  still available (``consolidate_task``).
* Evaluation applies the test-time update inside a per-loader stream and
  restores the stored weights afterwards. Its entropy is over the classes in
  the output space (the seen classes in Class-IL), not over heads of tasks not
  yet trained.
"""

import contextlib
import sys
from pathlib import Path

from cl_base import ContinualMethod

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'lwu'))
from LwU.zones import ZoneDecomposition  # noqa: E402


class LwUMethod(ContinualMethod):
    name = 'lwu'

    def __init__(
        self,
        model,
        device,
        *args,
        tau_r=1e-6,
        tau_f=0.35,
        momentum_decay=0.95,
        adaptation_rate=0.01,
        surprise_threshold=0.05,
        stabilization_decay=0.3,
        test_time_adapt=True,
        **kw
    ):
        # LwU keys its masks by parameter name when it is built, so every head
        # must exist now; seen_upto still limits the Class-IL output to seen tasks.
        for t in range(model.num_tasks):
            model.ensure_head(t)
        model.seen_upto = -1
        super().__init__(model, device, *args, **kw)
        self.lwu = ZoneDecomposition(
            self.model,
            model.num_tasks * model.classes_per_task,
            tau_r=tau_r,
            tau_f=tau_f,
            momentum_decay=momentum_decay,
            adaptation_rate=adaptation_rate,
            surprise_threshold=surprise_threshold,
            stabilization_decay=stabilization_decay,
            threshold_mode='absolute',
            device=device,
        )
        self.test_time_adapt = test_time_adapt
        # Live view of LwU's masks, SSV maps and TTU buffers, so the cost ledger
        # counts them as resident state.
        self.lwu_state = vars(self.lwu)

    def _head(self, out, task_id):
        """Task t's logits from the concatenated heads (Task-IL); unchanged in Class-IL."""
        if self.scenario != 'task_il' or task_id is None:
            return out
        c = self.model.classes_per_task
        return out[:, task_id * c : (task_id + 1) * c]

    def after_backward(self, task_id):
        for name, p in self.model.named_parameters():
            if p.grad is not None:
                p.grad.masked_fill_(self.lwu.accumulated_mask[name], 0.0)

    def after_task(self, task_id, train_loader, val_loader):
        self.lwu.consolidate_task(
            train_loader, criterion=lambda out, y: self.criterion(self._head(out, task_id), y)
        )
        stats = self.lwu.get_zone_statistics()
        return dict(zones=stats['overall'], accumulated=stats['accumulated'])

    def prediction_stream(self, task_id=None):
        return self.lwu.test_stream() if self.test_time_adapt else contextlib.nullcontext()

    def evaluate(self, loader, task_id=None):
        with self.prediction_stream(task_id):
            return super().evaluate(loader, task_id)

    def predict(self, x, task_id):
        if not self.test_time_adapt:
            return super().predict(x, task_id)
        return self._head(self.lwu.test_time_update(x), task_id)
