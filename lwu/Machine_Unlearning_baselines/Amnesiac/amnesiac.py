from typing import Dict, Iterable, List, Optional, Set, Tuple

import torch


class Amnesiac:
    def __init__(self, model, device='cuda', track_per_batch: bool = False):
        self.model = model
        self.device = device
        self.track_per_batch = track_per_batch
        self._before = None
        self.update_sum: Optional[Dict[str, torch.Tensor]] = None
        self.num_recorded = 0
        self.update_ledger: List[Dict[str, torch.Tensor]] = []
        self.ledger_ids: List[Set[int]] = []

    def before_step(self):
        self._before = {
            name: parameter.detach().clone() for name, parameter in self.model.named_parameters()
        }

    def after_step(
        self, sample_ids: Optional[Iterable[int]] = None, contains_sensitive: bool = False
    ):
        if self._before is None:
            raise RuntimeError('before_step() must be called before after_step()')
        ids = set(int(i) for i in sample_ids) if sample_ids is not None else set()
        if contains_sensitive or ids:
            delta = {
                name: parameter.detach() - self._before[name]
                for name, parameter in self.model.named_parameters()
            }
            if self.update_sum is None:
                self.update_sum = {name: value.clone() for name, value in delta.items()}
            else:
                for name, value in delta.items():
                    self.update_sum[name].add_(value)
            self.num_recorded += 1
            if self.track_per_batch:
                self.update_ledger.append({name: value.clone() for name, value in delta.items()})
                self.ledger_ids.append(ids)
        self._before = None

    def checkpoint(self) -> Tuple:
        sums = (
            None
            if self.update_sum is None
            else {name: value.clone() for name, value in self.update_sum.items()}
        )
        return sums, self.num_recorded, len(self.update_ledger)

    def rollback(self, state: Tuple):
        sums, num_recorded, ledger_length = state
        self.update_sum = (
            None if sums is None else {name: value.clone() for name, value in sums.items()}
        )
        self.num_recorded = num_recorded
        self.update_ledger = self.update_ledger[:ledger_length]
        self.ledger_ids = self.ledger_ids[:ledger_length]

    @torch.no_grad()
    def unlearn(
        self, forget_loader=None, retain_loader=None, forget_ids: Optional[Iterable[int]] = None
    ):
        del forget_loader, retain_loader
        if self.num_recorded == 0:
            raise RuntimeError(
                'Amnesiac requires optimizer updates recorded during original training; '
                'post-hoc gradients are not equivalent.'
            )
        if forget_ids is None:
            for name, parameter in self.model.named_parameters():
                parameter.sub_(self.update_sum[name])
            return self.model

        if not self.track_per_batch:
            raise RuntimeError(
                'selective forget_ids needs the per-batch deltas; construct '
                'Amnesiac(..., track_per_batch=True) before training.'
            )
        requested = set(int(i) for i in forget_ids)
        removed = 0
        for update, ids in zip(self.update_ledger, self.ledger_ids):
            if ids.isdisjoint(requested):
                continue
            for name, parameter in self.model.named_parameters():
                parameter.sub_(update[name])
            removed += 1
        if removed == 0:
            raise ValueError('no recorded training batch contains a requested sample')
        return self.model
