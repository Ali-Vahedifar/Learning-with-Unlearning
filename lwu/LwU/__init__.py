"""LwU (Learning with Unlearning).

One method, built in stages that are implementation detail rather than
selectable variants:

    zones.py                four-zone parameter decomposition (A safe-retain,
                            B pure-forget, C conflict, D plastic) + test-time
                            update hook
    teacher_repair.py       teacher/student repair, B -> C -> A ordering
    dominance.py            scale-free dominance zoning + CE-augmented A repair
    context_memory.py       He-reinitialised B, joint A/B repair, Zone-D memory
    self_distillation.py    exemplar-free context-privileged Zone D, Zone-C SDFT
    lwu.py                  LwU: the method, adding a causal test-time
                            fast-weight memory over the frozen slow zones

Only `LwU` is exported; the benchmark harness registers it as `lwu`.
"""

from LwU.lwu import LwU
from LwU.zones import ZoneDecomposition

__all__ = ['LwU', 'ZoneDecomposition']
