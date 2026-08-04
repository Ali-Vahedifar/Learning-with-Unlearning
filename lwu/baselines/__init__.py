"""Baselines package for LwU framework."""

from lwu.baselines.continual_methods import (
    EWC,
    SI,
    LwF,
    SGDBaseline,
    JointTraining,
    get_continual_method
)

from lwu.baselines.unlearning_methods import (
    SSD,
    BadTeacher,
    Amnesiac,
    UNSIR,
    retrain_from_scratch,
    get_unlearning_method
)

__all__ = [
    # Continual Learning
    'EWC',
    'SI',
    'LwF',
    'SGDBaseline',
    'JointTraining',
    'get_continual_method',
    
    # Machine Unlearning
    'SSD',
    'BadTeacher',
    'Amnesiac',
    'UNSIR',
    'retrain_from_scratch',
    'get_unlearning_method'
]
