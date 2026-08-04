"""Evaluation package for LwU framework."""

from lwu.evaluation.metrics import (
    AccuracyMatrix,
    evaluate_task,
    evaluate_all_tasks,
    MembershipInferenceAttack,
    compute_kl_divergence,
    UnlearningEvaluator,
    measure_execution_time,
    ContinualLearningEvaluator
)

__all__ = [
    'AccuracyMatrix',
    'evaluate_task',
    'evaluate_all_tasks',
    'MembershipInferenceAttack',
    'compute_kl_divergence',
    'UnlearningEvaluator',
    'measure_execution_time',
    'ContinualLearningEvaluator'
]
