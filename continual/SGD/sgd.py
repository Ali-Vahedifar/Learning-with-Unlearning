"""sgd -- implementation moved here from baselines_regularization.py.

Imported by the method registry in baselines.py; baselines_regularization.py re-exports these
names so existing imports keep working.
"""

"""
Regularisation-based baselines: SGD (lower bound), EWC, SI, LwF.
"""


from cl_base import ContinualMethod


class SGDBaseline(ContinualMethod):
    """Plain sequential fine-tuning -- the lower bound."""

    name = 'sgd'
