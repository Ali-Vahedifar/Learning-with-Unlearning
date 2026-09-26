"""Machine-unlearning baselines, one paper/method per subpackage."""

from .Baseline import Baseline
from .Retrain import Retrain
from .FineTune import FineTune
from .BadTeacher import BadTeacher
from .Amnesiac import Amnesiac
from .UNSIR import UNSIR
from .SSD import SSD
from .UniCLUN import UniCLUN
from .SCRUB import SCRUB
from .SalUn import SalUn
from .RandomLabel import RandomLabel
from .GradientAscent import GradientAscent
from .L1Sparse import L1Sparse
from .BoundaryShrink import BoundaryShrink
from .BoundaryExpand import BoundaryExpand
from .InfluenceUnlearning import InfluenceUnlearning

METHODS = {
    'baseline': Baseline,
    'retrain': Retrain,
    'finetune': FineTune,
    'badteacher': BadTeacher,
    'amnesiac': Amnesiac,
    'unsir': UNSIR,
    'ssd': SSD,
    'uniclun': UniCLUN,
    'scrub': SCRUB,
    'salun': SalUn,
    'rl': RandomLabel,
    'ga': GradientAscent,
    'l1sparse': L1Sparse,
    'bs': BoundaryShrink,
    'be': BoundaryExpand,
    'iu': InfluenceUnlearning,
}

__all__ = [
    'Baseline',
    'Retrain',
    'FineTune',
    'BadTeacher',
    'Amnesiac',
    'UNSIR',
    'SSD',
    'UniCLUN',
    'SCRUB',
    'SalUn',
    'RandomLabel',
    'GradientAscent',
    'L1Sparse',
    'BoundaryShrink',
    'BoundaryExpand',
    'InfluenceUnlearning',
    'METHODS',
]
