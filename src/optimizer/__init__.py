from typing import Callable
from .base_optimizer import BaseOptimizer
from .adam import Adam
from .sam import SAM
from .sgd import SGD

optimizers: dict[str, Callable[..., BaseOptimizer]] = {
    'adam': Adam,
    'sgd': SGD,
    'sam': SAM
}