import torch
from typing import Callable

optimizers: dict[str, Callable[..., torch.optim.Optimizer]] = {
    'adam': torch.optim.Adam,
    'sgd': torch.optim.SGD
}