import torch
from typing import Type

optimizers: dict[str, Type[torch.optim.Optimizer]] = {
    'adam': torch.optim.Adam,
    'sgd': torch.optim.SGD
}