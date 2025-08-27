from .base_optimizer import BaseOptimizer
import torch.optim as optim


class SGD(BaseOptimizer):
    def __init__(self, params, **kwargs):
        super().__init__(params, base_optimizer=optim.SGD, **kwargs)
