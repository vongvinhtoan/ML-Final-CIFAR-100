from typing import Callable, Type, Any
import torch
import torch.optim as optim
from torch.optim import Optimizer


class BaseOptimizer(Optimizer):
    def __init__(self, params, base_optimizer: Type[optim.Optimizer], **kwargs: Any):
        self.base_optimizer = base_optimizer(params, **kwargs)
        super().__init__(self.base_optimizer.param_groups, self.base_optimizer.defaults)

    def step(self, closure: Callable[[], tuple[torch.Tensor, torch.Tensor]]) -> tuple[torch.Tensor, torch.Tensor]:
        loss, output = closure()
        loss.backward()
        self.base_optimizer.step()
        return loss, output

    def zero_grad(self, set_to_none: bool = False):
        return self.base_optimizer.zero_grad(set_to_none=set_to_none)
