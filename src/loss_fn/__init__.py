from .cross_entropy import loss_fn as cross_entropy_fn
from typing import Callable
from torch import nn


loss_fns: dict[str, nn.Module] = {
    "cross_entropy": cross_entropy_fn
}
