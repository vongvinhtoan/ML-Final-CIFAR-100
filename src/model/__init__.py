from torch import nn
from .resnet import Resnet
from typing import Type


models: dict[str, Type[nn.Module]] = {
    'resnet': Resnet
}