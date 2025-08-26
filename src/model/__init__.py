from torch import nn
from .resnet import Resnet
from .effnet_l2 import EfficientNetL2
from typing import Type


models: dict[str, Type[nn.Module]] = {
    'resnet': Resnet,
    'effnet_l2': EfficientNetL2
}