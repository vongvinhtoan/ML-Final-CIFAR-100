from torch import nn
from .resnet import Resnet
from .effnetv2_l import EfficientNetV2_L
from .effnetv2_m import EfficientNetV2_M
from .effnet_b7 import EfficientNet_B7
from typing import Type


models: dict[str, Type[nn.Module]] = {
    'resnet': Resnet,
    'effnetv2_l': EfficientNetV2_L,
    'effnetv2_m': EfficientNetV2_M,
    'effnet_b7': EfficientNet_B7
}