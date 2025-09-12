from torch import nn
from .resnet import Resnet
from .effnetv2_l import EfficientNetV2_L
from .effnetv2_m import EfficientNetV2_M
from .effnet_b7 import EfficientNet_B7
from .mlp import MLP
from .cnn import SmallCNN
from typing import Type, Any


models: dict[str, Type[nn.Module]] = {
    'resnet': Resnet,
    'effnetv2_l': EfficientNetV2_L,
    'effnetv2_m': EfficientNetV2_M,
    'effnet_b7': EfficientNet_B7,
    'mlp': MLP,
    'cnn': SmallCNN,
}


def model_size(model: nn.Module) -> dict[str, Any]:
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    return {
        "num_params": f"{total_params:,}",
        "trainable_params": f"{trainable_params:,}"
    }