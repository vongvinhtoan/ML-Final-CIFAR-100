import torch.nn as nn
import torch.nn.init as init
from torchvision import models

class EfficientNetV2_M(nn.Module):
    def __init__(self, freeze_pretrained=False, weight_init=False):
        super().__init__()

        # Load EfficientNet-V2-M with pretrained weights
        self.model = models.efficientnet_v2_m(
            weights=models.EfficientNet_V2_M_Weights.DEFAULT
        )

        # Change first conv to fit CIFAR (32x32)
        # Original: Conv2d(3, 24, kernel_size=3, stride=2, padding=1)
        first_block = self.model.features[0]
        if isinstance(first_block, nn.Sequential):
            conv = nn.Conv2d(3, 24, kernel_size=3, stride=1, padding=1, bias=False)
            if weight_init:
                init.kaiming_normal_(conv.weight, mode="fan_out", nonlinearity="relu")
            first_block[0] = conv

        # Modify classifier for CIFAR-100
        in_features: int = self.model.classifier[-1].in_features  # type: ignore
        fc = nn.Linear(in_features, 100)
        if weight_init:
            init.xavier_uniform_(fc.weight)
            if fc.bias is not None:
                init.zeros_(fc.bias)
        self.model.classifier[-1] = fc

        # Optionally freeze all pretrained layers except first conv and classifier
        if freeze_pretrained:
            for name, param in self.model.named_parameters():
                if not name.startswith("features.0.0") and not name.startswith("classifier"):
                    param.requires_grad = False

    def forward(self, x):
        return self.model(x)
