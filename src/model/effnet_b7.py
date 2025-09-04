import torch.nn as nn
import torch.nn.init as init
from torchvision import models

class EfficientNet_B7(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()

        freeze_pretrained=kwargs['freeze_pretrained']
        weight_init=kwargs['weight_init']

        # Load EfficientNet-V2-M with pretrained weights
        self.model = models.efficientnet_b7(
            weights=models.EfficientNet_B7_Weights.DEFAULT
        )

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
