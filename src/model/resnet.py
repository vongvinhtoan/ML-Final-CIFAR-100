import torch.nn as nn
import torch.nn.init as init
from torchvision import models

class Resnet(nn.Module):
    def __init__(self, freeze_pretrained=False, weight_init=False):
        super().__init__()

        # Load pretrained ResNet101
        self.model = models.resnet101(weights=models.ResNet101_Weights.DEFAULT)

        # Modify classifier
        in_features = self.model.fc.in_features
        fc = nn.Linear(in_features, 100)
        if weight_init:
            init.xavier_uniform_(fc.weight)
            if fc.bias is not None:
                nn.init.zeros_(fc.bias)
        self.model.fc = fc

        # Freeze layers except classifier if freeze=True
        if freeze_pretrained:
            for name, param in self.model.named_parameters():
                if not name.startswith("fc"):
                    param.requires_grad = False

    def forward(self, x):
        return self.model(x)
