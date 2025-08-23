import torch.nn as nn
from torchvision import models

class Resnet(nn.Module):
    def __init__(self):
        super().__init__()

        self.model = models.resnet101(weights=models.ResNet101_Weights.DEFAULT)
        self.modify_model()

    def modify_model(self):
        in_features = self.model.fc.in_features
        self.model.fc = nn.Linear(in_features, 100)

    def forward(self, x):
        return self.model(x)
