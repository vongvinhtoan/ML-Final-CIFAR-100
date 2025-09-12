import torch
import torch.nn as nn
import torch.nn.init as init


class SmallCNN(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()

        weight_init = kwargs['weight_init']

        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),  # 3x32x32 -> 32x32x32
            nn.ReLU(),
            nn.MaxPool2d(2),  # -> 32x16x16

            nn.Conv2d(32, 64, kernel_size=3, padding=1),  # -> 64x16x16
            nn.ReLU(),
            nn.MaxPool2d(2),  # -> 64x8x8

            nn.Conv2d(64, 128, kernel_size=3, padding=1),  # -> 128x8x8
            nn.ReLU(),
            nn.MaxPool2d(2),  # -> 128x4x4
        )

        self.classifier = nn.Sequential(
            nn.Linear(128 * 4 * 4, 256),
            nn.ReLU(),
            nn.Linear(256, 100)
        )

        if weight_init:
            self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
            init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, start_dim=1)
        return self.classifier(x)
