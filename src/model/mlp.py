import torch
import torch.nn as nn
import torch.nn.init as init

class MLP(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()

        weight_init: bool = kwargs['weight_init']
        hidden_layers: list[int] = kwargs['hidden_layers']

        input_dim = 32 * 32 * 3
        output_dim = 100

        layers = []
        prev_dim = input_dim
        for h in hidden_layers:
            layers.append(nn.Linear(prev_dim, h))
            layers.append(nn.ReLU())
            prev_dim = h

        layers.append(nn.Linear(prev_dim, output_dim))
        self.model = nn.Sequential(*layers)

        if weight_init:
            self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, x):
        x = torch.flatten(x, start_dim=1)
        return self.model(x)
