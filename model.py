"""
Neural network architecture for quadratic equation prediction.
"""
import torch
import torch.nn as nn


class DynamicQuadraticNet(nn.Module):
    """
    A scalable MLP that adjusts depth and width based on hyperparams.
    """

    def __init__(self, input_dim=3, output_dim=4, hidden_layers=3, hidden_units=64, dropout_rate=0.1):
        super(DynamicQuadraticNet, self).__init__()

        self.config = {
            'input_dim': input_dim,
            'output_dim': output_dim,
            'hidden_layers': hidden_layers,
            'hidden_units': hidden_units,
            'dropout_rate': dropout_rate
        }

        layers = []
        # Input layer
        layers.append(nn.Linear(input_dim, hidden_units))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(dropout_rate))

        # Hidden layers (The Depth)
        for _ in range(hidden_layers - 1):
            layers.append(nn.Linear(hidden_units, hidden_units))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))

        # Output layer
        layers.append(nn.Linear(hidden_units, output_dim))

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

