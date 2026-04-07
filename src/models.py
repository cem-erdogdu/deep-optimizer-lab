"""Neural network models."""

import torch
import torch.nn as nn


class MLP(nn.Module):
    """Multi-Layer Perceptron for classification tasks.

    A simple feedforward neural network with configurable hidden layers
    and activation functions.

    Args:
        input_dim: Number of input features
        hidden_dims: List of hidden layer dimensions
        output_dim: Number of output classes
        activation: Activation function to use ('relu', 'tanh', 'sigmoid')
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: list[int],
        output_dim: int,
        activation: str = "relu",
    ):
        super().__init__()

        # Build layers
        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            if activation == "relu":
                layers.append(nn.ReLU())
            elif activation == "tanh":
                layers.append(nn.Tanh())
            elif activation == "sigmoid":
                layers.append(nn.Sigmoid())
            else:
                raise ValueError(f"Unknown activation: {activation}")
            prev_dim = hidden_dim

        # Output layer (no activation - handled by loss function)
        layers.append(nn.Linear(prev_dim, output_dim))

        self.network = nn.Sequential(*layers)

        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize weights using Xavier initialization."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape (batch_size, input_dim)

        Returns:
            Output logits of shape (batch_size, output_dim)
        """
        return self.network(x)
