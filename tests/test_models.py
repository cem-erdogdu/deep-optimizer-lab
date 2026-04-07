"""Tests for model implementations."""

import pytest
import torch

from src.models import MLP


def test_mlp_creation():
    """Test MLP can be created with various configurations."""
    model = MLP(input_dim=2, hidden_dims=[8, 8], output_dim=2)
    assert isinstance(model, MLP)


def test_mlp_forward():
    """Test MLP forward pass produces correct output shape."""
    model = MLP(input_dim=2, hidden_dims=[8], output_dim=2)
    batch_size = 4
    x = torch.randn(batch_size, 2)

    output = model(x)

    assert output.shape == (batch_size, 2)


def test_mlp_different_activations():
    """Test MLP works with different activation functions."""
    for activation in ["relu", "tanh", "sigmoid"]:
        model = MLP(
            input_dim=2,
            hidden_dims=[4],
            output_dim=2,
            activation=activation,
        )
        x = torch.randn(2, 2)
        output = model(x)
        assert output.shape == (2, 2)


def test_mlp_invalid_activation():
    """Test MLP raises error for invalid activation."""
    with pytest.raises(ValueError):
        MLP(input_dim=2, hidden_dims=[4], output_dim=2, activation="invalid")


def test_mlp_no_hidden_layers():
    """Test MLP works with empty hidden layers (logistic regression)."""
    model = MLP(input_dim=2, hidden_dims=[], output_dim=2)
    x = torch.randn(4, 2)
    output = model(x)
    assert output.shape == (4, 2)
