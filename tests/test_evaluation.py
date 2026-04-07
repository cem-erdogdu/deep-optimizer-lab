"""Tests for evaluation utilities."""

import torch
from torch.utils.data import DataLoader

from src.datasets import XORDataset
from src.evaluation import evaluate_model, get_decision_boundary
from src.models import MLP


def test_evaluate_model():
    """Test model evaluation returns correct metrics."""
    model = MLP(input_dim=2, hidden_dims=[8], output_dim=2)
    dataset = XORDataset()
    dataloader = DataLoader(dataset, batch_size=4)

    metrics = evaluate_model(model, dataloader)

    assert "loss" in metrics
    assert "accuracy" in metrics
    assert "predictions" in metrics
    assert "labels" in metrics
    assert 0 <= metrics["accuracy"] <= 1


def test_get_decision_boundary():
    """Test decision boundary generation."""
    model = MLP(input_dim=2, hidden_dims=[4], output_dim=2)

    xx, yy, Z = get_decision_boundary(model, -1, 1, -1, 1, resolution=50)

    assert xx.shape == (50, 50)
    assert yy.shape == (50, 50)
    assert Z.shape == (50, 50)
    assert 0 <= Z.min() <= Z.max() <= 1  # Probabilities
