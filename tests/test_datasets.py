"""Tests for dataset implementations."""

import numpy as np
import torch

from src.datasets import MoonsDataset, XORDataset, get_dataset


def test_xor_dataset_length():
    """Test XOR dataset has 4 samples."""
    dataset = XORDataset()
    assert len(dataset) == 4


def test_xor_dataset_samples():
    """Test XOR dataset returns correct data types."""
    dataset = XORDataset()
    x, y = dataset[0]

    assert isinstance(x, torch.Tensor)
    assert isinstance(y, torch.Tensor)
    assert x.shape == (2,)
    assert y.shape == ()


def test_xor_labels():
    """Test XOR labels are correct."""
    dataset = XORDataset()
    expected_labels = [0, 1, 1, 0]  # XOR truth table

    for i in range(len(dataset)):
        _, y = dataset[i]
        assert y.item() == expected_labels[i]


def test_xor_with_noise():
    """Test XOR dataset with noise."""
    dataset = XORDataset(noise_std=0.1)
    assert len(dataset) == 4


def test_moons_dataset():
    """Test Moons dataset creation."""
    dataset = MoonsDataset(n_samples=100, noise=0.1, random_state=42)
    assert len(dataset) == 100

    x, y = dataset[0]
    assert x.shape == (2,)
    assert y.shape == ()


def test_moons_reproducibility():
    """Test Moons dataset is reproducible with same seed."""
    dataset1 = MoonsDataset(n_samples=50, random_state=42)
    dataset2 = MoonsDataset(n_samples=50, random_state=42)

    for i in range(len(dataset1)):
        x1, y1 = dataset1[i]
        x2, y2 = dataset2[i]
        assert torch.allclose(x1, x2)
        assert y1 == y2


def test_get_dataset_factory():
    """Test dataset factory function."""
    xor_config = {"name": "xor", "noise_std": 0.0}
    dataset = get_dataset(xor_config)
    assert isinstance(dataset, XORDataset)

    moons_config = {"name": "moons", "n_samples": 100}
    dataset = get_dataset(moons_config)
    assert isinstance(dataset, MoonsDataset)
