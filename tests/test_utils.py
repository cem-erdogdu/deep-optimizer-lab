"""Tests for utility functions."""

import numpy as np
import torch

from src.models import MLP
from src.utils import count_parameters, ensure_dir, get_device, set_seed


def test_set_seed():
    """Test set_seed produces reproducible results."""
    set_seed(42)
    rand1 = np.random.rand(5)

    set_seed(42)
    rand2 = np.random.rand(5)

    assert np.allclose(rand1, rand2)


def test_set_seed_torch():
    """Test set_seed produces reproducible PyTorch results."""
    set_seed(42)
    rand1 = torch.rand(5)

    set_seed(42)
    rand2 = torch.rand(5)

    assert torch.allclose(rand1, rand2)


def test_get_device():
    """Test get_device returns valid device string."""
    device = get_device()
    assert device in ["cpu", "cuda"]


def test_ensure_dir(tmp_path):
    """Test ensure_dir creates directory."""
    test_dir = tmp_path / "test_subdir"
    result = ensure_dir(test_dir)
    assert result.exists()
    assert result.is_dir()


def test_count_parameters():
    """Test count_parameters returns correct count."""
    model = MLP(input_dim=2, hidden_dims=[4], output_dim=2)
    count = count_parameters(model)

    # Manual calculation: (2*4 + 4) + (4*2 + 2) = 12 + 10 = 22
    expected = (2 * 4 + 4) + (4 * 2 + 2)
    assert count == expected
