"""Toy datasets for experiments."""

import numpy as np
import torch
from sklearn.datasets import make_moons
from torch.utils.data import Dataset


class XORDataset(Dataset):
    """XOR classification dataset.

    The classic XOR problem: 4 data points at the corners of a square.
    Labels are 1 if x != y (XOR), 0 otherwise.

    Args:
        noise_std: Standard deviation of Gaussian noise to add
    """

    def __init__(self, noise_std: float = 0.0):
        # XOR truth table
        self.X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float32)
        self.y = np.array([0, 1, 1, 0], dtype=np.int64)

        # Add noise if requested
        if noise_std > 0:
            self.X += np.random.normal(0, noise_std, self.X.shape)

        self.X = torch.from_numpy(self.X)
        self.y = torch.from_numpy(self.y)

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.X[idx], self.y[idx]


class MoonsDataset(Dataset):
    """Two interleaving half circles (moons) dataset.

    Uses sklearn's make_moons to generate non-linearly separable data.

    Args:
        n_samples: Number of samples to generate
        noise: Standard deviation of Gaussian noise
        random_state: Random seed for reproducibility
    """

    def __init__(self, n_samples: int = 200, noise: float = 0.1, random_state: int = 42):
        X, y = make_moons(n_samples=n_samples, noise=noise, random_state=random_state)
        self.X = torch.from_numpy(X.astype(np.float32))
        self.y = torch.from_numpy(y.astype(np.int64))

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.X[idx], self.y[idx]


def get_dataset(config: dict) -> Dataset:
    """Factory function to create datasets from config.

    Args:
        config: Dataset configuration dict with 'name' and other params

    Returns:
        Dataset instance
    """
    name = config["name"].lower()

    if name == "xor":
        return XORDataset(noise_std=config.get("noise_std", 0.0))
    elif name == "moons":
        return MoonsDataset(
            n_samples=config.get("n_samples", 200),
            noise=config.get("noise", 0.1),
            random_state=config.get("random_state", 42),
        )
    else:
        raise ValueError(f"Unknown dataset: {name}")
