"""Configuration management."""

from pathlib import Path
from typing import Any

import yaml


def load_config(config_path: str | Path) -> dict[str, Any]:
    """Load YAML configuration file.

    Args:
        config_path: Path to YAML config file

    Returns:
        Configuration dictionary
    """
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def save_config(config: dict[str, Any], save_path: str | Path) -> None:
    """Save configuration to YAML file.

    Args:
        config: Configuration dictionary
        save_path: Path to save YAML file
    """
    with open(save_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)


def get_default_config() -> dict[str, Any]:
    """Get default configuration for experiments.

    Returns:
        Default configuration dictionary
    """
    return {
        "experiment": {
            "name": "xor_experiment",
            "seed": 42,
        },
        "dataset": {
            "name": "xor",
            "noise_std": 0.0,
        },
        "model": {
            "name": "mlp",
            "input_dim": 2,
            "hidden_dims": [8, 8],
            "output_dim": 2,
            "activation": "relu",
        },
        "training": {
            "epochs": 100,
            "batch_size": 4,
            "learning_rate": 0.01,
        },
        "output": {
            "save_dir": "outputs",
        },
    }
