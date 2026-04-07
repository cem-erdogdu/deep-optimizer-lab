"""Plotting utilities for visualizing training and results."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import Dataset


def plot_training_history(
    history: dict[str, list[float]],
    save_path: str | Path | None = None,
) -> plt.Figure:
    """Plot training loss and accuracy curves.

    Args:
        history: Training history dictionary with 'train_loss' and 'train_acc'
        save_path: Optional path to save the figure

    Returns:
        Matplotlib figure
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    epochs = range(1, len(history["train_loss"]) + 1)

    # Loss plot
    ax1.plot(epochs, history["train_loss"], "b-", linewidth=2, label="Train Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("Training Loss")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Accuracy plot
    ax2.plot(epochs, history["train_acc"], "g-", linewidth=2, label="Train Accuracy")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.set_title("Training Accuracy")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim([0, 1.05])

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


def plot_decision_boundary(
    model: torch.nn.Module,
    dataset: Dataset,
    device: str = "cpu",
    title: str = "Decision Boundary",
    save_path: str | Path | None = None,
) -> plt.Figure:
    """Plot dataset with model decision boundary.

    Args:
        model: Trained model
        dataset: Dataset with 2D features
        device: Device for model inference
        title: Plot title
        save_path: Optional path to save the figure

    Returns:
        Matplotlib figure
    """
    from src.evaluation import get_decision_boundary

    # Get all data points
    X = []
    y = []
    for i in range(len(dataset)):
        xi, yi = dataset[i]
        X.append(xi.numpy())
        y.append(yi.item())
    X = np.array(X)
    y = np.array(y)

    # Determine plot boundaries with padding
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5

    # Get decision boundary
    model.eval()
    xx, yy, Z = get_decision_boundary(
        model, x_min, x_max, y_min, y_max, device=device
    )

    # Create plot
    fig, ax = plt.subplots(figsize=(8, 6))

    # Plot decision boundary
    contour = ax.contourf(xx, yy, Z, levels=50, alpha=0.6, cmap="RdYlBu")
    plt.colorbar(contour, ax=ax, label="Class 1 Probability")

    # Plot data points
    scatter = ax.scatter(
        X[:, 0], X[:, 1], c=y, cmap="RdYlBu", edgecolors="black", s=100, zorder=10
    )

    ax.set_xlabel("Feature 1")
    ax.set_ylabel("Feature 2")
    ax.set_title(title)
    ax.set_xlim([x_min, x_max])
    ax.set_ylim([y_min, y_max])

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


def plot_dataset(
    dataset: Dataset,
    title: str = "Dataset",
    save_path: str | Path | None = None,
) -> plt.Figure:
    """Simple scatter plot of a 2D dataset.

    Args:
        dataset: Dataset with 2D features
        title: Plot title
        save_path: Optional path to save the figure

    Returns:
        Matplotlib figure
    """
    # Get all data points
    X = []
    y = []
    for i in range(len(dataset)):
        xi, yi = dataset[i]
        X.append(xi.numpy())
        y.append(yi.item())
    X = np.array(X)
    y = np.array(y)

    fig, ax = plt.subplots(figsize=(6, 6))

    scatter = ax.scatter(
        X[:, 0], X[:, 1], c=y, cmap="RdYlBu", edgecolors="black", s=100
    )
    plt.colorbar(scatter, ax=ax, label="Class")

    ax.set_xlabel("Feature 1")
    ax.set_ylabel("Feature 2")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig
