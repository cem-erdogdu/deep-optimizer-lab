"""Evaluation utilities."""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader


def evaluate_model(
    model: nn.Module,
    dataloader: DataLoader,
    device: str = "cpu",
) -> dict[str, float]:
    """Evaluate model on a dataset.

    Args:
        model: Neural network model
        dataloader: Data loader for evaluation
        device: Device to evaluate on

    Returns:
        Dictionary with loss, accuracy, and predictions
    """
    model.eval()
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    correct = 0
    total = 0

    all_predictions = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for batch_x, batch_y in dataloader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            logits = model(batch_x)
            loss = criterion(logits, batch_y)

            total_loss += loss.item() * len(batch_x)

            probs = torch.softmax(logits, dim=1)
            predictions = logits.argmax(dim=1)

            correct += (predictions == batch_y).sum().item()
            total += len(batch_x)

            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(batch_y.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    metrics = {
        "loss": total_loss / total,
        "accuracy": correct / total,
        "predictions": np.array(all_predictions),
        "labels": np.array(all_labels),
        "probabilities": np.array(all_probs),
    }

    return metrics


def get_decision_boundary(
    model: nn.Module,
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
    resolution: int = 100,
    device: str = "cpu",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate decision boundary grid for 2D classification.

    Args:
        model: Trained model
        x_min, x_max: X-axis range
        y_min, y_max: Y-axis range
        resolution: Grid resolution
        device: Device for computation

    Returns:
        Tuple of (xx, yy, Z) where Z contains predicted class probabilities
    """
    model.eval()

    # Create grid
    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, resolution),
        np.linspace(y_min, y_max, resolution),
    )

    # Flatten and predict
    grid_points = np.c_[xx.ravel(), yy.ravel()]
    grid_tensor = torch.from_numpy(grid_points.astype(np.float32)).to(device)

    with torch.no_grad():
        logits = model(grid_tensor)
        probs = torch.softmax(logits, dim=1)
        # Get probability of class 1 for binary classification
        if probs.shape[1] == 2:
            Z = probs[:, 1].cpu().numpy()
        else:
            Z = probs.argmax(dim=1).cpu().numpy()

    Z = Z.reshape(xx.shape)

    return xx, yy, Z
