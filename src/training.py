"""Training utilities."""

from typing import Callable

import torch
import torch.nn as nn
from torch.utils.data import DataLoader


class Trainer:
    """Simple training loop for classification tasks.

    Args:
        model: Neural network model
        optimizer: PyTorch optimizer
        loss_fn: Loss function
        device: Device to train on ('cpu' or 'cuda')
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        loss_fn: nn.Module,
        device: str = "cpu",
    ):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.device = device
        self.history = {"train_loss": [], "train_acc": []}

    def train_epoch(self, dataloader: DataLoader) -> dict[str, float]:
        """Train for one epoch.

        Args:
            dataloader: Training data loader

        Returns:
            Dictionary with train_loss and train_acc
        """
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        for batch_x, batch_y in dataloader:
            batch_x = batch_x.to(self.device)
            batch_y = batch_y.to(self.device)

            # Forward pass
            logits = self.model(batch_x)
            loss = self.loss_fn(logits, batch_y)

            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # Statistics
            total_loss += loss.item() * len(batch_x)
            predictions = logits.argmax(dim=1)
            correct += (predictions == batch_y).sum().item()
            total += len(batch_x)

        avg_loss = total_loss / total
        accuracy = correct / total

        return {"train_loss": avg_loss, "train_acc": accuracy}

    def fit(
        self,
        train_loader: DataLoader,
        epochs: int,
        verbose: bool = True,
    ) -> dict[str, list[float]]:
        """Train for multiple epochs.

        Args:
            train_loader: Training data loader
            epochs: Number of epochs to train
            verbose: Whether to print progress

        Returns:
            Training history dictionary
        """
        for epoch in range(epochs):
            metrics = self.train_epoch(train_loader)

            # Store history
            for key, value in metrics.items():
                self.history[key].append(value)

            if verbose:
                print(
                    f"Epoch [{epoch+1}/{epochs}] "
                    f"Loss: {metrics['train_loss']:.4f} "
                    f"Acc: {metrics['train_acc']:.4f}"
                )

        return self.history
