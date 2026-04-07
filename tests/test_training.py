"""Tests for training components."""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.datasets import XORDataset
from src.models import MLP
from src.training import Trainer


def test_trainer_initialization():
    """Test trainer can be initialized."""
    model = MLP(input_dim=2, hidden_dims=[4], output_dim=2)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    loss_fn = nn.CrossEntropyLoss()

    trainer = Trainer(model, optimizer, loss_fn)
    assert trainer is not None
    assert trainer.history == {"train_loss": [], "train_acc": []}


def test_trainer_train_epoch():
    """Test trainer can run one epoch."""
    model = MLP(input_dim=2, hidden_dims=[8], output_dim=2)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    loss_fn = nn.CrossEntropyLoss()

    dataset = XORDataset()
    dataloader = DataLoader(dataset, batch_size=4)

    trainer = Trainer(model, optimizer, loss_fn)
    metrics = trainer.train_epoch(dataloader)

    assert "train_loss" in metrics
    assert "train_acc" in metrics
    assert 0 <= metrics["train_acc"] <= 1


def test_trainer_fit():
    """Test trainer fit method."""
    model = MLP(input_dim=2, hidden_dims=[8], output_dim=2)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    loss_fn = nn.CrossEntropyLoss()

    dataset = XORDataset()
    dataloader = DataLoader(dataset, batch_size=4)

    trainer = Trainer(model, optimizer, loss_fn)
    history = trainer.fit(dataloader, epochs=5, verbose=False)

    assert len(history["train_loss"]) == 5
    assert len(history["train_acc"]) == 5
