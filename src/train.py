"""Main training script for running experiments."""

import argparse
import sys
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.config import get_default_config, load_config, save_config
from src.datasets import get_dataset
from src.evaluation import evaluate_model
from src.models import MLP
from src.plotting import plot_decision_boundary, plot_training_history
from src.training import Trainer
from src.utils import count_parameters, ensure_dir, get_device, set_seed


def run_experiment(config: dict) -> dict:
    """Run a training experiment from configuration.

    Args:
        config: Experiment configuration dictionary

    Returns:
        Dictionary with training results
    """
    # Set seed for reproducibility
    seed = config["experiment"].get("seed", 42)
    set_seed(seed)

    # Setup device
    device = get_device()
    print(f"Using device: {device}")

    # Create dataset
    print(f"Loading dataset: {config['dataset']['name']}")
    dataset = get_dataset(config["dataset"])
    dataloader = DataLoader(
        dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=True,
    )

    # Create model
    print(f"Creating model: {config['model']['name']}")
    model = MLP(
        input_dim=config["model"]["input_dim"],
        hidden_dims=config["model"]["hidden_dims"],
        output_dim=config["model"]["output_dim"],
        activation=config["model"].get("activation", "relu"),
    )
    print(f"Model parameters: {count_parameters(model):,}")

    # Setup optimizer and loss
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config["training"]["learning_rate"],
    )
    loss_fn = nn.CrossEntropyLoss()

    # Train
    print(f"Training for {config['training']['epochs']} epochs...")
    trainer = Trainer(model, optimizer, loss_fn, device=device)
    history = trainer.fit(
        dataloader,
        epochs=config["training"]["epochs"],
        verbose=True,
    )

    # Evaluate
    print("\nEvaluating model...")
    metrics = evaluate_model(model, dataloader, device=device)
    print(f"Final Loss: {metrics['loss']:.4f}")
    print(f"Final Accuracy: {metrics['accuracy']:.4f}")

    # Save outputs
    output_dir = Path(config["output"]["save_dir"]) / config["experiment"]["name"]
    ensure_dir(output_dir)

    # Save config
    save_config(config, output_dir / "config.yaml")

    # Save plots
    plot_training_history(history, save_path=output_dir / "training_history.png")
    print(f"Saved training history to {output_dir / 'training_history.png'}")

    plot_decision_boundary(
        model,
        dataset,
        device=device,
        title=f"{config['dataset']['name'].upper()} Decision Boundary",
        save_path=output_dir / "decision_boundary.png",
    )
    print(f"Saved decision boundary to {output_dir / 'decision_boundary.png'}")

    # Save model
    torch.save(model.state_dict(), output_dir / "model.pt")
    print(f"Saved model to {output_dir / 'model.pt'}")

    return {
        "history": history,
        "final_metrics": metrics,
        "output_dir": output_dir,
    }


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Train a neural network on toy datasets")
    parser.add_argument(
        "--config",
        type=str,
        help="Path to config YAML file (uses default if not provided)",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["xor", "moons"],
        help="Override dataset name",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        help="Override number of epochs",
    )
    parser.add_argument(
        "--lr",
        type=float,
        help="Override learning rate",
    )

    args = parser.parse_args()

    # Load config
    if args.config:
        config = load_config(args.config)
    else:
        config = get_default_config()

    # Apply overrides
    if args.dataset:
        config["dataset"]["name"] = args.dataset
    if args.epochs:
        config["training"]["epochs"] = args.epochs
    if args.lr:
        config["training"]["learning_rate"] = args.lr

    # Run experiment
    print(f"Running experiment: {config['experiment']['name']}")
    print("-" * 50)

    try:
        results = run_experiment(config)
        print("\nExperiment completed successfully!")
        return 0
    except Exception as e:
        print(f"\nError: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
