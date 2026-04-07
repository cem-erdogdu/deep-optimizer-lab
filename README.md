# Deep Learning Experiment Lab

A clean, minimal deep learning experiment framework built with PyTorch. This project serves as a foundation for experimenting with neural networks on toy datasets before scaling to more complex problems.

## Overview

This lab provides a complete training pipeline with:

- **Modular Architecture**: Clean separation of concerns (models, datasets, training, evaluation)
- **Configuration-Driven**: YAML-based experiment configuration
- **Reproducibility**: Deterministic seeding for reproducible experiments
- **Visualization**: Plotting utilities for training curves and decision boundaries
- **Testing**: Comprehensive pytest test suite

## Project Structure

```
.
├── configs/              # Experiment configuration files
│   ├── xor.yaml         # XOR dataset configuration
│   └── moons.yaml       # Moons dataset configuration
├── data/                 # Downloaded/generated data (gitignored)
├── outputs/              # Experiment outputs (models, plots, configs)
├── src/                  # Source code
│   ├── __init__.py
│   ├── config.py        # Configuration loading/saving
│   ├── datasets.py      # Toy dataset implementations
│   ├── evaluation.py    # Model evaluation utilities
│   ├── models.py        # Neural network architectures
│   ├── plotting.py      # Visualization utilities
│   ├── train.py         # Main training script
│   ├── training.py      # Training loop implementation
│   └── utils.py         # Utility functions (seeding, device selection)
├── tests/                # Test suite
│   ├── __init__.py
│   ├── test_datasets.py
│   ├── test_evaluation.py
│   ├── test_models.py
│   ├── test_training.py
│   └── test_utils.py
├── .gitignore           # Git ignore patterns
├── requirements.txt     # Python dependencies
└── README.md            # This file
```

## Installation

1. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Quick Start

### Running Experiments

Run with default configuration (XOR dataset):
```bash
python -m src.train
```

Run with a specific config file:
```bash
python -m src.train --config configs/moons.yaml
```

Override specific parameters:
```bash
python -m src.train --dataset moons --epochs 500 --lr 0.001
```

### Available Datasets

1. **XOR**: Classic 4-point non-linearly separable problem
   - Perfect for testing if your network can learn non-linear boundaries
   - Located at: [src/datasets.py](src/datasets.py)

2. **Moons**: Two interleaving half-circles from scikit-learn
   - More complex non-linear decision boundary
   - Configurable noise and sample count
   - Located at: [src/datasets.py](src/datasets.py)

## Configuration

Experiments are configured via YAML files. Here's the structure:

```yaml
experiment:
  name: my_experiment
  seed: 42              # Random seed for reproducibility

dataset:
  name: xor             # or 'moons'
  noise_std: 0.0        # Dataset-specific parameters

model:
  name: mlp
  input_dim: 2
  hidden_dims: [8, 8]   # List of hidden layer sizes
  output_dim: 2
  activation: relu      # relu, tanh, or sigmoid

training:
  epochs: 100
  batch_size: 4
  learning_rate: 0.01

output:
  save_dir: outputs
```

## Components

### Models ([src/models.py](src/models.py))

**MLP (Multi-Layer Perceptron)**: A configurable feedforward neural network
- Variable number of hidden layers
- Choice of activation functions (ReLU, Tanh, Sigmoid)
- Xavier weight initialization

Example usage:
```python
from src.models import MLP

model = MLP(
    input_dim=2,
    hidden_dims=[16, 16],
    output_dim=2,
    activation='relu'
)
```

### Datasets ([src/datasets.py](src/datasets.py))

Toy datasets for quick experimentation:
- `XORDataset`: 4-point XOR problem
- `MoonsDataset`: Scikit-learn's make_moons

Both implement PyTorch's `Dataset` interface for compatibility with `DataLoader`.

### Training ([src/training.py](src/training.py))

The `Trainer` class handles the training loop:
- Tracks loss and accuracy history
- Device-agnostic (CPU/CUDA)
- Verbose training progress

### Evaluation ([src/evaluation.py](src/evaluation.py))

Evaluation utilities:
- `evaluate_model()`: Compute loss, accuracy, and predictions
- `get_decision_boundary()`: Generate decision boundary grids for visualization

### Plotting ([src/plotting.py](src/plotting.py))

Visualization functions:
- `plot_training_history()`: Loss and accuracy curves
- `plot_decision_boundary()`: Decision boundary with data overlay
- `plot_dataset()`: Simple dataset scatter plot

### Utilities ([src/utils.py](src/utils.py))

Helper functions:
- `set_seed()`: Reproducible experiments
- `get_device()`: Auto-select CUDA if available
- `count_parameters()`: Model size statistics

## Running Tests

Run the full test suite:
```bash
pytest
```

Run with verbose output:
```bash
pytest -v
```

Run specific test file:
```bash
pytest tests/test_models.py
```

## Example: Training on XOR

The XOR problem is a classic test of a neural network's ability to learn non-linear decision boundaries. A single-layer network cannot solve it, but our MLP with hidden layers can.

```python
from src.datasets import XORDataset
from src.models import MLP
from src.training import Trainer
from src.utils import set_seed
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Set seed for reproducibility
set_seed(42)

# Create dataset
dataset = XORDataset()
dataloader = DataLoader(dataset, batch_size=4)

# Create model
model = MLP(input_dim=2, hidden_dims=[8, 8], output_dim=2)

# Train
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
trainer = Trainer(model, optimizer, nn.CrossEntropyLoss())
history = trainer.fit(dataloader, epochs=200)

# Should achieve 100% accuracy
print(f"Final accuracy: {history['train_acc'][-1]:.4f}")
```

## Extending the Lab

### Adding a New Dataset

1. Create a new class in [src/datasets.py](src/datasets.py):
```python
class MyDataset(Dataset):
    def __init__(self, ...):
        # Load/generate data
        pass

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
```

2. Register in `get_dataset()` factory function

3. Create a config file in `configs/`

### Adding a New Model

1. Create model class in [src/models.py](src/models.py) or new file
2. Inherit from `nn.Module`
3. Implement `forward()` method
4. Update training script to support the new model

## Tips for Experimentation

1. **Start Small**: Use XOR to verify your network architecture works
2. **Check Decision Boundaries**: Visualize what the model learned
3. **Monitor Training**: Watch for overfitting/underfitting in loss curves
4. **Vary Architecture**: Try different hidden layer sizes and depths
5. **Experiment with Seeds**: Run multiple seeds to check robustness

## Future Extensions

This lab is designed to grow. Potential additions:
- Convolutional layers for image data
- Recurrent layers for sequence data
- Regularization techniques (dropout, weight decay)
- Learning rate scheduling
- Validation set support
- TensorBoard logging
- Hyperparameter search

## License

MIT License - Feel free to use and extend for your own experiments!
