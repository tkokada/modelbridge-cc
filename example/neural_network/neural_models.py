"""Simple neural network models for model bridge demonstration."""

import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import torchvision
import torchvision.transforms as transforms

from modelbridge.types import ParamDict


class MicroCNN(nn.Module):
    """Micro model: Detailed CNN with more parameters (higher accuracy, slower)."""

    def __init__(self, dropout_rate: float = 0.5, hidden_size: int = 128) -> None:
        super().__init__()
        self.dropout_rate = dropout_rate
        self.hidden_size = hidden_size

        # Convolutional layers
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(dropout_rate)

        # Fully connected layers
        self.fc1 = nn.Linear(64 * 7 * 7, hidden_size)
        self.fc2 = nn.Linear(hidden_size, 64)
        self.fc3 = nn.Linear(64, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 64 * 7 * 7)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class MacroMLP(nn.Module):
    """Macro model: Simple MLP (lower accuracy, faster)."""

    def __init__(self, hidden_size: int = 64, dropout_rate: float = 0.3) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.dropout_rate = dropout_rate

        # Simple fully connected network
        self.fc1 = nn.Linear(28 * 28, hidden_size)
        self.fc2 = nn.Linear(hidden_size, 32)
        self.fc3 = nn.Linear(32, 10)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        x = x.view(-1, 28 * 28)  # Flatten
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class NeuralNetworkTrainer:
    """Utility for training neural networks quickly."""

    def __init__(self, device: str = "cpu") -> None:
        self.device = device

    def prepare_mnist_data(
        self, subset_size: int = 1000
    ) -> tuple[DataLoader, DataLoader]:
        """Prepare MNIST data with small subset for fast training."""
        transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
        )

        # Load MNIST dataset
        train_dataset = torchvision.datasets.MNIST(
            root="./data", train=True, download=True, transform=transform
        )
        test_dataset = torchvision.datasets.MNIST(
            root="./data", train=False, download=True, transform=transform
        )

        # Use small subsets for fast training
        train_subset = Subset(train_dataset, list(range(subset_size)))
        test_subset = Subset(test_dataset, list(range(subset_size // 5)))

        train_loader = DataLoader(train_subset, batch_size=32, shuffle=True)
        test_loader = DataLoader(test_subset, batch_size=32, shuffle=False)

        return train_loader, test_loader

    def train_model(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        learning_rate: float = 0.001,
        epochs: int = 2,
    ) -> float:
        """Train model and return final loss."""
        model.to(self.device)
        model.train()

        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        criterion = nn.CrossEntropyLoss()

        total_loss = 0.0
        num_batches = 0

        for _epoch in range(epochs):
            epoch_loss = 0.0
            batch_count = 0

            for batch_idx, (data, target) in enumerate(train_loader):
                data, target = data.to(self.device), target.to(self.device)

                optimizer.zero_grad()
                output = model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                batch_count += 1

                # Early stopping for speed (max 10 batches per epoch)
                if batch_idx >= 9:
                    break

            total_loss += epoch_loss / batch_count
            num_batches += 1

        return total_loss / num_batches

    def evaluate_model(self, model: nn.Module, test_loader: DataLoader) -> float:
        """Evaluate model accuracy."""
        model.to(self.device)
        model.eval()

        correct = 0
        total = 0

        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(test_loader):
                data, target = data.to(self.device), target.to(self.device)
                output = model(data)
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
                total += target.size(0)

                # Early stopping for speed
                if batch_idx >= 4:
                    break

        return correct / total if total > 0 else 0.0


class NeuralNetworkObjectives:
    """Objective functions for neural network optimization."""

    def __init__(self, subset_size: int = 500) -> None:
        self.trainer = NeuralNetworkTrainer()
        self.train_loader, self.test_loader = self.trainer.prepare_mnist_data(
            subset_size
        )
        print(
            f"Loaded MNIST data: {subset_size} training samples, {subset_size // 5} test samples"
        )

    def micro_objective(self, params: ParamDict) -> float:
        """Micro model objective (detailed CNN)."""
        dropout_rate = float(params["dropout_rate"])
        hidden_size = int(params["hidden_size"])
        learning_rate = float(params["learning_rate"])

        # Create and train micro model
        model = MicroCNN(dropout_rate=dropout_rate, hidden_size=hidden_size)

        start_time = time.time()
        loss = self.trainer.train_model(
            model, self.train_loader, learning_rate, epochs=1
        )
        training_time = time.time() - start_time

        # Evaluate accuracy
        accuracy = self.trainer.evaluate_model(model, self.test_loader)

        # Objective: minimize (1 - accuracy) + small penalty for training time
        objective_value = (1.0 - accuracy) + 0.001 * training_time

        print(
            f"Micro CNN - Loss: {loss:.4f}, Accuracy: {accuracy:.4f}, "
            f"Time: {training_time:.2f}s, Objective: {objective_value:.4f}"
        )

        return objective_value

    def macro_objective(self, params: ParamDict, target_value: float) -> float:
        """Macro model objective (simple MLP approximation)."""
        hidden_size = int(params.get("macro_hidden_size", 32))
        dropout_rate = float(params.get("macro_dropout_rate", 0.2))
        learning_rate = float(params["learning_rate"])

        # Create and train macro model
        model = MacroMLP(hidden_size=hidden_size, dropout_rate=dropout_rate)

        start_time = time.time()
        loss = self.trainer.train_model(
            model, self.train_loader, learning_rate, epochs=1
        )
        training_time = time.time() - start_time

        # Evaluate accuracy
        accuracy = self.trainer.evaluate_model(model, self.test_loader)

        # Objective: minimize (1 - accuracy) + penalty for training time
        objective_value = (1.0 - accuracy) + 0.002 * training_time

        print(
            f"Macro MLP - Loss: {loss:.4f}, Accuracy: {accuracy:.4f}, "
            f"Time: {training_time:.2f}s, Objective: {objective_value:.4f}"
        )

        return objective_value


def quick_mnist_test() -> None:
    """Quick test to ensure MNIST data loading works."""
    print("Testing MNIST data loading...")

    trainer = NeuralNetworkTrainer()
    train_loader, test_loader = trainer.prepare_mnist_data(subset_size=100)

    # Test with a simple model
    model = MacroMLP(hidden_size=32)
    loss = trainer.train_model(model, train_loader, epochs=1)
    accuracy = trainer.evaluate_model(model, test_loader)

    print(f"Quick test - Loss: {loss:.4f}, Accuracy: {accuracy:.4f}")
    print("MNIST data loading successful!")


if __name__ == "__main__":
    quick_mnist_test()
