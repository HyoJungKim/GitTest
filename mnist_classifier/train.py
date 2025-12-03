import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from model import MNISTNet
from utils import accuracy


def get_data_loaders(batch_size=64, test_batch_size=1000):
    """
    Create train and test data loaders for MNIST

    Args:
        batch_size: Batch size for training
        test_batch_size: Batch size for testing

    Returns:
        train_loader, test_loader
    """
    # Data preprocessing
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    # Load MNIST dataset
    train_dataset = datasets.MNIST(
        root='./data',
        train=True,
        download=True,
        transform=transform
    )

    test_dataset = datasets.MNIST(
        root='./data',
        train=False,
        download=True,
        transform=transform
    )

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=test_batch_size,
        shuffle=False
    )

    return train_loader, test_loader


def train_epoch(model, device, train_loader, optimizer, criterion, epoch):
    """
    Train for one epoch

    Args:
        model: Neural network model
        device: Device to train on (cpu/cuda)
        train_loader: Training data loader
        optimizer: Optimizer
        criterion: Loss function
        epoch: Current epoch number
    """
    model.train()
    running_loss = 0.0
    running_acc = 0.0

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)

        # Zero gradients
        optimizer.zero_grad()

        # Forward pass
        output = model(data)
        loss = criterion(output, target)

        # Backward pass and optimize
        loss.backward()
        optimizer.step()

        # Calculate accuracy
        acc = accuracy(output, target)

        running_loss += loss.item()
        running_acc += acc

        # Print progress
        if batch_idx % 100 == 0:
            print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} '
                  f'({100. * batch_idx / len(train_loader):.0f}%)]\t'
                  f'Loss: {loss.item():.6f}\tAccuracy: {acc:.2f}%')

    # Print epoch statistics
    avg_loss = running_loss / len(train_loader)
    avg_acc = running_acc / len(train_loader)
    print(f'\nTrain Epoch: {epoch} - Average Loss: {avg_loss:.6f}, Average Accuracy: {avg_acc:.2f}%\n')


def evaluate(model, device, test_loader, criterion):
    """
    Evaluate model on test set

    Args:
        model: Neural network model
        device: Device to evaluate on (cpu/cuda)
        test_loader: Test data loader
        criterion: Loss function

    Returns:
        Average loss and accuracy
    """
    model.eval()
    test_loss = 0
    test_acc = 0

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)

            # Forward pass
            output = model(data)
            test_loss += criterion(output, target).item()

            # Calculate accuracy
            test_acc += accuracy(output, target) * target.size(0)

    # Calculate average metrics
    test_loss /= len(test_loader)
    test_acc /= len(test_loader.dataset)
    test_acc *= 100

    print(f'Test set: Average loss: {test_loss:.4f}, Accuracy: {test_acc:.2f}%\n')

    return test_loss, test_acc


def main():
    """
    Main training function
    """
    # Hyperparameters
    batch_size = 64
    test_batch_size = 1000
    epochs = 10
    lr = 0.001

    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}\n')

    # Load data
    print('Loading data...')
    train_loader, test_loader = get_data_loaders(batch_size, test_batch_size)
    print(f'Training samples: {len(train_loader.dataset)}')
    print(f'Test samples: {len(test_loader.dataset)}\n')

    # Initialize model
    model = MNISTNet().to(device)
    print('Model architecture:')
    print(model)
    print()

    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Training loop
    print('Starting training...\n')
    for epoch in range(1, epochs + 1):
        train_epoch(model, device, train_loader, optimizer, criterion, epoch)
        evaluate(model, device, test_loader, criterion)

    # Save model
    torch.save(model.state_dict(), 'mnist_cnn.pth')
    print('Model saved to mnist_cnn.pth')


if __name__ == '__main__':
    main()
