import torch
import torch.nn as nn
import torch.nn.functional as F


class MNISTNet(nn.Module):
    """
    Basic CNN for MNIST classification
    Architecture:
    - Conv1: 1 -> 32 channels, 3x3 kernel
    - Conv2: 32 -> 64 channels, 3x3 kernel
    - FC1: 9216 -> 128
    - FC2: 128 -> 10
    """
    def __init__(self):
        super(MNISTNet, self).__init__()
        # Convolutional layers
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)

        # Fully connected layers
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)

        # Dropout for regularization
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        # Conv1 -> ReLU -> AvgPool
        x = F.relu(self.conv1(x))
        x = F.avg_pool2d(x, 2)

        # Conv2 -> ReLU -> AvgPool
        x = F.relu(self.conv2(x))
        x = F.avg_pool2d(x, 2)

        # Flatten
        x = x.view(-1, 64 * 7 * 7)

        # FC1 -> ReLU -> Dropout
        x = F.relu(self.fc1(x))
        x = self.dropout(x)

        # FC2 (output layer)
        x = self.fc2(x)

        return x
