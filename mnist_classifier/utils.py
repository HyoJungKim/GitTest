import torch


def accuracy(output, target):
    """
    Calculate accuracy for classification

    Args:
        output: Model predictions (logits), shape (batch_size, num_classes)
        target: Ground truth labels, shape (batch_size,)

    Returns:
        Accuracy as a percentage (float)
    """
    with torch.no_grad():
        pred = output.argmax(dim=1, keepdim=True)
        correct = pred.eq(target.view_as(pred)).sum().item()
        acc = 100. * correct / target.size(0)
    return acc
