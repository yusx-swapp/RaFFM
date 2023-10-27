import torch
from torch.nn import Parameter


@staticmethod
def count_non_zero_params(model) -> int:
    """
    Count the number of non-zero parameters in a PyTorch model.

    Args:
    - model (nn.Module): A PyTorch model.

    Returns:
    - int: Number of non-zero parameters.
    """
    return sum((param != 0).sum().item() for param in model.parameters())


@staticmethod
def calculate_params(model):
    """calculate the number of trainable parameters in the model
    Args:
        model: the model to be evaluated
    Returns:
        total_trainable_params: the number of trainable parameters in the model
        total_params: the number of parameters in the model
        percentage: the percentage of trainable parameters in the model
    """

    millions = 1000000
    total_params = 0
    for name, module in model.named_modules():
        if hasattr(module, "weight") and isinstance(module.weight, Parameter):
            total_params += torch.prod(torch.tensor(module.weight.size())).item()

    return total_params / millions
