import torch
from torch.nn import Parameter
import numpy as np
import random
from datasets import concatenate_datasets, Dataset


@staticmethod
def step_lr(initial_lr, epoch, decay_step, decay_rate):
    return initial_lr * (decay_rate ** (epoch // decay_step))


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


class EarlyStopping:
    def __init__(self, patience=10, verbose=False, delta=0):
        """
        Args:
            patience (int): How long to wait after last improvement.
                            Default: 10
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.delta = delta

    def __call__(self, score):
        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0

    def has_converged(self):
        return self.early_stop


class DatasetSplitter:
    def __init__(self, dataset, seed=None):
        self.dataset = dataset
        if seed is not None:
            random.seed(seed)

    def split(self, n, replacement=False):
        if replacement:
            return self._split_with_replacement(n)
        return self._split_without_replacement(n)

    def k_shot(self, n, k_shot=12, replacement=False):
        if replacement:
            return self._split_k_shot_with_replacement(n, k_shot)
        return self._split_k_shot(n, k_shot)

    def _split_with_replacement(self, n):
        size = len(self.dataset) // n
        sub_datasets = []
        for _ in range(n):
            indices = random.choices(range(len(self.dataset)), k=size)
            sub_dataset = Dataset.from_dict(self.dataset[indices])
            sub_datasets.append(sub_dataset)
        return sub_datasets

    def _split_without_replacement(self, n):
        indices = list(range(len(self.dataset)))
        random.shuffle(indices)
        size = len(indices) // n
        sub_datasets = [indices[i * size : (i + 1) * size] for i in range(n)]
        sub_datasets[-1].extend(indices[n * size :])
        sub_datasets = [
            Dataset.from_dict(self.dataset[sub_dataset]) for sub_dataset in sub_datasets
        ]

        return sub_datasets

    def _split_k_shot(self, n, k_shot):
        # Step 1: Group the dataset by class
        class_groups = {}
        for i in range(len(self.dataset)):
            label = self.dataset[i]["label"]
            if label not in class_groups:
                class_groups[label] = []
            class_groups[label].append(i)

        # Check if each class has enough samples
        for label, indices in class_groups.items():
            if len(indices) < n * k_shot:
                raise ValueError(
                    f"Not enough samples in class {label} for a {k_shot}-shot split into {n} parts"
                )

        # Step 2 & 3: For each class, select n * k samples and split them
        sub_datasets_indices = [[] for _ in range(n)]
        for label, indices in class_groups.items():
            selected_indices = random.sample(indices, n * k_shot)
            for i in range(n):
                sub_datasets_indices[i].extend(
                    selected_indices[i * k_shot : (i + 1) * k_shot]
                )

        # Step 4: Combine the corresponding sub-datasets from all classes
        sub_datasets = [
            Dataset.from_dict(self.dataset[indices]) for indices in sub_datasets_indices
        ]
        return sub_datasets

    def _split_k_shot_with_replacement(self, n, k_shot):
        # Step 1: Group the dataset by class
        class_groups = {}
        for i in range(len(self.dataset)):
            label = self.dataset[i]["label"]
            if label not in class_groups:
                class_groups[label] = []
            class_groups[label].append(i)

        # Step 2: For each class, select n * k samples with replacement and split them
        sub_datasets_indices = [[] for _ in range(n)]
        for label, indices in class_groups.items():
            for i in range(n):
                selected_indices = random.choices(indices, k=k_shot)
                sub_datasets_indices[i].extend(selected_indices)

        # Step 3: Combine the corresponding sub-datasets from all classes
        sub_datasets = [
            Dataset.from_dict(self.dataset[indices]) for indices in sub_datasets_indices
        ]
        return sub_datasets
