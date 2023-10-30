import torch
from torch.nn import Parameter
import numpy as np


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
        else:
            return self._split_without_replacement(n)

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


def get_k_shot_indice_vit(dataset, k, num_classes, num_clients, replace=False):
    class_examples = [[] for _ in range(num_classes)]

    for idx, (_, label) in enumerate(dataset):
        class_examples[label].append(idx)

    client_indices = []
    for _ in range(num_clients):
        indices = []
        for class_idx in range(num_classes):
            indices += np.random.choice(
                class_examples[class_idx], k, replace=replace
            ).tolist()
        client_indices.append(indices)

    return client_indices


def k_shot_data(dataset, num_clients, k_shot, dataset_name):
    datasets = []

    if dataset_name in ["sst2", "mrpc", "qnli", "mnli", "qqp", "rte", "cola"]:
        class_examples = []
        num_classes = 3 if dataset_name == "mnli" else 2

        for i in range(num_classes):
            class_examples.append(
                dataset.filter(lambda example: example["label"] == i).shuffle()
            )

        examples_per_client = k_shot * num_classes

        for i in range(num_clients):
            subsets = []

            for j in range(num_classes):
                start = i * k_shot
                end = (i + 1) * k_shot
                subsets.append(class_examples[j].select(range(start, end)))

            client_dataset = concatenate_datasets(subsets)
            datasets.append(client_dataset)

    elif dataset_name == "stsb":
        dataset = dataset.shuffle()
        examples_per_client = (
            k_shot * 2
        )  # Assuming an equal number of examples for high and low scores

        for i in range(num_clients):
            start = i * examples_per_client
            end = (i + 1) * examples_per_client

            client_dataset = dataset.select(range(start, end))
            datasets.append(client_dataset)

    return datasets
