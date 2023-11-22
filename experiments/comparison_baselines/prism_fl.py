import os
import time
import torch
import numpy as np
from arguments import arguments
from utils import DatasetSplitter, step_lr, EarlyStopping, calculate_params, aggregate
from PriSM import VisionTransformer_Orth, VisionTransformer

import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader, random_split, Dataset
from torchvision import datasets, transforms, models
from torch.optim.lr_scheduler import ExponentialLR
from torch.utils.data import Subset

# import timm
import copy
import argparse
from sklearn.metrics import f1_score
from transformers import ViTForImageClassification
from utils import (
    step_lr,
    EarlyStopping,
)
import random

from raffm import RaFFM


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
            sub_dataset = Subset(self.dataset, indices)
            sub_datasets.append(sub_dataset)
        return sub_datasets

    def _split_without_replacement(self, n):
        indices = list(range(len(self.dataset)))
        random.shuffle(indices)
        size = len(indices) // n
        sub_datasets = [indices[i * size : (i + 1) * size] for i in range(n)]
        sub_datasets[-1].extend(indices[n * size :])

        client_datasets = [Subset(self.dataset, indices) for indices in sub_datasets]

        return client_datasets


def eval(model, val_dataloader, device):
    model.to(device)
    model.eval()

    correct_val = 0
    all_preds, all_labels = [], []

    for inputs, labels in val_dataloader:
        inputs, labels = inputs.to(device), labels.to(device)

        with torch.no_grad():
            outputs = model(inputs).logits
            preds = outputs.argmax(-1)
            correct_val += torch.sum(preds == labels.data)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    val_accuracy = correct_val.double() / len(val_dataloader.dataset)
    val_f1_score = f1_score(all_labels, all_preds, average="weighted")
    return val_accuracy, val_f1_score


# class Subset(Dataset):
#     def __init__(self, dataset, indices):
#         self.dataset = dataset
#         self.indices = indices

#     def __getitem__(self, idx):
#         image, label = self.dataset[self.indices[idx]]
#         return image, label

#     def __len__(self):
#         return len(self.indices)


def get_k_shot_indices(dataset, k, num_classes, num_clients, replace=False):
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


def set_parameter_requires_grad(model, feature_extracting, model_name):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False

        if model_name == "resnet":
            for param in model.fc.parameters():
                param.requires_grad = True
        elif model_name == "vit":
            for param in model.head.parameters():
                param.requires_grad = True


def initialize_model(model_name, num_classes, feature_extract, use_pretrained=True):
    if model_name == "resnet":
        model_ft = models.resnet50(pretrained=use_pretrained)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_classes)
        # set_parameter_requires_grad(model_ft, feature_extract, model_name)

    elif model_name == "vit":
        # model_ft = timm.create_model("vit_base_patch16_224", pretrained=use_pretrained, num_classes=num_classes)
        model_ft = ViTForImageClassification.from_pretrained(
            "google/vit-base-patch16-224",
            num_labels=num_classes,
            ignore_mismatched_sizes=True,
        )

        # set_parameter_requires_grad(model_ft, feature_extract, model_name)
    elif model_name == "vit-large":
        model_ft = ViTForImageClassification.from_pretrained(
            "google/vit-large-patch16-224-in21k",
            num_labels=num_classes,
            ignore_mismatched_sizes=True,
        )
        # model_ft = timm.create_model("vit_large_patch16_224", pretrained=use_pretrained, num_classes=num_classes)

        # set_parameter_requires_grad(model_ft, feature_extract, model_name)

    return model_ft


def load_data(dataset_name, k_shot, transform, num_clients):
    if dataset_name == "cifar100":
        train_dataset = datasets.CIFAR100(
            root="./data", train=True, download=True, transform=transform
        )
        val_dataset = datasets.CIFAR100(
            root="./data", train=False, download=True, transform=transform
        )
        num_classes = 100

    elif dataset_name == "cifar10":
        train_dataset = datasets.CIFAR10(
            root="./data", train=True, download=True, transform=transform
        )
        val_dataset = datasets.CIFAR10(
            root="./data", train=False, download=True, transform=transform
        )
        num_classes = 10

    elif dataset_name == "flowers102":
        train_dataset = datasets.Flowers102(
            root="./data", split="train", download=True, transform=transform
        )
        num_classes = 102
        val_dataset = datasets.Flowers102(
            root="./data", split="test", download=True, transform=transform
        )

    elif dataset_name == "Caltech101":
        transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.Grayscale(num_output_channels=1),
                transforms.ToTensor(),
                transforms.Normalize((0.5), (0.5)),
            ]
        )

        dataset = datasets.Caltech101(root="./data", download=True, transform=transform)

        train_dataset, val_dataset = random_split(
            dataset, [int(0.8 * len(dataset)), len(dataset) - int(0.8 * len(dataset))]
        )
        num_classes = 101

    elif dataset_name == "Food101":
        train_dataset = datasets.Food101(
            root="./data", split="train", download=True, transform=transform
        )
        num_classes = 101
        val_dataset = datasets.Food101(
            root="./data", split="test", download=True, transform=transform
        )

    # replace = False
    # if dataset_name == "flowers102":
    #     replace = True

    # indices = get_k_shot_indices(
    #     train_dataset, k_shot, num_classes, num_clients, replace=replace
    # )
    # client_datasets = [Subset(train_dataset, indices) for indices in indices]

    return train_dataset, val_dataset, num_classes


def federated_learning(
    args, model, local_dataloaders, val_dataloader, criterion, device="cuda"
):
    early_stopping = EarlyStopping(patience=5, verbose=True)

    global_model = RaFFM(model.to("cpu"))
    best_acc = 0.0
    best_f1 = 0.0
    for round in range(args.num_rounds):
        local_models = []
        lr = step_lr(args.lr, round, 5, 0.98)

        np.random.seed(int(time.time()))  # Set the seed to the current time
        client_indices = np.random.choice(
            len(local_dataloaders),
            size=int(0.1 * len(local_dataloaders)),
            replace=False,
        )
        if args.spp:
            global_model.salient_parameter_prioritization()

        avg_trainable_params = 0

        # Train the model on each client's dataset
        # for local_dataloader in local_dataloaders:
        for idx, client_id in enumerate(client_indices):
            local_dataloader = local_dataloaders[client_id]
            print(f"Training client {client_id} in communication round {round}")

            if args.method == "raffm":
                if idx == 0:
                    local_model = copy.deepcopy(global_model.model)
                    local_model_params = global_model.total_params

                else:
                    (
                        local_model,
                        local_model_params,
                    ) = global_model.random_resource_aware_model()
            elif args.method == "vanilla":
                local_model = copy.deepcopy(global_model.model)
                local_model_params = global_model.total_params

            avg_trainable_params += local_model_params

            print(
                f"Client {client_id} local model has {local_model_params} parameters out of {global_model.total_params} parameters in communication round {round}"
            )

            local_model = local_model.to(device)
            # local_model = nn.DataParallel(local_model)
            local_optimizer = optim.Adam(local_model.parameters(), lr=lr)
            lr_scheduler = ExponentialLR(local_optimizer, gamma=0.95)
            # Fine-tune the local model on the client's dataset
            for epoch in range(args.num_local_epochs):
                local_model.train()
                for inputs, labels in local_dataloader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    local_optimizer.zero_grad()
                    loss = local_model(inputs, labels=labels).loss
                    loss.backward()
                    local_optimizer.step()
                lr_scheduler.step()

            print("Eval local model\n")
            val_accuracy, val_f1_score = eval(local_model, val_dataloader, device)
            print(f"Local Validation Accuracy: {val_accuracy:.4f}")
            print(f"Local Validation F1 Score: {val_f1_score:.4f}")

            local_model.to("cpu")
            local_models.append(local_model)
            print("Training finished!")
        print("Eval global model finished!")
        global_model.aggregate(local_models)
        val_accuracy, val_f1_score = eval(global_model.model, val_dataloader, device)

        global_model.model.to("cpu")
        if val_accuracy > best_acc:
            best_acc = val_accuracy
            global_model.model.save_pretrained(
                os.path.join(args.save_dir, args.dataset, "best_model")
            )
        if val_f1_score > best_f1:
            best_f1 = val_f1_score

        print(f"Validation Accuracy: {val_accuracy:.4f}")
        print(f"Validation F1 Score: {val_f1_score:.4f}")

        print(f"Best Validation Accuracy: {best_acc:.4f}")
        print(f"Best Validation F1 Score: {best_f1:.4f}")
        early_stopping(val_f1_score)
        if early_stopping.has_converged():
            print("Model has converged. Stopping training.")
            break
    return global_model


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])
    # define imagenet transforms
    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ]
    )

    if args.method == "centralized":
        args.num_clients = 1

    train_dataset, val_dataset, num_classes = load_data(
        args.dataset, args.k_shot, transform, args.num_clients
    )
    splitter = DatasetSplitter(train_dataset, seed=123)

    local_datasets = splitter.split(args.num_clients, replacement=False)

    train_dataloaders = [
        DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        for train_dataset in local_datasets
    ]
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    # Load the pretrained ResNet-50 model
    model = initialize_model(args.model, num_classes, True, use_pretrained=True)
    model = model.to(device)
    if args.ckpt:
        model.from_pretrained(args.ckpt)
    # model = nn.DataParallel(model)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=5e-5)

    global_model = federated_learning(
        args, model, train_dataloaders, val_dataloader, criterion, device="cuda"
    )
    global_model.model.save_pretrained(
        os.path.join(args.save_dir, args.dataset, "final")
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Few-shot learning with pre-trained models"
    )
    parser.add_argument(
        "--method",
        type=str,
        default="raffm",
        choices=["vanilla", "raffm"],
        help="Method to use (centralized or federated)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="vit",
        choices=["resnet", "vit", "vit-large"],
        help="Model architecture to use (resnet or vit)",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default="log_vit",
        help="dir save the model",
    )

    parser.add_argument(
        "--ckpt",
        type=str,
        default=None,
        help="dir save the model",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="cifar100",
        choices=["cifar100", "flowers102", "Caltech101", "cifar10", "Food101"],
        help="Dataset to use (currently only cifar100 is supported)",
    )
    parser.add_argument(
        "--k_shot",
        type=int,
        default=50,
        help="Number of samples per class for few-shot learning",
    )
    parser.add_argument(
        "--num_epochs", type=int, default=10, help="Number of training epochs"
    )
    parser.add_argument(
        "--num_clients",
        type=int,
        default=100,
        help="Number of clients in a federated learning setting",
    )
    parser.add_argument(
        "--lr", type=float, default=5e-5, help="Learning rate for the optimizer"
    )
    parser.add_argument(
        "--step_size",
        type=int,
        default=10,
        help="Step size for the learning rate scheduler",
    )
    parser.add_argument(
        "--num_local_epochs",
        type=int,
        default=5,
        help="Number of local epochs for each client in a federated learning setting",
    )
    parser.add_argument(
        "--num_rounds",
        type=int,
        default=100,
        help="Number of communication rounds for federated learning",
    )
    parser.add_argument(
        "--spp", action="store_true", help="salient parameter prioritization"
    )
    parser.add_argument(
        "--batch_size", type=int, help="per device batch size", default=64
    )

    args = parser.parse_args()
    main(args)
