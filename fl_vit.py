import os
import time
import torch
import numpy as np
from datasets import load_dataset
import functools
import evaluate
from torch.utils.tensorboard import SummaryWriter
import copy
import argparse
from transformers import (
    ViTForImageClassification,
    ViTImageProcessor,
    TrainingArguments,
    Trainer,
)
import peft
from raffm.utils import DatasetSplitter, step_lr, EarlyStopping
from raffm import RaFFM


@staticmethod
def compute_metrics(p):
    accuracy_metric = evaluate.load("accuracy")
    f1_metric = evaluate.load("f1")

    accuracy = accuracy_metric.compute(
        predictions=np.argmax(p.predictions, axis=1), references=p.label_ids
    )
    f1 = f1_metric.compute(
        predictions=np.argmax(p.predictions, axis=1),
        references=p.label_ids,
        average="weighted",
    )

    return {"accuracy": accuracy["accuracy"], "f1": f1["f1"]}


def federated_learning(
    args, global_model: RaFFM, local_datasets, val_dataset, test_dataset=None
):
    early_stopping = EarlyStopping(patience=5, verbose=True)

    writer = SummaryWriter(os.path.join(args.save_dir, args.dataset))
    best_acc = 0.0
    best_f1 = 0.0

    for round in range(args.num_rounds):
        local_models = []
        lr = step_lr(args.lr, round, args.step_size, 0.98)

        np.random.seed(int(time.time()))  # Set the seed to the current time

        client_indices = np.random.choice(
            len(local_datasets),
            size=int(0.1 * len(local_datasets)),
            replace=False,
        )

        if args.spp:
            global_model.salient_parameter_prioritization()
        avg_trainable_params = 0
        processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224")
        # Train the model on each client's dataset
        # for local_dataloader in local_dataloaders:
        for idx, client_id in enumerate(client_indices):
            local_dataset = local_datasets[client_id]
            print(f"Training client {client_id} in communication round {round}")

            if args.method == "raffm":
                if idx == 0:
                    local_model = copy.deepcopy(global_model.model)
                    local_model_params = global_model.total_params

                else:
                    (
                        local_model,
                        local_model_params,
                        arc_config,
                    ) = global_model.random_resource_aware_model()
            elif args.method == "vanilla":
                local_model = copy.deepcopy(global_model.model)
                local_model_params = global_model.total_params

            avg_trainable_params += local_model_params

            print(
                f"Client {client_id} local model has {local_model_params} parameters out of {global_model.total_params} parameters in communication round {round}"
            )
            writer.add_scalar(
                str(client_id) + "/params",
                local_model_params,
                round,
            )
            training_args = TrainingArguments(
                output_dir=os.path.join(args.save_dir, "clients", str(client_id)),
                per_device_train_batch_size=args.batch_size,
                evaluation_strategy="no",
                save_strategy="no",
                num_train_epochs=args.num_local_epochs,
                # save_steps=100,
                # eval_steps=100,
                # logging_steps=10,
                learning_rate=lr,
                # save_total_limit=2,
                remove_unused_columns=False,
                push_to_hub=False,
                # report_to="tensorboard",
                # load_best_model_at_end=True,
            )

            trainer = Trainer(
                model=local_model,
                args=training_args,
                data_collator=collate_fn,
                compute_metrics=compute_metrics,
                train_dataset=local_dataset,
                eval_dataset=val_dataset,
                tokenizer=processor,
            )
            train_results = trainer.train()

            if round > 50:
                print(f"Eval local model {client_id}\n")
                metrics = trainer.evaluate(val_dataset)

                trainer.log_metrics("eval", metrics)
                val_accuracy, val_f1_score = (
                    metrics["eval_accuracy"],
                    metrics["eval_f1"],
                )
                writer.add_scalar(
                    str(client_id) + "/eval_accuracy",
                    val_accuracy,
                    round,
                )
                writer.add_scalar(
                    str(client_id) + "/eval_f1",
                    val_f1_score,
                    round,
                )

            local_model.to("cpu")
            local_models.append(local_model)
            print("Local training finished!")

        print(f"Eval global model in communication round {round}")
        global_model.aggregate(local_models)

        training_args = TrainingArguments(
            output_dir=os.path.join(args.save_dir, "global"),
            per_device_train_batch_size=args.batch_size,
            evaluation_strategy="no",
            save_strategy="no",
            num_train_epochs=args.num_local_epochs,
            # save_steps=100,
            # eval_steps=100,
            # logging_steps=10,
            learning_rate=lr,
            # save_total_limit=2,
            remove_unused_columns=False,
            push_to_hub=False,
            # report_to="tensorboard",
            # load_best_model_at_end=True,
        )

        trainer = Trainer(
            model=global_model.model,
            args=training_args,
            data_collator=collate_fn,
            compute_metrics=compute_metrics,
            train_dataset=local_dataset,
            eval_dataset=val_dataset,
            tokenizer=processor,
        )
        metrics = trainer.evaluate(val_dataset)
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)
        val_accuracy, val_f1_score = metrics["eval_accuracy"], metrics["eval_f1"]

        writer.add_scalar(
            "global/eval_accuracy",
            val_accuracy,
            round,
        )
        writer.add_scalar(
            "global/eval_f1",
            val_f1_score,
            round,
        )

        global_model.model.to("cpu")
        if val_accuracy > best_acc:
            best_acc = val_accuracy
            global_model.model.save_pretrained(
                os.path.join(args.save_dir, args.dataset, "best_model")
            )

            global_model.save_ckpt(
                os.path.join(args.save_dir, args.dataset, "best_model")
            )
        if val_f1_score > best_f1:
            best_f1 = val_f1_score
        writer.add_scalar(
            "global/best_accuracy",
            best_acc,
            round,
        )
        writer.add_scalar(
            "global/best_f1",
            best_f1,
            round,
        )

        print(f"Best Validation Accuracy: {best_acc:.4f}")
        print(f"Best Validation F1 Score: {best_f1:.4f}")

        if test_dataset:
            metrics = trainer.evaluate(test_dataset, metric_key_prefix="test")
            trainer.log_metrics("eval", metrics)
            trainer.save_metrics("eval", metrics)

        early_stopping(val_f1_score)
        if early_stopping.has_converged():
            print("Model has converged. Stopping training.")
            break
    return global_model


def collate_fn(batch):
    return {
        "pixel_values": torch.stack([x["pixel_values"] for x in batch]),
        "labels": torch.tensor([x["labels"] for x in batch]),
    }


def transform(example_batch, processor):
    # Take a list of PIL images and turn them to pixel values
    inputs = processor([x for x in example_batch["img"]], return_tensors="pt")

    # Don't forget to include the labels!
    inputs["labels"] = example_batch["label"]
    return inputs


def main(args):
    if args.model == "vit":
        model_name = "google/vit-base-patch16-224-in21k"
    elif args.model == "vit-large":
        model_name = "google/vit-large-patch16-224-in21k"

    # load data and preprocess
    dataset = load_dataset(args.dataset)
    if args.dataset == "cifar100":
        dataset = dataset.rename_column("fine_label", "label")

    train_val = dataset["train"].train_test_split(
        test_size=0.2, stratify_by_column="label"
    )
    dataset["train"] = train_val["train"]
    dataset["validation"] = train_val["test"]
    labels = dataset["train"].features["label"].names

    processor = ViTImageProcessor.from_pretrained(model_name)
    prepared_ds = dataset.with_transform(
        functools.partial(transform, processor=processor)
    )

    splitter = DatasetSplitter(dataset["train"], seed=123)

    local_datasets = splitter.split(
        args.num_clients, k_shot=args.k_shot, replacement=False
    )

    for i, local_data in enumerate(local_datasets):
        local_datasets[i] = local_data.with_transform(
            functools.partial(transform, processor=processor)
        )

    # load/initialize global model and convert to raffm model
    if args.resume_ckpt:
        ckpt_path = args.resume_ckpt
        elastic_config = (
            os.path.join(ckpt_path, "elastic.pt")
            if os.path.exists(os.path.join(ckpt_path, "elastic.pt"))
            else None
        )

    else:
        ckpt_path = model_name
        elastic_config = None

    model = ViTForImageClassification.from_pretrained(
        ckpt_path,
        num_labels=len(labels),
        id2label={str(i): c for i, c in enumerate(labels)},
        label2id={c: str(i) for i, c in enumerate(labels)},
        ignore_mismatched_sizes=True,
    )

    global_model = RaFFM(model.to("cpu"), elastic_config)
    global_model = federated_learning(
        args, global_model, local_datasets, prepared_ds["validation"]
    )
    global_model.save_ckpt(os.path.join(args.save_dir, args.dataset, "final"))


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
        "--resume_ckpt",
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
        "--k-shot",
        type=int,
        default=None,
        help="split k-shot local data",
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
        default=5,
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
