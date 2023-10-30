import numpy as np
import time
import os
import time
import torch
import numpy as np
from datasets import load_dataset, load_metric
import functools

# import timm
import copy
import argparse
from transformers import (
    ViTForImageClassification,
    ViTImageProcessor,
    TrainingArguments,
    Trainer,
)
from raffm.utils import DatasetSplitter, step_lr, EarlyStopping
import random

from raffm import RaFFM


def server_updates(args, global_model, round):
    if round == 0:
        pass

    if args.spp:
        global_model.salient_parameter_prioritization()
    pass


def local_updates(model, training_args):
    trainer = Trainer(
        model=local_model,
        args=training_args,
        data_collator=collate_fn,
        compute_metrics=compute_metrics,
        train_dataset=local_dataset,
        eval_dataset=val_dataset,
        # tokenizer=processor,
    )
    train_results = trainer.train()
    pass


def federated_learning(
    server, clients, communication_rounds, n_clients, participate_rate=0.1, sampler=None
):
    early_stopping = EarlyStopping(patience=5, verbose=True)

    best_acc = 0.0
    best_f1 = 0.0

    for round in range(communication_rounds):
        local_models = []
        lr = step_lr(args.lr, round, 5, 0.98)

        np.random.seed(int(time.time()))  # Set the seed to the current time

        if sampler:
            client_indices = sampler(n_clients, participate_rate)
        else:
            client_indices = np.random.choice(
                n_clients,
                size=int(0.1 * n_clients),
                replace=False,
            )

        if args.spp:
            global_model.salient_parameter_prioritization()

        avg_trainable_params = 0

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
                    ) = global_model.random_resource_aware_model()
            elif args.method == "vanilla":
                local_model = copy.deepcopy(global_model.model)
                local_model_params = global_model.total_params

            avg_trainable_params += local_model_params

            print(
                f"Client {client_id} local model has {local_model_params} parameters out of {global_model.total_params} parameters in communication round {round}"
            )

            training_args = TrainingArguments(
                output_dir="./log/debug",
                per_device_train_batch_size=16,
                evaluation_strategy="steps",
                num_train_epochs=4,
                save_steps=100,
                eval_steps=100,
                logging_steps=10,
                learning_rate=2e-4,
                save_total_limit=2,
                remove_unused_columns=False,
                push_to_hub=False,
                report_to="tensorboard",
                load_best_model_at_end=True,
            )

            trainer = Trainer(
                model=local_model,
                args=training_args,
                data_collator=collate_fn,
                compute_metrics=compute_metrics,
                train_dataset=local_dataset,
                eval_dataset=val_dataset,
                # tokenizer=processor,
            )
            train_results = trainer.train()

            print(f"Eval local model {client_id}\n")
            metrics = trainer.evaluate(val_dataset)
            trainer.log_metrics("eval", metrics)
            trainer.save_metrics("eval", metrics)

            local_model.to("cpu")
            local_models.append(local_model)
            print("Training finished!")

        print(f"Eval global model in communication round {round}")
        global_model.aggregate(local_models)
        trainer = Trainer(
            model=global_model.model,
            args=training_args,
            data_collator=collate_fn,
            compute_metrics=compute_metrics,
            train_dataset=local_dataset,
            eval_dataset=val_dataset,
            # tokenizer=processor,
        )
        metrics = trainer.evaluate(val_dataset)
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)
        val_accuracy, val_f1_score = metrics["eval_accuracy"], metrics["eval_f1"]

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

        if test_dataset:
            metrics = trainer.evaluate(test_dataset, metric_key_prefix="test")
            trainer.log_metrics("eval", metrics)
            trainer.save_metrics("eval", metrics)

        early_stopping(val_f1_score)
        if early_stopping.has_converged():
            print("Model has converged. Stopping training.")
            break
    return global_model
