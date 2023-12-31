import os
import time
import torch
import numpy as np
from datasets import load_dataset
import functools
import evaluate
from torch.utils.tensorboard import SummaryWriter
import copy
from transformers import (
    ViTForImageClassification,
    ViTImageProcessor,
    TrainingArguments,
    Trainer,
)
from peft import LoraConfig, get_peft_model, PeftModel, PeftConfig
from utils import DatasetSplitter, step_lr, EarlyStopping, calculate_params, aggregate
from PruneFL import prunefl
from arguments import arguments



def compute_metrics(eval_pred):
    accuracy_metric = evaluate.load("accuracy")
    f1_metric = evaluate.load("f1")

    accuracy = accuracy_metric.compute(
        predictions=np.argmax(eval_pred.predictions, axis=1),
        references=eval_pred.label_ids,
    )
    f1 = f1_metric.compute(
        predictions=np.argmax(eval_pred.predictions, axis=1),
        references=eval_pred.label_ids,
        average="weighted",
    )

    return {"accuracy": accuracy["accuracy"], "f1": f1["f1"]}


def federated_learning(
    args, global_model, local_datasets, val_dataset, test_dataset=None
):
    early_stopping = EarlyStopping(patience=5, verbose=True)

    writer = SummaryWriter(os.path.join(args.save_dir, args.dataset))
    best_acc = 0.0
    best_f1 = 0.0

    for round in range(args.num_rounds):
        global_model, model_sparsity = prunefl(global_model, rank=0.8, threshold=1e-2)
        print(
            f"Starting communication round {round}, the model sparsity  after pruning {model_sparsity}"
        )
        local_models = []
        lr = step_lr(args.lr, round, args.step_size, 0.98)

        np.random.seed(int(time.time()))  # Set the seed to the current time

        client_indices = np.random.choice(
            len(local_datasets),
            size=int(0.1 * len(local_datasets)),
            replace=False,
        )

        avg_trainable_params = 0
        processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224")
        # Train the model on each client's dataset
        # for local_dataloader in local_dataloaders:
        for idx, client_id in enumerate(client_indices):
            local_dataset = local_datasets[client_id]
            print(f"Training client {client_id} in communication round {round}")

            local_model = copy.deepcopy(global_model)
            local_trainable_params = calculate_params(local_model)
            avg_trainable_params += local_trainable_params

            print(
                f"Client {client_id} local backbone model has {local_trainable_params} parameters in communication round {round}"
            )
            local_model.print_trainable_parameters()

            training_args = TrainingArguments(
                output_dir=os.path.join(args.save_dir, "clients", str(client_id)),
                per_device_train_batch_size=args.batch_size,
                per_device_eval_batch_size=args.batch_size,
                evaluation_strategy="no",
                save_strategy="no",
                num_train_epochs=args.num_local_epochs,
                learning_rate=lr,
                # save_total_limit=2,
                remove_unused_columns=False,
                push_to_hub=False,
                report_to="none",
                label_names=["labels"],
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

            if round > args.num_rounds - 10:
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

        avg_trainable_params = avg_trainable_params / len(client_indices)
        avg_model_param = avg_model_param / len(client_indices)
        writer.add_scalar(
            "global/tranable_params",
            avg_trainable_params,
            round,
        )
        writer.add_scalar(
            "global/backbone_params",
            avg_model_param,
            round,
        )

        print(
            f"Communication round {round} federated learning finished. \n Average trainable parameters:{avg_trainable_params}.\n Eval global model."
        )
        aggregate(global_model, local_models)

        training_args = TrainingArguments(
            output_dir=os.path.join(args.save_dir, "global"),
            per_device_train_batch_size=args.batch_size,
            per_device_eval_batch_size=args.batch_size,
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
            report_to="none",
            label_names=["labels"],
            # load_best_model_at_end=True,
        )

        trainer = Trainer(
            model=global_model,
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

        global_model.to("cpu")
        if val_accuracy > best_acc:
            best_acc = val_accuracy
            global_model.save_pretrained(
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
        test_size=0.2, stratify_by_column="label", seed=123
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
            os.path.join(ckpt_path, "elastic_space.json")
            if os.path.exists(os.path.join(ckpt_path, "elastic_space.json"))
            else args.elastic_config
        )

    else:
        ckpt_path = model_name
        elastic_config = args.elastic_config

    model = ViTForImageClassification.from_pretrained(
        ckpt_path,
        num_labels=len(labels),
        id2label={str(i): c for i, c in enumerate(labels)},
        label2id={c: str(i) for i, c in enumerate(labels)},
        ignore_mismatched_sizes=True,
    )
    if args.peft:
        if args.adapter_ckpt:
            model = PeftModel.from_pretrained(model, args.adapter_ckpt)
            elastic_config = (
                os.path.join(args.adapter_ckpt, "elastic.pt")
                if os.path.exists(os.path.join(args.adapter_ckpt, "elastic.pt"))
                else None
            )
            config = None
        else:
            config = LoraConfig(
                r=16,
                lora_alpha=16,
                target_modules=["query", "key", "value"],
                lora_dropout=0.1,
                bias="none",
                modules_to_save=["classifier"],
            )
            print(f"[Warning]: default PEFT method LoRA, default configure:", config)

            model = get_peft_model(model, config)
        model.print_trainable_parameters()

    # global_model = RaFFM(model.to("cpu"), elastic_config)
    global_model = model
    global_model = federated_learning(
        args, global_model, local_datasets, prepared_ds["validation"]
    )
    # global_model.save_ckpt(os.path.join(args.save_dir, args.dataset, "final"))


if __name__ == "__main__":
    args = arguments()
    main(args)
