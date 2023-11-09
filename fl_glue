import os
import time
import numpy as np
from datasets import load_dataset, concatenate_datasets
import functools
import evaluate
from torch.utils.tensorboard import SummaryWriter
import copy
import argparse
from transformers import (
    DistilBertForSequenceClassification,
    DistilBertTokenizerFast,
    Trainer,
    TrainingArguments,
    RobertaTokenizerFast,
    T5Tokenizer,
    AutoTokenizer,
    BertForSequenceClassification,
    BertTokenizerFast,
    DistilBertForSequenceClassification,
    RobertaForSequenceClassification,
    T5ForConditionalGeneration,
    Trainer,
    TrainingArguments,
)
from raffm.utils import DatasetSplitter, step_lr, EarlyStopping
from raffm import RaFFM
from arguments import arguments


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
                model=local_model,
                args=training_args,
                compute_metrics=compute_metrics,
                train_dataset=local_dataset,
                eval_dataset=val_dataset,
            )
            train_results = trainer.train()

            if round > 95:
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
        writer.add_scalar(
            "global/params",
            avg_trainable_params,
            round,
        )
        print(
            f"Communication round {round} federated learning finished. \n Average trainable parameters:{avg_trainable_params}.\n Eval global model."
        )
        global_model.aggregate(local_models)

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
            model=global_model.model,
            args=training_args,
            compute_metrics=compute_metrics,
            train_dataset=local_dataset,
            eval_dataset=val_dataset,
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


def tokenize_function(examples, tokenizer):
    if "sentence" in examples.keys() and "question" not in examples.keys():
        return tokenizer(
            examples["sentence"],
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
    elif "premise" in examples.keys() and "hypothesis" in examples.keys():
        return tokenizer(
            examples["premise"],
            examples["hypothesis"],
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
    elif "question1" in examples.keys() and "question2" in examples.keys():
        return tokenizer(
            examples["question1"],
            examples["question2"],
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
    elif "question" in examples.keys() and "sentence" in examples.keys():
        return tokenizer(
            examples["question"],
            examples["sentence"],
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
    elif "sentence1" in examples.keys() and "sentence2" in examples.keys():
        return tokenizer(
            examples["sentence1"],
            examples["sentence2"],
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )


def main(args):
    num_classes = {
        "mnli": 3,
        "qqp": 2,
        "qnli": 2,
        "sst2": 2,
        "stsb": 1,
        "mrpc": 2,
        "rte": 2,
        "cola": 2,
    }

    if args.model == "distilbert":
        model_name = "distilbert-base-uncased"
        tokenizer = DistilBertTokenizerFast.from_pretrained(model_name)
        global_model = DistilBertForSequenceClassification.from_pretrained(
            model_name, num_labels=num_classes[args.dataset]
        )

    elif args.model == "roberta":
        model_name = "roberta-base"
        tokenizer = RobertaTokenizerFast.from_pretrained(model_name)
        global_model = RobertaForSequenceClassification.from_pretrained(
            model_name, num_labels=num_classes[args.dataset]
        )

    elif args.model == "t5":
        model_name = "t5-small"  # You can also use "t5-base" or other T5 variants
        tokenizer = T5Tokenizer.from_pretrained(model_name)
        global_model = T5ForConditionalGeneration.from_pretrained(model_name)
    elif args.model == "bert-base":
        model_name = "bert-base-uncased"
        tokenizer = BertTokenizerFast.from_pretrained(model_name)
        global_model = BertForSequenceClassification.from_pretrained(
            model_name, num_labels=num_classes[args.dataset]
        )

    elif args.model == "bert-large":
        model_name = "bert-large-uncased-whole-word-masking"
        tokenizer = BertTokenizerFast.from_pretrained(model_name)
        global_model = BertForSequenceClassification.from_pretrained(
            model_name, num_labels=num_classes[args.dataset]
        )

    # load data and preprocess
    dataset = load_dataset("glue", args.dataset)

    if args.dataset == "mnli":
        dataset["validation"] = concatenate_datasets(
            [dataset["validation_matched"], dataset["validation_mismatched"]]
        )

    train_dataset = dataset["train"]
    val_dataset = dataset["validation"]

    tokenize_val_dataset = val_dataset.map(
        lambda examples: tokenize_function(examples, tokenizer),
        batched=True,
    )

    tokenized_train_dataset = train_dataset.map(
        lambda examples: tokenize_function(examples, tokenizer),
        batched=True,
    )

    splitter = DatasetSplitter(tokenized_train_dataset, seed=123)

    tokenized_local_datasets = splitter.split(
        args.num_clients, k_shot=args.k_shot, replacement=False
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

    global_model = RaFFM(global_model.to("cpu"), elastic_config)
    global_model = federated_learning(
        args, global_model, tokenized_local_datasets, tokenize_val_dataset
    )
    global_model.save_ckpt(os.path.join(args.save_dir, args.dataset, "final"))


if __name__ == "__main__":
    args = arguments()
    main(args)
