import os
import time
import numpy as np
from datasets import load_dataset, concatenate_datasets
import functools
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
from scipy.stats import pearsonr
from sklearn.metrics import accuracy_score, matthews_corrcoef



# set no_deprecation_warning to True to avoid warning messages
def compute_metrics(eval_pred, task):
    predictions, labels = eval_pred
    if task == "stsb":
        pearson_corr, _ = pearsonr(predictions.squeeze(), labels)
        return {"pearson_corr": pearson_corr}
    elif task == "cola":
        probabilities_class_1 = predictions[:, 1]
        # Convert continuous predictions to binary (0 or 1) for the CoLA task
        binary_predictions = (probabilities_class_1 > 0.5).astype(int)
        return {"matthews_corr": matthews_corrcoef(labels, binary_predictions)}
    else:
        predictions = predictions.argmax(-1)
        return {"accuracy": accuracy_score(labels, predictions)}


def evaluate(args, global_model, tokenized_test_dataset):
    # tokenized_test_dataset = test_dataset.map(lambda examples: tokenize_function(examples, tokenizer, args.dataset, args.model), batched=True)

    training_args = TrainingArguments(
        args.log_dir,
        logging_dir=args.log_dir,
        logging_steps=1000,
        save_strategy="no",
        evaluation_strategy="no",
    )

    global_model.to("cuda")  # Move the global model to GPU memory for evaluation
    # global_model = torch.compile(global_model)
    trainer = Trainer(
        model=global_model,
        args=training_args,
    )

    predictions = trainer.predict(tokenized_test_dataset)
    true_labels = tokenized_test_dataset["label"]
    true_labels = np.array(tokenized_test_dataset["label"])

    global_model.to("cpu")  # Move the global model back to CPU memory after evaluation

    if args.dataset == "stsb":
        pearson_corr = compute_metrics(
            (predictions.predictions, true_labels), args.dataset
        )["pearson_corr"]
        print(f"Pearson correlation: {pearson_corr}")
        return pearson_corr
    elif args.dataset == "cola":
        probabilities_class_1 = predictions.predictions[:, 1]
        binary_predictions = (probabilities_class_1 > 0.5).astype(int)
        matthews_corr = matthews_corrcoef(true_labels, binary_predictions)

        print(f"matthews correlation: {matthews_corr}")
        return matthews_corr

    else:
        predicted_labels = predictions.predictions.argmax(-1)
        accuracy = accuracy_score(true_labels, predicted_labels)
        print(f"Accuracy: {accuracy}")
        return accuracy


def federated_learning(
    args, global_model: RaFFM, local_datasets, val_dataset, test_dataset=None
):
    early_stopping = EarlyStopping(patience=5, verbose=True)

    writer = SummaryWriter(os.path.join(args.save_dir, args.dataset))
    best_performance = 0.0

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
                elif idx == 1:
                    (
                        local_model,
                        local_model_params,
                        arc_config,
                    ) = global_model.sample_smallest_model()
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
                learning_rate=lr,
                num_train_epochs=args.num_local_epochs,
                weight_decay=0.01,
                report_to="none",
            )
            trainer = Trainer(
                model=local_model,
                args=training_args,
                compute_metrics=functools.partial(compute_metrics, task=args.dataset),
                train_dataset=local_dataset,
                eval_dataset=val_dataset,
            )
            train_results = trainer.train()

            if round > 95:
                print(f"Eval local model {client_id}\n")
                metrics = trainer.evaluate(val_dataset)

                trainer.log_metrics("eval", metrics)
                eval_performance = list(metrics.values())[0]

                writer.add_scalar(
                    str(client_id) + "/eval_performance",
                    eval_performance,
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
            f"Communication round {round} federated learning finished. \n Average local model parameters:{avg_trainable_params}.\n Eval global model."
        )
        global_model.aggregate(local_models)

        training_args = TrainingArguments(
            output_dir=os.path.join(args.save_dir, "global"),
            per_device_train_batch_size=args.batch_size,
            per_device_eval_batch_size=args.batch_size,
            evaluation_strategy="no",
            save_strategy="no",
            learning_rate=lr,
            num_train_epochs=args.num_local_epochs,
            weight_decay=0.01,
            report_to="none",
        )
        trainer = Trainer(
            model=local_model,
            args=training_args,
            compute_metrics=functools.partial(compute_metrics, task=args.dataset),
            train_dataset=local_dataset,
            eval_dataset=val_dataset,
        )
        metrics = trainer.evaluate(val_dataset)
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

        eval_performance = list(metrics.values())[0]

        writer.add_scalar(
            "global/eval_performance",
            eval_performance,
            round,
        )

        global_model.model.to("cpu")
        if eval_performance > best_performance:
            best_performance = eval_performance
            global_model.save_ckpt(
                os.path.join(args.save_dir, args.dataset, "best_model")
            )

        writer.add_scalar(
            "global/best_performance",
            best_performance,
            round,
        )

        print(
            f"Best Validation Performance (Accuracy/Pearson Correlation/Matthews correlation coefficient): {best_performance:.4f}"
        )

        if test_dataset:
            metrics = trainer.evaluate(test_dataset, metric_key_prefix="test")
            trainer.log_metrics("test", metrics)
            trainer.save_metrics("test", metrics)

        early_stopping(eval_performance)
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
            os.path.join(ckpt_path, "elastic_space.json")
            if os.path.exists(os.path.join(ckpt_path, "elastic_space.json"))
            else args.elastic_config
        )

    else:
        ckpt_path = model_name
        elastic_config = args.elastic_config

    global_model = RaFFM(global_model.to("cpu"), elastic_config)
    global_model = federated_learning(
        args, global_model, tokenized_local_datasets, tokenize_val_dataset
    )
    global_model.save_ckpt(os.path.join(args.save_dir, args.dataset, "final"))


if __name__ == "__main__":
    args = arguments()
    main(args)
