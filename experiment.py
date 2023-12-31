import time
import numpy as np
import torch
import argparse
from transformers import (
    DistilBertForSequenceClassification,
    DistilBertTokenizerFast,
    Trainer,
    TrainingArguments,
    RobertaTokenizerFast,
    T5Tokenizer,
    BertForSequenceClassification,
    BertTokenizerFast,
    DistilBertForSequenceClassification,
    RobertaForSequenceClassification,
    T5ForConditionalGeneration,
    Trainer,
    TrainingArguments,
)
from torch.utils.tensorboard import SummaryWriter

from datasets import load_dataset, concatenate_datasets
import logging
import sys
import copy
import os
from datasets import concatenate_datasets
import logging
from scipy.stats import pearsonr
from sklearn.metrics import accuracy_score, matthews_corrcoef

# from utils import *
from utils.split_data import DatasetSplitter, k_shot_data
from utils.adaptive import calculate_trainable_params, gradient_masking_extraction
from utils import (
    salient_parameter_prioritization,
    salient_submodel_extraction,
    EarlyStopping,
)

from raffm import arc_config_sampler, bert_module_handler

random_seed = 123


def step_lr(initial_lr, epoch, decay_step, decay_rate):
    return initial_lr * (decay_rate ** (epoch // decay_step))


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


def tokenize_function(examples, tokenizer, dataset):
    if dataset in ["sst2", "cola"]:
        return tokenizer(
            examples["sentence"],
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

    elif dataset == "mnli":
        return tokenizer(
            examples["premise"],
            examples["hypothesis"],
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
    elif dataset == "qqp":
        return tokenizer(
            examples["question1"],
            examples["question2"],
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
    elif dataset == "qnli":
        return tokenizer(
            examples["question"],
            examples["sentence"],
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

    elif dataset in ["mrpc", "stsb", "rte"]:
        return tokenizer(
            examples["sentence1"],
            examples["sentence2"],
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )


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
        logging.info(f"Pearson correlation: {pearson_corr}")
        return pearson_corr
    elif args.dataset == "cola":
        probabilities_class_1 = predictions.predictions[:, 1]
        binary_predictions = (probabilities_class_1 > 0.5).astype(int)
        matthews_corr = matthews_corrcoef(true_labels, binary_predictions)

        print(f"matthews correlation: {matthews_corr}")
        logging.info(f"matthews correlation: {matthews_corr}")
        return matthews_corr

    else:
        predicted_labels = predictions.predictions.argmax(-1)
        accuracy = accuracy_score(true_labels, predicted_labels)
        print(f"Accuracy: {accuracy}")
        logging.info(f"Accuracy: {accuracy}")
        return accuracy


def federated_learning(
    args,
    global_model,
    tokenized_local_datasets,
    tokenize_val_dataset,
    tokenize_test_dataset=None,
):
    # global_model = DistilBertForSequenceClassification.from_pretrained(model_name, num_labels=2)
    early_stopping = EarlyStopping(patience=10, verbose=True)

    best_model = copy.deepcopy(global_model.to("cpu"))
    best_acc = 0
    for communication_round in range(args.num_rounds):
        local_models = []
        lr = step_lr(args.learning_rate, communication_round, 5, 0.98)
        global_model.to("cpu")
        # randomly select 10% client index for training
        np.random.seed(int(time.time()))  # Set the seed to the current time
        client_indices = np.random.choice(
            len(tokenized_local_datasets),
            size=int(0.1 * len(tokenized_local_datasets)),
            replace=False,
        )
        avg_trainable_params = 0

        if args.spp:
            global_model = salient_parameter_prioritization(global_model)

        for idx, client_id in enumerate(client_indices):
            # for client_id, client_dataset in enumerate(train_datasets):
            tokenized_client_dataset = tokenized_local_datasets[client_id]
            print(
                f"Training client {client_id} in communication round {communication_round}"
            )

            if args.algo == "raffm":
                if idx == 0:
                    local_model = copy.deepcopy(global_model)
                    (
                        total_trainable_params,
                        total_params,
                        percentage,
                    ) = calculate_trainable_params(local_model)
                else:
                    # (
                    #     local_model,
                    #     total_trainable_params,
                    #     total_params,
                    #     percentage,
                    # ) = salient_submodel_extraction(
                    #     global_model,
                    #     target_model_params_size=None,
                    #     zero_fill=args.zero_fill,
                    # )  # Target model params size is None for randomly sample subnetwork
                    atten_out_space = [768 - i * 24 for i in range(0, 10)]
                    inter_hidden_space = [3072 - i * 128 for i in range(0, 25)]
                    out_hidden_space = [768 - i * 24 for i in range(0, 10)]

                    arc_config = arc_config_sampler(
                        atten_out_space, inter_hidden_space, out_hidden_space
                    )

                    (local_model, total_trainable_params) = bert_module_handler(
                        global_model, arc_config
                    )  # Target model params size is None for randomly sample subnetwork
                    percentage = total_trainable_params / total_params
            elif args.algo == "vanilla":
                local_model = copy.deepcopy(global_model)
                (
                    total_trainable_params,
                    total_params,
                    percentage,
                ) = calculate_trainable_params(local_model)
            avg_trainable_params += total_trainable_params

            writer.add_scalar(
                str(client_id) + "/trainable_params",
                total_trainable_params,
                communication_round,
            )
            writer.add_scalar(
                str(client_id) + "/total_params", total_params, communication_round
            )
            print(
                f"Client {client_id} has {total_trainable_params} trainable parameters out of {total_params} parameters, which is {percentage}% in communication round {communication_round}"
            )
            logging.info(
                f"Client {client_id} has {total_trainable_params} trainable parameters out of {total_params} parameters, which is {percentage}% in communication round {communication_round}"
            )

            writer.add_scalar(
                str(client_id) + "/trainable_params_percentage",
                percentage,
                communication_round,
            )

            logdir = os.path.join(args.log_dir, f"client_{client_id}")

            training_args = TrainingArguments(
                logdir,
                # logging_dir = logdir,
                # logging_steps = 1000,
                # logging_strategy="epoch",
                evaluation_strategy="no",
                save_strategy="no",
                learning_rate=lr,
                per_device_train_batch_size=args.per_device_train_batch_size,
                per_device_eval_batch_size=args.per_device_eval_batch_size,
                num_train_epochs=args.num_local_epochs,
                weight_decay=0.01,
            )

            trainer = Trainer(
                model=local_model,
                args=training_args,
                train_dataset=tokenized_client_dataset,
            )

            trainer.train()

            print("local model training finished")
            logging.info(f"local model training finished")

            if args.eval_lm:
                res = evaluate(args, local_model, tokenize_val_dataset)
                print(f"client {client_id} local model val accuracy is {res}")
                logging.info(f"client {client_id} local model val accuracy is {res}")
            local_model.to("cpu")  # Move the local model to CPU memory
            local_models.append(local_model)

        with torch.no_grad():
            for name, param in global_model.named_parameters():
                param *= 0
                for local_model in local_models:
                    local_param = local_model.state_dict()[name].cpu()
                    if len(local_param.shape) == 2:
                        param[
                            : local_param.shape[0], : local_param.shape[1]
                        ] += local_param / len(local_models)
                    else:
                        param[: local_param.shape[0]] += local_param / len(local_models)

                # param /= len(local_models)

        print(
            f"Evaluating global model after communication round {communication_round}"
        )
        logging.info(
            f"Evaluating global model after communication round {communication_round}"
        )

        writer.add_scalar(
            "trainable_params/avg",
            avg_trainable_params / len(client_indices),
            communication_round,
        )
        writer.add_scalar("trainable_params/org", total_params, communication_round)
        print(
            f"Average trainable parameters is {avg_trainable_params/len(client_indices)} out of {total_params} parameters"
        )
        logging.info(
            f"Average trainable parameters is {avg_trainable_params/len(client_indices)} out of {total_params} parameters"
        )

        res = evaluate(args, global_model, tokenize_val_dataset)
        writer.add_scalar("val_accuracy", res, communication_round)
        print(f"Val accuracy is {res}")
        logging.info(f"Val accuracy is {res}")

        if tokenize_test_dataset is not None:
            try:
                test_acc = evaluate(args, global_model, tokenize_test_dataset)
                writer.add_scalar("test_accuracy", test_acc, communication_round)
                print(f"Test accuracy is {test_acc}")
                logging.info(f"Test accuracy is {test_acc}")
            except:
                print("Test accuracy is not calculated")
                logging.info("Test accuracy is not calculated")

        if res > best_acc:
            best_acc = res
            best_model = copy.deepcopy(global_model.to("cpu"))
        early_stopping(res)
        if early_stopping.has_converged():
            print("Model has converged. Stopping training.")
            break
    return global_model, best_model


def main(args):
    if args.model == "distilbert":
        model_name = "distilbert-base-uncased"
        tokenizer = DistilBertTokenizerFast.from_pretrained(model_name)
    elif args.model == "roberta":
        model_name = "roberta-base"
        tokenizer = RobertaTokenizerFast.from_pretrained(model_name)
    elif args.model == "t5":
        model_name = "t5-small"  # You can also use "t5-base" or other T5 variants
        tokenizer = T5Tokenizer.from_pretrained(model_name)

    elif args.model == "bert-base":
        model_name = "bert-base-uncased"
        tokenizer = BertTokenizerFast.from_pretrained(model_name)

    elif args.model == "bert-large":
        model_name = "bert-large-uncased-whole-word-masking"
        tokenizer = BertTokenizerFast.from_pretrained(model_name)

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
        global_model = DistilBertForSequenceClassification.from_pretrained(
            model_name, num_labels=num_classes[args.dataset]
        )
    elif args.model == "roberta":
        global_model = RobertaForSequenceClassification.from_pretrained(
            model_name, num_labels=num_classes[args.dataset]
        )
    elif args.model == "t5":
        global_model = T5ForConditionalGeneration.from_pretrained(
            model_name
        ).cpu()  # T5 doesn't use num_labels
    elif args.model == "bert-base":
        global_model = BertForSequenceClassification.from_pretrained(
            model_name, num_labels=num_classes[args.dataset]
        )

    elif args.model == "bert-large":
        global_model = BertForSequenceClassification.from_pretrained(
            model_name, num_labels=num_classes[args.dataset]
        )

    if args.dataset in ["sst2", "mrpc", "qnli", "rte", "cola"]:
        dataset = load_dataset("glue", args.dataset)
    elif args.dataset == "mnli":
        dataset = load_dataset("glue", "mnli")
        # MNLI has matched and mismatched validation sets
        # Here we concatenate them for simplicity
        dataset["validation"] = concatenate_datasets(
            [dataset["validation_matched"], dataset["validation_mismatched"]]
        )
    elif args.dataset == "qqp":
        dataset = load_dataset("glue", "qqp")
    elif args.dataset == "stsb":
        dataset = load_dataset("glue", "stsb")

    train_dataset = dataset["train"]
    val_dataset = dataset["validation"]

    if args.dataset == "mnli":
        # test_dataset = dataset["validation_matched"]
        test_dataset = dataset["validation"]
    else:
        test_dataset = dataset["test"]
    # num_labels = test_dataset.features["label"].num_classes

    tokenize_test_dataset = test_dataset.map(
        lambda examples: tokenize_function(examples, tokenizer, args.dataset),
        batched=True,
    )
    tokenize_val_dataset = val_dataset.map(
        lambda examples: tokenize_function(examples, tokenizer, args.dataset),
        batched=True,
    )

    tokenized_train_dataset = train_dataset.map(
        lambda examples: tokenize_function(examples, tokenizer, args.dataset),
        batched=True,
    )

    dash_line = "-" * 80

    if args.k_shot and not args.split_data:
        print(dash_line + "\nFederated learning for few-shot learning")
        logging.info(dash_line + "\nFederated learning for few-shot learning")
        # local_datasets = k_shot_data(train_dataset, args.num_clients, args.k_shot,args.dataset)
        tokenized_local_datasets = k_shot_data(
            tokenized_train_dataset, args.num_clients, args.k_shot, args.dataset
        )
    else:
        print(dash_line + "\nFederated learning")
        logging.info(dash_line + "\nFederated learning")
        # splitter = DatasetSplitter(train_dataset, seed=random_seed)
        splitter = DatasetSplitter(tokenized_train_dataset, seed=random_seed)

        tokenized_local_datasets = splitter.split(n=args.num_clients, replacement=False)

    # tokenized_local_datasets = []
    # for client_dataset in local_datasets:
    #     tokenized_local_datasets.append(client_dataset.map(lambda examples: tokenize_function(examples, tokenizer, args.dataset), batched=True))

    global_model, best_model = federated_learning(
        args,
        global_model,
        tokenized_local_datasets,
        tokenize_val_dataset,
        tokenize_test_dataset=None,
    )

    print(dash_line + "\nFinal evaluation")
    logging.info(dash_line + "\nFinal evaluation")
    evaluate(args, global_model, tokenize_val_dataset)

    print(dash_line + "\nBest model evaluation")
    logging.info(dash_line + "\nBest model evaluation")
    evaluate(args, best_model, tokenize_val_dataset)
    if args.save_model:
        best_model.save_pretrained(os.path.join(args.log_dir, "best_model"))


"""
baseline running command:
python fl_glue.py --save_model --algo vanilla --split_data --num_clients 100 --num_rounds 50 --num_local_epochs 3 --dataset mnli --per_device_train_batch_size 24 --per_device_eval_batch_size 24 --model bert-base --log_dir log_glue/baseline/mnli > baseline_bert_base_100_mnli.txt 

python fl_glue.py --save_model --algo vanilla --split_data --num_clients 100 --num_rounds 50 --num_local_epochs 3 --dataset sst2 --per_device_train_batch_size 24 --per_device_eval_batch_size 24 --model roberta --log_dir log_glue_roberta/baseline/sst2 > baseline_roberta_100_sst2.txt
python fl_glue.py --save_model --algo vanilla --split_data --num_clients 100 --num_rounds 50 --num_local_epochs 3 --dataset sst2 --per_device_train_batch_size 48 --per_device_eval_batch_size 48 --model distilbert --log_dir log_glue_distilbert/baseline/sst2 > baseline_distilbert_100_sst2.txt

python fl_glue.py --save_model --algo vanilla --split_data --num_clients 100 --num_rounds 50 --num_local_epochs 3 --dataset rte --per_device_train_batch_size 24 --per_device_eval_batch_size 24 --model roberta --log_dir log_glue_roberta/baseline/rte > baseline_roberta_100_rte.txt
python fl_glue.py --save_model --algo vanilla --split_data --num_clients 100 --num_rounds 50 --num_local_epochs 3 --dataset rte --per_device_train_batch_size 48 --per_device_eval_batch_size 48 --model distilbert --log_dir log_glue_distilbert/baseline/rte > baseline_distilbert_100_rte.txt

python fl_glue.py --save_model --algo vanilla --split_data --num_clients 100 --num_rounds 50 --num_local_epochs 3 --dataset mrpc --per_device_train_batch_size 24 --per_device_eval_batch_size 24 --model roberta --log_dir log_glue_roberta/baseline/mrpc > baseline_roberta_100_mrpc.txt
python fl_glue.py --save_model --algo vanilla --split_data --num_clients 100 --num_rounds 50 --num_local_epochs 3 --dataset mrpc --per_device_train_batch_size 48 --per_device_eval_batch_size 48 --model distilbert --log_dir log_glue_distilbert/baseline/mrpc > baseline_distilbert_100_mrpc.txt

python fl_glue.py --save_model --algo vanilla --split_data --num_clients 100 --num_rounds 50 --num_local_epochs 3 --dataset cola --per_device_train_batch_size 24 --per_device_eval_batch_size 24 --model roberta --log_dir log_glue_roberta/baseline/cola > baseline_roberta_100_cola.txt
python fl_glue.py --save_model --algo vanilla --split_data --num_clients 100 --num_rounds 50 --num_local_epochs 3 --dataset cola --per_device_train_batch_size 48 --per_device_eval_batch_size 48 --model distilbert --log_dir log_glue_distilbert/baseline/cola > baseline_distilbert_100_cola.txt

python fl_glue.py --save_model --algo vanilla --split_data --num_clients 100 --num_rounds 50 --num_local_epochs 3 --dataset stsb --per_device_train_batch_size 24 --per_device_eval_batch_size 24 --model roberta --log_dir log_glue_roberta/baseline/stsb > baseline_roberta_100_stsb.txt
python fl_glue.py --save_model --algo vanilla --split_data --num_clients 100 --num_rounds 50 --num_local_epochs 3 --dataset stsb --per_device_train_batch_size 48 --per_device_eval_batch_size 48 --model distilbert --log_dir log_glue_distilbert/baseline/stsb > baseline_distilbert_100_stsb.txt

python fl_glue.py --save_model --algo vanilla --split_data --num_clients 100 --num_rounds 50 --num_local_epochs 3 --dataset qqp --per_device_train_batch_size 24 --per_device_eval_batch_size 24 --model roberta --log_dir log_glue_roberta/baseline/qqp > baseline_roberta_100_qqp.txt
python fl_glue.py --save_model --algo vanilla --split_data --num_clients 100 --num_rounds 50 --num_local_epochs 3 --dataset qqp --per_device_train_batch_size 48 --per_device_eval_batch_size 48 --model distilbert --log_dir log_glue_distilbert/baseline/qqp > baseline_distilbert_100_qqp.txt

python fl_glue.py --save_model --algo vanilla --split_data --num_clients 100 --num_rounds 50 --num_local_epochs 3 --dataset qnli --per_device_train_batch_size 24 --per_device_eval_batch_size 24 --model roberta --log_dir log_glue_roberta/baseline/qnli > baseline_roberta_100_qnli.txt
python fl_glue.py --save_model --algo vanilla --split_data --num_clients 100 --num_rounds 50 --num_local_epochs 3 --dataset qnli --per_device_train_batch_size 48 --per_device_eval_batch_size 48 --model distilbert --log_dir log_glue_distilbert/baseline/qnli > baseline_distilbert_100_qnli.txt

python fl_glue.py --save_model --algo vanilla --split_data --num_clients 100 --num_rounds 50 --num_local_epochs 3 --dataset mnli --per_device_train_batch_size 24 --per_device_eval_batch_size 24 --model roberta --log_dir log_glue_roberta/baseline/mnli > baseline_roberta_100_mnli.txt
python fl_glue.py --save_model --algo vanilla --split_data --num_clients 100 --num_rounds 50 --num_local_epochs 3 --dataset mnli --per_device_train_batch_size 48 --per_device_eval_batch_size 48 --model distilbert --log_dir log_glue_distilbert/baseline/mnli > baseline_distilbert_100_mnli.txt


python fl_glue.py --save_model --algo vanilla --split_data --num_clients 100 --num_rounds 50 --num_local_epochs 3 --dataset mrpc --per_device_train_batch_size 8 --per_device_eval_batch_size 8 --model bert-large --log_dir log_glue_bert_large/baseline/mrpc > baseline_bert_large_100_mrpc.txt
python fl_glue.py --save_model --algo vanilla --split_data --num_clients 100 --num_rounds 50 --num_local_epochs 3 --dataset mnli --per_device_train_batch_size 8 --per_device_eval_batch_size 8 --model bert-large --log_dir log_glue_bert_large/baseline/mnli > baseline_bert_large_100_mnli.txt
python fl_glue.py --save_model --algo vanilla --split_data --num_clients 100 --num_rounds 20 --num_local_epochs 3 --dataset qqp --per_device_train_batch_size 8 --per_device_eval_batch_size 8 --model bert-large --log_dir log_glue_bert_large/baseline/qqp > baseline_bert_large_100_qqp.txt
python fl_glue.py --save_model --algo vanilla --split_data --num_clients 100 --num_rounds 100 --num_local_epochs 3 --dataset qnli --per_device_train_batch_size 8 --per_device_eval_batch_size 8 --model bert-large --log_dir log_glue_bert_large/baseline/qnli > baseline_bert_large_100_qnli.txt
*python fl_glue.py --save_model --algo vanilla --split_data --num_clients 50 --num_rounds 100 --num_local_epochs 3 --dataset rte --per_device_train_batch_size 8 --per_device_eval_batch_size 8 --model bert-large --log_dir log_glue_bert_large/baseline/rte > baseline_bert_large_100_rte.txt
*python fl_glue.py --save_model --algo vanilla --split_data --num_clients 50 --num_rounds 100 --num_local_epochs 3 --dataset cola --per_device_train_batch_size 8 --per_device_eval_batch_size 8 --model bert-large --log_dir log_glue_bert_large/baseline/cola > baseline_bert_large_100_cola.txt
*python fl_glue.py --save_model --algo vanilla --split_data --num_clients 50 --num_rounds 100 --num_local_epochs 3 --dataset stsb --per_device_train_batch_size 8 --per_device_eval_batch_size 8 --model bert-large --log_dir log_glue_bert_large/baseline/stsb > baseline_bert_large_100_stsb.txt


+python fl_glue.py --save_model --algo vanilla --split_data --num_clients 100 --num_rounds 50 --num_local_epochs 3 --dataset rte --per_device_train_batch_size 8 --per_device_eval_batch_size 8 --model bert-large --log_dir log_glue_bert_large/baseline/rte > baseline_bert_large_100_rte.txt

-python fl_glue.py --save_model --algo vanilla --split_data --num_clients 100 --num_rounds 50 --num_local_epochs 3 --dataset stsb --per_device_train_batch_size 8 --per_device_eval_batch_size 8 --model bert-large --log_dir log_glue_bert_large/baseline/stsb > baseline_bert_large_100_stsb.txt
"""

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # parser.add_argument("--method", choices=["centralized", "federated", "federated_foundation"], required=True)
    parser.add_argument(
        "--algo", type=str, default="raffm", choices=["vanilla", "raffm"]
    )
    parser.add_argument(
        "--spp", action="store_true", help="salient parameter prioritization"
    )
    parser.add_argument("--eval_lm", action="store_true", help="evaluate local models")
    parser.add_argument("--split_data", action="store_true")
    parser.add_argument("--num_clients", type=int, default=100)
    parser.add_argument("--k_shot", type=int, default=4)
    parser.add_argument("--num_rounds", type=int, default=100)
    # parser.add_argument("--num_epochs", type=int, default=3)
    parser.add_argument("--num_local_epochs", type=int, default=3)
    parser.add_argument(
        "--dataset",
        type=str,
        default="sst2",
        choices=["sst2", "mrpc", "mnli", "qqp", "qnli", "stsb", "rte", "cola"],
        help="Choose between 'sst2', 'mrpc', 'mnli', 'qqp', 'qnli', 'stsb', 'rte', 'cola' datasets",
    )
    parser.add_argument("--per_device_train_batch_size", type=int, default=40)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=40)
    parser.add_argument("--log_dir", type=str, default="centralized/4")
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument(
        "--model",
        type=str,
        default="distilbert",
        choices=["distilbert", "bert-base", "roberta", "t5", "bert-large"],
        help="Choose between 'distilbert', 'roberta', and 't5' models",
    )
    parser.add_argument("--save_model", action="store_true")
    parser.add_argument("--zero_fill", action="store_true")

    args = parser.parse_args()

    os.makedirs(args.log_dir, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(os.path.join(args.log_dir, "log.txt")),
            logging.StreamHandler(sys.stdout),
        ],
    )
    writer = SummaryWriter(args.log_dir)
    main(args)

"""

SPP experiment
python fl_glue_spp.py --spp --algo vanilla --split_data --num_clients 100 --num_rounds 50 --num_local_epochs 3 --dataset cola --per_device_train_batch_size 24 --per_device_eval_batch_size 24 --model bert-base --log_dir log_glue/spp/cola > cola_spp_baseline_bertbase_100.txt

python fl_glue_spp.py --spp --algo raffm --split_data --num_clients 100 --num_rounds 50 --num_local_epochs 3 --dataset qqp --per_device_train_batch_size 24 --per_device_eval_batch_size 24 --model distilbert --log_dir log_glue/spp/qqp > raffm_distilbert_100_qqp.txt
python fl_glue_spp.py --spp --algo raffm --split_data --num_clients 100 --num_rounds 50 --num_local_epochs 3 --dataset mnli --per_device_train_batch_size 24 --per_device_eval_batch_size 24 --model distilbert --log_dir log_glue/spp/mnli > raffm_distilbert_100_mnli.txt
python fl_glue_spp.py --eval_lm --spp --algo raffm --split_data --num_clients 100 --num_rounds 50 --num_local_epochs 3 --dataset sst2 --per_device_train_batch_size 24 --per_device_eval_batch_size 24 --model distilbert --log_dir log_glue/spp/sst2 > raffm_distilbert_100_sst2.txt
python fl_glue_spp.py --eval_lm --spp --algo raffm --split_data --num_clients 100 --num_rounds 50 --num_local_epochs 3 --dataset stsb --per_device_train_batch_size 24 --per_device_eval_batch_size 24 --model distilbert --log_dir log_glue/spp/stsb > raffm_distilbert_100_stsb.txt
python fl_glue_spp.py --eval_lm --spp --algo raffm --split_data --num_clients 100 --num_rounds 50 --num_local_epochs 3 --dataset qnli --per_device_train_batch_size 24 --per_device_eval_batch_size 24 --model distilbert --log_dir log_glue/spp/qnli > raffm_distilbert_100_qnli.txt
python fl_glue_spp.py --eval_lm --spp --algo raffm --split_data --num_clients 100 --num_rounds 50 --num_local_epochs 3 --dataset mrpc --per_device_train_batch_size 24 --per_device_eval_batch_size 24 --model distilbert --log_dir log_glue/spp/mrpc > raffm_distilbert_100_mrpc.txt
python fl_glue_spp.py --eval_lm --spp --algo raffm --split_data --num_clients 100 --num_rounds 50 --num_local_epochs 3 --dataset cola --per_device_train_batch_size 24 --per_device_eval_batch_size 24 --model distilbert --log_dir log_glue/spp/cola > raffm_distilbert_100_cola.txt

python fl_glue_spp.py --eval_lm --spp --algo raffm --split_data --num_clients 100 --num_rounds 50 --num_local_epochs 3 --dataset mrpc --per_device_train_batch_size 8 --per_device_eval_batch_size 8 --model roberta --log_dir log_glue/roberta/mrpc > raffm_roberta_100_mrpc.txt
python fl_glue_spp.py --eval_lm --spp --algo raffm --split_data --num_clients 100 --num_rounds 50 --num_local_epochs 3 --dataset qqp --per_device_train_batch_size 8 --per_device_eval_batch_size 8 --model roberta --log_dir log_glue/roberta/qqp > raffm_roberta_100_qqp.txt
python fl_glue_spp.py --eval_lm --spp --algo raffm --split_data --num_clients 100 --num_rounds 50 --num_local_epochs 3 --dataset mnli --per_device_train_batch_size 8 --per_device_eval_batch_size 8 --model roberta --log_dir log_glue/roberta/mnli > raffm_roberta_100_mnli.txt

python fl_glue_spp.py --eval_lm --spp --algo raffm --split_data --num_clients 100 --num_rounds 50 --num_local_epochs 3 --dataset qnli --per_device_train_batch_size 12 --per_device_eval_batch_size 12 --model roberta --log_dir log_glue/roberta/qnli > raffm_roberta_100_qnli.txt
python fl_glue_spp.py --eval_lm --spp --algo raffm --split_data --num_clients 100 --num_rounds 50 --num_local_epochs 3 --dataset stsb --per_device_train_batch_size 12 --per_device_eval_batch_size 12 --model roberta --log_dir log_glue/roberta/stsb > raffm_roberta_100_stsb.txt
python fl_glue_spp.py --eval_lm --spp --algo raffm --split_data --num_clients 100 --num_rounds 50 --num_local_epochs 3 --dataset sst2 --per_device_train_batch_size 12 --per_device_eval_batch_size 12 --model roberta --log_dir log_glue/roberta/sst2 > raffm_roberta_100_sst2.txt
python fl_glue_spp.py --eval_lm --spp --algo raffm --split_data --num_clients 100 --num_rounds 50 --num_local_epochs 3 --dataset rte --per_device_train_batch_size 12 --per_device_eval_batch_size 12 --model roberta --log_dir log_glue/roberta/rte > raffm_roberta_100_rte.txt
python fl_glue_spp.py --eval_lm --spp --algo raffm --split_data --num_clients 100 --num_rounds 50 --num_local_epochs 3 --dataset cola --per_device_train_batch_size 12 --per_device_eval_batch_size 12 --model roberta --log_dir log_glue/roberta/cola > raffm_roberta_100_cola.txt


python experiment.py --save_model --eval_lm --spp --algo raffm --split_data --num_clients 100 --num_rounds 10 --num_local_epochs 3 --dataset rte --per_device_train_batch_size 24 --per_device_eval_batch_size 24 --model bert-base --log_dir log_glue/spp/bert > raffm_bert_100_rte.txt


Ablation study:

CUDA_VISIBLE_DEVICES=0,1 python fl_glue_spp.py --eval_lm --spp --algo raffm --split_data --num_clients 100 --num_rounds 100 --num_local_epochs 3 --dataset sst2 --per_device_train_batch_size 12 --per_device_eval_batch_size 12 --model roberta --log_dir log_sst2 > spp_roberta_100_sst2.txt
CUDA_VISIBLE_DEVICES=2,3 python fl_glue_spp.py --eval_lm --algo raffm --split_data --num_clients 100 --num_rounds 100 --num_local_epochs 3 --dataset sst2 --per_device_train_batch_size 12 --per_device_eval_batch_size 12 --model roberta --log_dir log_sst2 > nospp_roberta_100_sst2.txt

CUDA_VISIBLE_DEVICES=0,1 python fl_glue_spp.py --zero_fill --eval_lm --spp --algo raffm --split_data --num_clients 100 --num_rounds 100 --num_local_epochs 3 --dataset sst2 --per_device_train_batch_size 12 --per_device_eval_batch_size 12 --model roberta --log_dir log_sst2 > zero_spp_roberta_100_sst2.txt
CUDA_VISIBLE_DEVICES=2,3 python fl_glue_spp.py --zero_fill --eval_lm --algo raffm --split_data --num_clients 100 --num_rounds 100 --num_local_epochs 3 --dataset sst2 --per_device_train_batch_size 12 --per_device_eval_batch_size 12 --model roberta --log_dir log_sst2 > zero_nospp_roberta_100_sst2.txt

"""
