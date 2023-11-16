# Salient Parameter Prioritization (SPP)

## Experiment Goal

In this experiment we show:

- **The novelty and nessisity of RaFFM's specialized SPP**
- **RaFFM SPP preserves the pre-trained knowledge in FMs**
- **Comparison with Standard Pruning-Based Weights Ranking**

## Reproduce the Experiments

### Installation

Refer the detailed [installation guide](../../README.md).

```bash
conda create -n raffm python=3.10
conda activate raffm
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
pip install -r requirements.txt
```

### Run the Experiments

We provide a Jupyter Notebook Tutorial **[salient_parameter_prioritization](./salient_parameter_prioritization.ipynb)** with detailed instruction and high level APIs to reproduce our experiments.

## Experiment setting

We use Vision Transformer (ViT) as base fundation mdoel in this experiments, and we leverage standard rank-based pruning algorithm compare with RaFFM's specialized salient parameter prioritization.

We first perform the parameter prioritization use different method, and then compare the ranked FMs's performance on dowmstream tasks and comparison with original FMs.

Detailed instructions and hyperparameters please refer our tutorial **[salient_parameter_prioritization](./salient_parameter_prioritization.ipynb)**, where we have detailed instruction and high level APIs to reproduce our experiments.

## Results

| Model    | Method       | Cifar10  |        | Cifar100 |        | Flower102 |        |
| -------- | ------------ | -------- | ------ | -------- | ------ | --------- | ------ |
|          |              | Accuracy | F1     | Accuracy | F1     | Accuracy  | F1     |
| ViT-base | Original FM  | 96.70%   | 96.26% | 84.50%   | 84.30% | 98.20%    | 97.85% |
|          | Pruning-Rank | 48.82%   | 46.53% | 15.40%   | 14.88% | 55.60%    | 54.70% |
|          | RaFFM SPP    | 96.78%   | 96.53% | 85.20%   | 85.00% | 98.50%    | 98.10% |
