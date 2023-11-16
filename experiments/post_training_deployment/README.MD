# Post-training High-performance Resource-aware Model Evaluation

## Experiment Goal

In this experiment we show:

**Post federated training, FMs trained by our method is scalable, and we can generate surprisingly large number of scaled FMs (> $10^{10}$) with out further training**

Such scaled FMs can fit different resource-constraints at the edge while maintaining the same level of accuracy.

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

We provide a Jupyter Notebook Tutorial **[post_training_deployment.ipynb](./post_training_deployment.ipynb)** with detailed instruction and high level APIs to reproduce our experiments.

## Results

We have some simple meta results shown on the tutorial: **[post_training_deployment.ipynb](./post_training_deployment.ipynb)**

<div style="display:flex; justify-content:space-around;">

<figure>
    <img src="./figures/RoBERTa_performance_vs_params.png" alt="Performance vs Params">
    <center><figcaption>Fig.1 Scalable RoBERTa on SST-2</figcaption></center>
</figure>
<figure>
    <img src="./figures/vit_performance_vs_params.png" alt="ViT Performance vs Params">
    <center><figcaption>Fig.2 Scalable ViT on CIFAR-10.</figcaption></center>
</figure>
</div>

In summry, Foundation Models trained by RaFFM are scalable, which can enables heterogeneous model deployment post-federated learning without further training.
