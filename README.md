# Resource-aware Federated Foundation Models (RaFFM): Bridging the Gap Between Foundation Models and Heterogeneous Federated Learning ([paper](https://arxiv.org/pdf/2310.00247.pdf))

This is the official implementation for the paper: 

Bridging the Gap Between Foundation Models and Heterogeneous Federated Learning ([paper](https://arxiv.org/pdf/2310.00247.pdf))

## Updates

- [x] [10/30/2023] Scallable ViT Checkpoints released for heterogeneous resource edge-clients
- [x] [10/30/2023] Demo scripts for train ViT on CIFAR-10/100 via heterogeneous resource FL


## Installation

```bash
conda create -n raffm python=3.10
conda activate raffm
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
pip install -r requirements.txt
```

##  Scalable FMs Checkpoints

RaFFM enables **resource-aware foundation model deployments** in edge-FL based on client local resources. That means, RaFFM can dynamically scaling down the size of FMs to heterogeneous resource local clients and enables efficient and fair local resource utilization.

### Download Links
We provide resource-aware Foundation model checkpoints trained via FL, you can here:

- [ViT-base Large checkpoints]() [Trained on *Large*  system heterogeneity setting]
- [ViT-base Small checkpoints]() [Trained on *Small*  system heterogeneity setting]

### Usage
```bash
cd RaFFM

RaFFM
├── 
|   ├── .gitignore
|   ├── fl_vit.py
|   ├── requirements.txt
|   ├── ...
├── raffm
|   ├── ...

```


```python
from transformers import ViTImageProcessor, ViTForImageClassification, TrainingArguments, Trainer
from raffm import RaFFM

# Generate resource aware models for cifar10
ckpt_path = "ckpt_folder_path" # the downloaded and unzipped ckpt folder path 

model = ViTForImageClassification.from_pretrained(
    ckpt_path,
    num_labels=10,
    ignore_mismatched_sizes=True,
)

raffm_model = RaFFM(model.to("cpu"))
print("Original FM number of parameters:",raffm_model.total_params)

#Random sample a scaled FM
submodel, config, params = raffm_model.random_resource_aware_model()
print("subnetwork params",params)
```


**Detailed instructions for generate reource-aware FMs can be find in [Demo](./fm_scaling.ipynb)**



