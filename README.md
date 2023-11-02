# Resource-aware Federated Foundation Models (RaFFM): Bridging the Gap Between Foundation Models and Heterogeneous Federated Learning ([paper](https://arxiv.org/pdf/2310.00247.pdf))

This is the official implementation for the paper: 

*Bridging the Gap Between Foundation Models and Heterogeneous Federated Learning* ([**paper**](https://arxiv.org/pdf/2310.00247.pdf))

## Updates

- [x] [10/30/2023] Scalable ViT Checkpoints released for heterogeneous resource edge-clients
- [x] [10/30/2023] Demo scripts for train ViT on CIFAR-10/100 via heterogeneous resource FL
- [x] [10/31/2023] Pushed elastic space APIs for system-heteo
- [x] [11/02/2023] ViT-base CIFAR-100 checkpoints released, trained on *large-budget* edge-FL settings with 100 clients.

## Installation

```bash
conda create -n raffm python=3.10
conda activate raffm
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
pip install -r requirements.txt
```

##  Scalable FMs Checkpoints

RaFFM enables **resource-aware foundation model deployments** in edge-FL based on client local resources. That means RaFFM can dynamically scale down the size of FMs to heterogeneous resource local clients and enables efficient and fair local resource utilization.

### Download Links
We provide resource-aware FMs checkpoints trained via FL. You can download here:

- [ViT-base CIFAR-100 *large-budget*](https://iowastate-my.sharepoint.com/:f:/g/personal/yusx_iastate_edu/El-nYhaOiINOoLHXsqqXqAkBOUEc_ohu-gGK0ms8neEFDQ) [Trained on *Large-budget*  system heterogeneity edge-FL with 100 clients]
- [ViT-base CIFAR-10 *large-budget*](https://iowastate-my.sharepoint.com/:f:/g/personal/yusx_iastate_edu/EkAE4V6VyO1JnpN9j-0QznUBjtREcIVttgVL9sjeP7mfvA) [Trained on *Large-budget*  system heterogeneity edge-FL with 100 clients]
- [ViT-base CIFAR-10 *small-budget*](https://iowastate-my.sharepoint.com/:u:/g/personal/yusx_iastate_edu/EYVquHHGqFBLlwZpSrfVTUYBJDcOCvD3Cw2AWk3BdseWjQ) [Trained on *Small-budget*  system heterogeneity edge-FL with 100 clients]


### Checkpoints Usage
```bash
cd RaFFM
```
<!-- # RaFFM
# ├── 
# |   ├── .gitignore
# |   ├── fl_vit.py
# |   ├── requirements.txt
# |   ├── ...
# ├── raffm
# |   ├── ...

``` -->


```python
from transformers import ViTImageProcessor, ViTForImageClassification, TrainingArguments, Trainer
from raffm import RaFFM

# Generate resource-aware models for cifar10
ckpt_path = "ckpt_folder_path" # the downloaded and unzipped ckpt folder path 

model = ViTForImageClassification.from_pretrained(
    ckpt_path,
    num_labels=10,
    ignore_mismatched_sizes=True,
)

raffm_model = RaFFM(model.to("cpu"))
print("Original FM number of parameters:",raffm_model.total_params)

#Random sample a scaled FM
submodel, params, config = raffm_model.random_resource_aware_model()
print("subnetwork params",params)
```


**Detailed instructions for generating resource-aware FMs can be found in [Demo](./fm_scaling.ipynb)**



## Train FMs on Heterogeneous Resource edge-FL
RaFFM is able to scale down a given FMs based on edge resource constraints, and hence, enabling resource-aware federated learning.

Here we provide scripts to reproduce the experimental results we reported in paper.

### Resource-aware ViT
Train ViT on 100 clients edge-FL settings with 10% participate rate each communication round, simply run scripts:

```bash
python fl_vit.py --method raffm --spp --model vit --save_dir log/vit --dataset cifar10 --num_clients 100 --lr 3e-5
```

To check the results, you can:

- Check the output information from terminal console
- Use tensorboard: `tensorboard --logdir log/vit`

**[Note]**: More APIs and scripts will post, please check the [**Updates**](#updates).

## Training on Edge
The above scripts is simulate on central device for reproducibility of RaFFM, if you want to deploy RaFFM on edge-FL:

please see [EDGE-FL.md](TRAINING.md) for detailed training instructions.

## Contact

Sixing Yu: <yusx@iastate.edu>

## TODO

- [x] ViT pre-trained ckpts
- [x] ViT FL simulation scripts
- [x] Tensorboard logger
- [x] Elastic space APIs for system-heteo
- [x] Load ckpt high-level APIs
- [ ] Evaluate Scripts for resource-aware models
- [ ] BERT-large, FLAN-T5 ckpts
- [ ] Simulation scripts on SQUAD, GLUE
- [ ] ViT CIFAR-100 ckpts
- [ ] High level API for real edge-FL
- [ ] ONNX and TensorRT APIs for edge
- [ ] Tiny fedlib 

## Citation

If you find our work is helpful, please kindly support our efforts by citing our paper:

```
@misc{yu2023raffm,
      title={Bridging the Gap Between Foundation Models and Heterogeneous Federated Learning}, 
      author={Sixing Yu and J. Pablo Muñoz and Ali Jannesari},
      year={2023},
      eprint={2310.00247},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```

## Acknowledgement
The experiments of this work is supported by **SHF CloudBank** and **Intel Labs ISL-cluster**.