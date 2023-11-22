# Comparison RaFFM with baselines

## Baselines and Acknowledgement

We compare RaFFM with baseline:

**PriSM (From official Implementation)**: Overcoming Resource Constraints in Federated Learning: Large Models Can Be Trained with only Weak Clients

[PriSM Official Implementation](https://github.com/yuehniu/modeldecomp-fl/tree/master)

**PruneFL (Our Implementation)**: Model pruning enables efficient federated learning on edge devices

[PruneFL Official Implementation](https://github.com/jiangyuang/PruneFL)

### Acknowledgment

We thanks for the excellence of the baseline work and their great efforts. Despite these differences in objectives and model focus, **we recognize and value the foundational work in resource-efficient FL represented by PruneFL and PriSM**. These methods have provided essential insights and laid the groundwork for advancements like RaFFM, which aim to further enhance efficiency and scalability in FL.

## Reproduce experiments

We provide simulation scripts for you to quickly verify our experiments.

To run the baselines **PriSM** you can simply use the low-level APIs bellow:

<!-- ```bash
CUDA_VISIBLE_DEVICES=0 python prism_fl.py
```

Similarly, to run the baselines **PruneLF** you can simply run the following commands:

```bash
CUDA_VISIBLE_DEVICES=0 python prune_fl.py
``` -->

<!-- The full hyper-parameters lists and arguments can be find at **[`arguments.py`](arguments.py)**. -->

### Low-level APIs for Baselines

We provide low-level API for you to reproduce our experiment results and run the baselines in your own senarios (such as end-users).

For instance, to run the baseline PruneFL in your environments, you can use the API:

```python
from PruneFL import prunefl
from transformers import ViTForImageClassification

model = ViTForImageClassification.from_pretrained(
    'google/vit-base-patch16-224-in21k',
    num_labels=len(labels),
    id2label={str(i): c for i, c in enumerate(labels)},
    label2id={c: str(i) for i, c in enumerate(labels)},
    ignore_mismatched_sizes=True,
)

#Prune the model
pruned_model, model_sparsity = prunefl(global_model,rank=0.8, threshold=1e-2)

#Train your model here ....

```

Similarly, you can use the low-level APIs to run PriSM, and converts a ViT to PriSM model:

```python
from PriSM import VisionTransformer_Orth, VisionTransformer, convert_to_orth_model

# First define a ViT
model = VisionTransformer()


# Then convert to baseline PriSM
blocks_orth = convert_to_orth_model( model.blocks, keep=0.8, fl=True )
prism_model = VisionTransformer_Orth(model, blocks_orth)

#train your model here ....

```

## Experiment Details and Analysis

### Results

Table 1 shows the reults compared with baselines. To optimize target FMs to target accuracy 95%, RaFFM outperform baselines with faster training speed, less communication cost, and less energy usage.

<table>
  <caption style="text-align: center;">Table 1. Comparison with baselines</caption>

  <tr>
    <th>Model</th>
    <th>Method</th>
    <th>Target Accuracy</th>
    <th>Training Accel. / Round</th>
    <th>Communication cost</th>
    <th>Peak Energy Usage</th>
  </tr>
  <tr>
    <td>ViT-base</td>
    <td>Full-size FM</td>
    <td>95%</td>
    <td>1.00×</td>
    <td>9.52GB</td>
    <td>24kWh</td>
  </tr>
  <tr>
    <td></td>
    <td>PruneFL</td>
    <td></td>
    <td>0.85×</td>
    <td>17.62GB</td>
    <td>28kWh</td>
  </tr>
  <tr>
    <td></td>
    <td>PriSM</td>
    <td></td>
    <td>0.75×</td>
    <td>10.92GB</td>
    <td>32kWh</td>
  </tr>
  <tr>
    <td></td>
    <td>RaFFM</td>
    <td></td>
    <td>2.12×</td>
    <td>5.31GB</td>
    <td>4.42kWh</td>
  </tr>
</table>

### Analysis

We recognize and value the foundational work in resource-efficient FL represented by PruneFL and PriSM. However, due to the differences in objectives and model focus, RaFFM shows significantly superior in applying FMs to FL senarios.

**Application Senarios**

First comparison with baselines, we have different senarios. Baselines achieve training acceleration by **zero-masking** partial parameters. This can accelerate training only in **paralell devices like GPUs**. In contrast, RaFFM directly operates on model architecture, accelerate on both **paralell devices and end-user devices**.

**Optimization Objectives**

Baseline method focus on train a large model in FL senarios only, and didn't consider the pre-trained knowledge in FMs. Masking out part of the weights without certain stratedy will destroy the pre-trained knowledge in FMs. In constrast, RaFFM with specialized model compression algorithm can retain such knowledge, and faster convergence (with only 5.31GB communication cost!).

**Post-FL**
Additionally, RaFFM considers post-FL model deployment, which enables heterogeneous model deployment post-federated learning without further training. (See experiment in [**post_training_deployment**](../post_training_deployment/README.MD))
