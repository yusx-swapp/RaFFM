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

To run the baselines **PriSM** you can simply run the following commands:

```bash
CUDA_VISIBLE_DEVICES=0 python prism_fl.py
```

Similarly, to run the baselines **PruneLF** you can simply run the following commands:

```bash
CUDA_VISIBLE_DEVICES=0 python prunefl_fl.py
```

The full hyper-parameters lists and arguments can be find at **[`arguments.py`](arguments.py)**.

### Low-level APIs

You can also reproduce the baseline reults in your own senarios.
For instance, converts a ViT to PriSM model:

```python
from PriSM import VisionTransformer_Orth, VisionTransformer

# First define a ViT
model = VisionTransformer()

# Then convert to baseline PriSM
prism_model = VisionTransformer_Orth(model, model.blocks)

# Then convert to baseline PriSM
prism_model = VisionTransformer_Orth(model)

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
