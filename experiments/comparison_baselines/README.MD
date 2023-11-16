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

We have conducted the experiments to optimize
**Application Senarios**

| Model    | Method       | Target Accuracy | Training Accel. / Round | Communication cost | Peak Energy Usage |
| -------- | ------------ | --------------- | ----------------------- | ------------------ | ----------------- |
| ViT-base | Full-size FM | 95%             | 1.00×                   | 9.52GB             | 24kWh             |
|          | PruneFL      |                 | 0.85×                   | 17.62GB            | 28kWh             |
|          | PriSM        |                 | 0.75×                   | 10.92GB            | 32kWh             |
|          | RaFFM        |                 | 2.12×                   | 5.31GB             | 4.42kWh           |
