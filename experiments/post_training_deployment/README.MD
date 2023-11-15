# Post-training High-performance Resource-aware Model deployment

We can quickly get a specialized scaled FM (sub-network) by selecting from the trained RaFFM foundation model without additional training. It can obtain a surprisingly large number of sub-networks (> $10^{10}$) that can fit different hardware platforms and latency constraints while maintaining the same level of accuracy as training independently.

You can find out the experiments on the jupyter notebook: [fm_scaling.ipynb](./fm_scaling.ipynb)
