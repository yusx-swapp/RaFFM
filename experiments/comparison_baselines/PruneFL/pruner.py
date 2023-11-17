"""
/***************************************************************************************
*    Title: PruneFL (Our implementation)
*    Description:  Implementation for PruneFL with threashold pruning for ViT
*    Original Repo: https://github.com/jiangyuang/PruneFL/tree/master
*   
*    Paper: Model pruning enables efficient federated learning on edge devices. IEEE Transactions on Neural Networks and Learning Systems.
*    
*
***************************************************************************************/
"""

import torch
from torch import nn
import numpy as np
import torch
import torch.nn as nn
import torch.sparse as sparse
import torch.nn.init as init
from torch.nn.parameter import Parameter


def _sparse_masked_select_abs(module, sparse_tensor: sparse.FloatTensor, thr):
    indices = sparse_tensor._indices()
    values = sparse_tensor._values()
    prune_mask = torch.abs(values) >= thr
    return torch.sparse_coo_tensor(
        indices=indices.masked_select(prune_mask).reshape(2, -1),
        values=values.masked_select(prune_mask),
        size=[module.out_features, module.in_features],
    ).coalesce()


def prune_by_threshold(module, thr):
    module.weight = Parameter(_sparse_masked_select_abs(module, module.weight, thr))


def prune_by_rank(module, rank):
    weight_val = module.weight._values()
    sorted_abs_weight = torch.sort(torch.abs(weight_val))[0]
    thr = sorted_abs_weight[rank]
    prune_by_threshold(module, thr)


def prune_by_pct(module, pct):
    if pct == 0:
        return
    prune_idx = int(module.weight._nnz() * pct)
    prune_by_rank(module, prune_idx)


def prunefl(model, threshold=1e-3):
    # Prune all the conv layers of the model
    layer_wise_sparsity = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            mask = torch.abs(module.weight) > threshold
            module.weight.data[~mask] = 0
            module_sparsity = 1 - float(torch.sum(module.weight.data != 0)) / float(
                module.weight.data.nelement()
            )
            layer_wise_sparsity.append(module_sparsity)
            # print('Layer: {}, Sparsity: {:.2f}%'.format(name, module_sparsity*100))

    model_sparsity = np.mean(layer_wise_sparsity)
    # print('Model sparsity: {:.2f}%'.format(model_sparsity*100))

    return model_sparsity
