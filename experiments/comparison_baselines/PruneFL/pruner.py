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
import copy
import torch
from torch import nn
import numpy as np
import torch
import torch.nn as nn


def prune_by_threshold(module, threshold):
    """
    Prunes the weights of a neural network module by applying a threshold.

    This function sets the weights of the module to zero if their absolute value
    is less than the specified threshold, effectively pruning the module.

    Parameters:
    - module (nn.Module): The neural network module whose weights are to be pruned.
    - threshold (float): The threshold value. Weights with an absolute value less than this are pruned.

    Returns:
    - float: The sparsity of the module after pruning, calculated as the proportion of weights that are zero.
    """
    mask = torch.abs(module.weight) > threshold
    module.weight.data[~mask] = 0
    module_sparsity = 1 - float(torch.sum(module.weight.data != 0)) / float(
        module.weight.data.nelement()
    )
    return module_sparsity


def prune_by_rank(module, rank=0.8):
    """
    Prunes the weights of a neural network module based on a rank.

    This function retains only a certain percentage of the top weights by their absolute value,
    setting all other weights to zero. This helps in keeping only the most significant weights.

    Parameters:
    - module (nn.Module): The neural network module whose weights are to be pruned.
    - rank (float): The percentage of weights to retain. Weights are ranked by their absolute values.

    Returns:
    - float: The sparsity of the module after pruning, calculated as the proportion of weights that are zero.
    """

    total_weights = module.weight.data.numel()
    rank = int(total_weights * rank)
    flat_weights = torch.abs(module.weight.data.view(-1))
    threshold = torch.kthvalue(flat_weights, total_weights - rank).values.item()

    # Create a mask where weights greater than the threshold are kept
    mask = torch.abs(module.weight) > threshold

    # Apply the mask to the weights, setting the others to zero
    module.weight.data[~mask] = 0

    # Calculate the sparsity (proportion of weights that are zero)
    module_sparsity = 1 - float(torch.sum(module.weight.data != 0)) / float(
        module.weight.data.nelement()
    )

    return module_sparsity


def prunefl(model, rank=0.8, threshold=1e-2):
    """
    Applies pruning to a neural network model based on both threshold and rank.

    This function iterates over all linear layers of the model and applies the prune_by_threshold
    function to each layer. It creates and returns a copy of the model with pruned layers.

    Parameters:
    - model (nn.Module): The neural network model to be pruned.
    - threshold (float, optional): The threshold value for pruning. Default is 1e-2.

    Returns:
    - nn.Module: A copy of the model with pruned layers.
    - float: The average sparsity across all pruned layers in the model.
    """
    pruned_model = copy.deepcopy(model)
    # Prune all the conv layers of the model
    layer_wise_sparsity = []
    for name, module in model.named_modules():
        if isinstance(
            module, nn.Linear
        ):  # Beacuse attention and dense layer are implement as Linear layer in Vit
            if "dense" in name:
                module_sparsity = prune_by_rank(module, rank)
            else:
                module_sparsity = prune_by_threshold(module, threshold)
            layer_wise_sparsity.append(module_sparsity)
            # print('Layer: {}, Sparsity: {:.2f}%'.format(name, module_sparsity*100))

    model_sparsity = np.mean(layer_wise_sparsity)
    # print('Model sparsity: {:.2f}%'.format(model_sparsity*100))

    return pruned_model, model_sparsity
