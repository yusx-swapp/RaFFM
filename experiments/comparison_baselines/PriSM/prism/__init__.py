from .model import (
    VisionTransformer,
    deit_small_patch16_224,
    deit_tiny_patch16_224,
    Block,
    Attention,
)
import torch
from timm.models.layers import PatchEmbed, Mlp
import copy
from .nn_transformer import Block_Orth, VisionTransformer_Orth, Mlp_Orth, Linear_Orth

import torch


def add_frob_decay(model, alpha=0.0001):
    """
    Add Frobenius decay to Conv_U/S/V layers
    Args:
        model: nn model definition
        alpha: decay coefficient

    Returns:

    """

    def __recursive_add_decay(module):
        for m in module.children():
            if isinstance(m, torch.nn.Sequential):
                __recursive_add_decay(m)
            else:
                # transformer layers
                if isinstance(m, Block_Orth):
                    __recursive_add_decay(m)
                elif isinstance(m, Mlp_Orth):
                    __recursive_add_decay(m)
                elif isinstance(m, Linear_Orth):
                    w_U, w_V = m.fc_U.weight.data, m.fc_V.weight.data
                    w_U *= m.mask_u.unsqueeze(1)
                    w_V *= m.mask_v
                    U_grad = torch.linalg.multi_dot((w_U.T, w_V.T, w_V))
                    V_grad = torch.linalg.multi_dot((w_U, w_U.T, w_V.T))

                    m.fc_U.weight.grad += alpha * U_grad.T
                    m.fc_V.weight.grad += alpha * V_grad.T

    __recursive_add_decay(model)


def convert_to_orth_model(model, keep, fl=False):
    """Convert normal model to model with orthogonal channels
    :param model original model definition
    :param keep channel keep ratio
    :param fl fl or centralized setting
    :return orthogonal model
    """
    model_orth = []

    def __convert_layer(module, in_sequential=False):
        module_new = []
        for m in module.children():
            if isinstance(m, torch.nn.Sequential):
                module_new = __convert_layer(m, in_sequential=True)
                model_orth.append(torch.nn.Sequential(*module_new))
            else:
                # transformer layers
                if isinstance(m, PatchEmbed):
                    m_orth = copy.deepcopy(m)
                elif isinstance(m, Block):
                    m_orth = Block_Orth(m, keep=keep)
                elif isinstance(m, torch.nn.LayerNorm):
                    m_orth = copy.deepcopy(m)

                else:
                    m_orth = copy.deepcopy(m)
                if in_sequential:
                    module_new.append(m_orth)
                else:
                    model_orth.append(m_orth)

        return module_new

    __convert_layer(model)

    return torch.nn.Sequential(*model_orth)
