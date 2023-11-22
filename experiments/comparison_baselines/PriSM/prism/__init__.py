from .model import VisionTransformer, deit_small_patch16_224, Block, Attention
import torch
from timm.models.layers import PatchEmbed, Mlp
import copy
from .nn_transformer import Block_Orth, VisionTransformer_Orth


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
