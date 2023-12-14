import numpy as np
import copy
from torch import nn
import torch
import time
from .utils import calculate_params
from peft import (
    PeftModel,
    PeftConfig,
    inject_adapter_in_model,
)


__all__ = [
    "bert_module_handler",
    "vit_module_handler",
    "vit_peft_module_handler",
    "arc_config_sampler",
    "sam_module_handler",
]


@staticmethod
def copy_weights_to_subnet(subnet, org_model):
    """
    Copies the weights from original foundation model to scaled subnet where the parameter names match.
    Only the overlapping parts of the weights are copied when the dimensions in the subnet
    are less than or equal to those in the larger model.

    Parameters:
    subnet (torch.nn.Module): The smaller model to which the weights will be copied.
    org_model (torch.nn.Module): The foundation model from which the weights will be sourced.

    Usage:
    This function is useful in extract subnet from pre-trained foundation model scenarios where a smaller model is initialized
    with weights from certain layers of a larger, pre-trained model.
    """

    for sm_param_name, sm_param in subnet.named_parameters():
        if sm_param_name in dict(org_model.named_parameters()):
            lg_param = dict(org_model.named_parameters())[sm_param_name]
            if all(
                sm_dim <= lg_dim
                for sm_dim, lg_dim in zip(sm_param.shape, lg_param.shape)
            ):
                # Create a slice object for each dimension to copy the corresponding weights
                slices = tuple(
                    slice(0, min(sm_dim, lg_dim))
                    for sm_dim, lg_dim in zip(sm_param.shape, lg_param.shape)
                )
                sm_param.data.copy_(lg_param.data[slices])


@staticmethod
def check_weight_copy_correctness(subnet, org_model):
    """
    Checks if the weights have been correctly copied from the larger model to the smaller model.

    Parameters:
    smaller_model (torch.nn.Module): The smaller model with copied weights.
    larger_model (torch.nn.Module): The larger model from which the weights were sourced.

    Returns:
    bool: True if the weights are correctly copied, False otherwise.

    Usage:
    Useful for verifying the correctness of a weight copying process in model adaptation or transfer learning.
    """

    for sm_param_name, sm_param in subnet.named_parameters():
        if sm_param_name in dict(org_model.named_parameters()):
            lg_param = dict(org_model.named_parameters())[sm_param_name]

            # Compare shapes
            if not all(
                sm_dim <= lg_dim
                for sm_dim, lg_dim in zip(sm_param.shape, lg_param.shape)
            ):
                return False

            # Compare values
            slices = tuple(
                slice(0, min(sm_dim, lg_dim))
                for sm_dim, lg_dim in zip(sm_param.shape, lg_param.shape)
            )
            if not torch.all(sm_param == lg_param[slices]):
                return False

    return True


@staticmethod
def arc_config_sampler(
    atten_out_space: list[int],
    inter_hidden_space: list[int],
    residual_hidden_space: list[int],
    n_layer=12,
    smallest=False,
) -> dict:
    """Generate subnet architecture configuration based on the provided configuration.

    Args:
        atten_out_space (list[int]): Attention head output hidden size space, NOT the hidden space.
        inter_hidden_space (list[int]): Intermediate dense hidden layer size space.
        residual_hidden_space (list[int]): Attention (input size) and Intermediate layer (out size) hidden size.
        n_layer (int, optional): Number of multi-head attention layers. Defaults to 12.
        smallest (bool, optional): Either return smallest subnet configuration. Defaults to False.

    Returns:
        dic: Subnet architecture configure.
    """
    arc_config = {}
    np.random.seed(int(time.time()))  # Set the seed to the current time

    residual_hidden = np.random.choice(residual_hidden_space)
    if smallest:
        residual_hidden = min(residual_hidden_space)

    for layer in range(n_layer):
        if smallest:
            inter_hidden = min(inter_hidden_space)
            atten_out = min(atten_out_space)

        else:
            inter_hidden = np.random.choice(inter_hidden_space)
            atten_out = np.random.choice(atten_out_space)

        arc_config[f"layer_{layer + 1}"] = {
            "atten_out": atten_out,
            "inter_hidden": inter_hidden,
            "residual_hidden": residual_hidden,
        }

    return arc_config


@staticmethod
def bert_module_handler(model, arc_config):
    from transformers.models.bert.modeling_bert import (
        BertSelfAttention,
        BertSelfOutput,
        BertIntermediate,
        BertOutput,
        BertEmbeddings,
        BertPooler,
    )
    from transformers import BertConfig

    BertLayerNorm = nn.LayerNorm

    class BertSelfAttention(BertSelfAttention):
        def __init__(self, config):
            super().__init__(config)

            self.num_attention_heads = config.num_attention_heads
            self.attention_head_size = config.attention_head_size
            self.all_head_size = self.num_attention_heads * self.attention_head_size

            self.query = nn.Linear(config.hidden_size, self.all_head_size)
            self.key = nn.Linear(config.hidden_size, self.all_head_size)
            self.value = nn.Linear(config.hidden_size, self.all_head_size)

            self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    class BertSelfOutput(BertSelfOutput):
        def __init__(self, config):
            super().__init__(config)
            self.dense = nn.Linear(
                config.attention_head_size * config.num_attention_heads,
                config.hidden_size,
            )
            self.LayerNorm = BertLayerNorm(
                config.hidden_size, eps=config.layer_norm_eps
            )

    class BertIntermediate(BertIntermediate):
        def __init__(self, config):
            super().__init__(config)
            self.dense = nn.Linear(config.hidden_size, config.intermediate_size)

    class BertOutput(BertOutput):
        def __init__(self, config):
            super().__init__(config)
            self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
            self.LayerNorm = BertLayerNorm(
                config.hidden_size, eps=config.layer_norm_eps
            )

    subnetwork = copy.deepcopy(model).cpu()

    bert_layers = subnetwork.bert.encoder.layer

    new_config = BertConfig.from_dict(model.config.to_dict())

    for i, (layer, key) in enumerate(zip(bert_layers, arc_config)):
        arc = arc_config[key]
        # new_config.hidden_size = arc  # Set to the new output dimension
        new_config.attention_head_size = (
            arc["atten_out"] // new_config.num_attention_heads
        )  # Ensure it divides evenly
        new_config.intermediate_size = arc["inter_hidden"]
        new_config.hidden_size = arc["residual_hidden"]

        new_attention_layer = BertSelfAttention(config=new_config)
        new_out_layer = BertSelfOutput(config=new_config)
        new_inter_layer = BertIntermediate(config=new_config)
        new_dens_out_layer = BertOutput(config=new_config)

        layer.attention.self = new_attention_layer
        layer.attention.output = new_out_layer
        layer.intermediate = new_inter_layer
        layer.output = new_dens_out_layer

    new_embeddings = BertEmbeddings(new_config)
    new_pooler = BertPooler(new_config)
    new_classifier = nn.Linear(new_config.hidden_size, model.classifier.out_features)

    subnetwork.bert.embeddings = new_embeddings
    subnetwork.bert.pooler = new_pooler
    subnetwork.classifier = new_classifier

    subnetwork.config = new_config
    copy_weights_to_subnet(subnetwork, model)

    total_params = calculate_params(subnetwork)

    return subnetwork, total_params


@staticmethod
def vit_module_handler(model, arc_config):
    from transformers.models.vit.modeling_vit import (
        ViTSelfAttention,
        ViTSelfOutput,
        ViTIntermediate,
        ViTOutput,
        ViTEmbeddings,
    )
    from transformers import ViTConfig
    from torch import nn

    class ViTSelfAttention(ViTSelfAttention):
        def __init__(self, config: ViTConfig):
            super().__init__(config)

            self.num_attention_heads = config.num_attention_heads
            self.attention_head_size = config.attention_head_size
            self.all_head_size = self.num_attention_heads * self.attention_head_size

            self.query = nn.Linear(
                config.hidden_size, self.all_head_size, bias=config.qkv_bias
            )
            self.key = nn.Linear(
                config.hidden_size, self.all_head_size, bias=config.qkv_bias
            )
            self.value = nn.Linear(
                config.hidden_size, self.all_head_size, bias=config.qkv_bias
            )

            self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    class ViTSelfOutput(ViTSelfOutput):
        def __init__(self, config: ViTConfig):
            super().__init__(config)
            self.dense = nn.Linear(
                config.attention_head_size * config.num_attention_heads,
                config.hidden_size,
            )
            self.dropout = nn.Dropout(config.hidden_dropout_prob)

    class ViTIntermediate(ViTIntermediate):
        def __init__(self, config: ViTConfig):
            super().__init__(config)
            self.dense = nn.Linear(config.hidden_size, config.intermediate_size)

    class ViTOutput(ViTOutput):
        def __init__(self, config):
            super().__init__(config)
            self.dense = nn.Linear(config.intermediate_size, config.hidden_size)

    subnetwork = copy.deepcopy(model).cpu()

    vit_layers = subnetwork.vit.encoder.layer
    new_config = ViTConfig.from_dict(model.config.to_dict())

    for i, (layer, key) in enumerate(zip(vit_layers, arc_config)):
        arc = arc_config[key]
        # new_config.hidden_size = arc  # Set to the new output dimension
        new_config.attention_head_size = (
            arc["atten_out"] // new_config.num_attention_heads
        )  # Ensure it divides evenly
        new_config.intermediate_size = arc["inter_hidden"]
        new_config.hidden_size = arc["residual_hidden"]

        new_attention_layer = ViTSelfAttention(config=new_config)
        new_out_layer = ViTSelfOutput(config=new_config)
        new_inter_layer = ViTIntermediate(config=new_config)
        new_dens_out_layer = ViTOutput(config=new_config)
        layernorm_before = nn.LayerNorm(
            new_config.hidden_size, eps=new_config.layer_norm_eps
        )
        layernorm_after = nn.LayerNorm(
            new_config.hidden_size, eps=new_config.layer_norm_eps
        )

        layer.attention.attention = new_attention_layer
        layer.attention.output = new_out_layer
        layer.intermediate = new_inter_layer
        layer.output = new_dens_out_layer
        layer.layernorm_before = layernorm_before
        layer.layernorm_after = layernorm_after

    new_embeddings = ViTEmbeddings(new_config)
    new_layernorm = nn.LayerNorm(new_config.hidden_size, eps=new_config.layer_norm_eps)
    new_classifier = nn.Linear(new_config.hidden_size, model.classifier.out_features)

    subnetwork.vit.embeddings = new_embeddings
    subnetwork.vit.layernorm = new_layernorm
    subnetwork.classifier = new_classifier

    subnetwork.config = new_config
    copy_weights_to_subnet(subnetwork, model)

    total_params = calculate_params(subnetwork)

    return subnetwork, total_params


@staticmethod
def sam_module_handler(model, arc_config):
    from transformers.models.sam.modeling_sam import (
        SamVisionAttention,
        SamMLPBlock,
        SamVisionLayer,
    )
    from transformers import SamVisionConfig

    sub_model = copy.deepcopy(model).cpu()
    vision_encoder = copy.deepcopy(sub_model.vision_encoder).cpu()

    sam_vit_layers = vision_encoder.layers

    class SamVisionAttention(SamVisionAttention):
        def __init__(self, config, window_size):
            import torch

            super().__init__(config, window_size)
            input_size = (
                (
                    config.image_size // config.patch_size,
                    config.image_size // config.patch_size,
                )
                if window_size == 0
                else (window_size, window_size)
            )

            self.num_attention_heads = config.num_attention_heads
            # head_dim = config.hidden_size // config.num_attention_heads
            head_dim = config.attention_head_size

            self.scale = head_dim**-0.5
            self.dropout = config.attention_dropout

            self.qkv = nn.Linear(
                config.hidden_size,
                head_dim * self.num_attention_heads * 3,
                bias=config.qkv_bias,
            )
            self.proj = nn.Linear(
                head_dim * self.num_attention_heads, config.hidden_size
            )

            self.use_rel_pos = config.use_rel_pos
            if self.use_rel_pos:
                if input_size is None:
                    raise ValueError(
                        "Input size must be provided if using relative positional encoding."
                    )

                # initialize relative positional embeddings
                self.rel_pos_h = nn.Parameter(
                    torch.zeros(2 * input_size[0] - 1, head_dim)
                )
                self.rel_pos_w = nn.Parameter(
                    torch.zeros(2 * input_size[1] - 1, head_dim)
                )

    class SamMLPBlock(SamMLPBlock):
        def __init__(self, config):
            super().__init__(config)

            self.lin1 = nn.Linear(config.hidden_size, config.mlp_dim)
            self.lin2 = nn.Linear(config.mlp_dim, config.hidden_size)

    for i, (layer, key) in enumerate(zip(sam_vit_layers, arc_config)):
        arc = arc_config[key]
        new_config = SamVisionConfig.from_dict(vision_encoder.config.to_dict())

        # new_config.hidden_size = arc  # Set to the new output dimension
        new_config.attention_head_size = (
            arc["atten_out"] // new_config.num_attention_heads
        )  # Ensure it divides evenly

        new_config.mlp_dim = arc["inter_hidden"]
        new_attention_layer = SamVisionAttention(
            config=new_config,
            window_size=new_config.window_size
            if i not in new_config.global_attn_indexes
            else 0,
        )

        new_mlp = SamMLPBlock(config=new_config)

        layer.attn = new_attention_layer
        layer.mlp = new_mlp

    sub_model.vision_encoder = vision_encoder
    copy_weights_to_subnet(sub_model, model)
    total_params = calculate_params(sub_model)

    return sub_model, total_params


@staticmethod
def vit_peft_module_handler(model: PeftModel, peft_config: PeftConfig, arc_config):
    from transformers.models.vit.modeling_vit import (
        ViTSelfAttention,
        ViTSelfOutput,
        ViTIntermediate,
        ViTOutput,
    )
    from transformers import ViTConfig

    class ViTSelfAttention(ViTSelfAttention):
        def __init__(self, config: ViTConfig):
            super().__init__(config)

            self.num_attention_heads = config.num_attention_heads
            self.attention_head_size = config.attention_head_size
            self.all_head_size = self.num_attention_heads * self.attention_head_size

            self.query = nn.Linear(
                config.hidden_size, self.all_head_size, bias=config.qkv_bias
            )
            self.key = nn.Linear(
                config.hidden_size, self.all_head_size, bias=config.qkv_bias
            )
            self.value = nn.Linear(
                config.hidden_size, self.all_head_size, bias=config.qkv_bias
            )

            self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    class ViTSelfOutput(ViTSelfOutput):
        def __init__(self, config: ViTConfig):
            super().__init__(config)
            self.dense = nn.Linear(
                config.attention_head_size * config.num_attention_heads,
                config.hidden_size,
            )
            self.dropout = nn.Dropout(config.hidden_dropout_prob)

    class ViTIntermediate(ViTIntermediate):
        def __init__(self, config: ViTConfig):
            super().__init__(config)
            self.dense = nn.Linear(config.hidden_size, config.intermediate_size)

    class ViTOutput(ViTOutput):
        def __init__(self, config):
            super().__init__(config)
            self.dense = nn.Linear(config.intermediate_size, config.hidden_size)

    subnetwork = copy.deepcopy(model).cpu()

    vit_layers = subnetwork.vit.encoder.layer

    for i, (layer, key) in enumerate(zip(vit_layers, arc_config)):
        arc = arc_config[key]
        new_config = ViTConfig.from_dict(model.config.to_dict())
        # new_config.hidden_size = arc  # Set to the new output dimension
        new_config.attention_head_size = (
            arc["atten_out"] // new_config.num_attention_heads
        )  # Ensure it divides evenly
        new_config.intermediate_size = arc["inter_hidden"]
        new_attention_layer = ViTSelfAttention(config=new_config).requires_grad_(False)
        new_out_layer = ViTSelfOutput(config=new_config).requires_grad_(False)
        new_inter_layer = ViTIntermediate(config=new_config).requires_grad_(False)
        new_dens_out_layer = ViTOutput(config=new_config).requires_grad_(False)

        if any(
            item in peft_config.target_modules for item in ["query", "key", "value"]
        ):
            new_attention_layer = inject_adapter_in_model(
                peft_config, new_attention_layer
            )

        if "dense" in peft_config.target_modules:
            new_out_layer = inject_adapter_in_model(peft_config, new_out_layer)
            new_inter_layer = inject_adapter_in_model(peft_config, new_inter_layer)
            new_dens_out_layer = inject_adapter_in_model(
                peft_config, new_dens_out_layer
            )

        load_subnet_state_dict(new_attention_layer, layer.attention.attention)
        load_subnet_state_dict(new_out_layer, layer.attention.output)
        load_subnet_state_dict(new_inter_layer, layer.intermediate)
        load_subnet_state_dict(new_dens_out_layer, layer.output)

        layer.attention.attention = new_attention_layer
        layer.attention.output = new_out_layer
        layer.intermediate = new_inter_layer
        layer.output = new_dens_out_layer

    # total_params = calculate_params(subnetwork)
    trainable_params, all_param = subnetwork.get_nb_trainable_parameters()

    return subnetwork, trainable_params


def _test():
    # print(model.config)
    atten_out_space = [768 - i * 24 for i in range(0, 10)]
    inter_hidden_space = [3072 - i * 128 for i in range(0, 25)]
    out_hidden_space = [768 - i * 24 for i in range(0, 10)]
    arc_config = arc_config_sampler(
        atten_out_space, inter_hidden_space, out_hidden_space
    )
    # submodel,total_params,_,_ = bert_module_handler(
    #     model, arc_config
    # )

    # print(submodel)
