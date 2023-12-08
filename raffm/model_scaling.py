import numpy as np
import copy
from torch import nn
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
def load_subnet_state_dict(sub_model, org_model):
    for (sub_name, sub_param), (org_name, org_param) in zip(
        sub_model.named_parameters(), org_model.named_parameters()
    ):
        if len(sub_param.shape) == 2:
            truncated_weight = org_param.data[
                : sub_param.shape[0], : sub_param.shape[1]
            ]
        else:
            truncated_weight = org_param.data[: sub_param.shape[0]]
        sub_param.data.copy_(truncated_weight)


@staticmethod
def arc_config_sampler(
    atten_out_space,
    inter_hidden_space,
    out_hidden_space,
    n_layer=12,
    embedding_size=768,
    model_out_hidden=768,
    smallest=False,
):
    """_summary_

    Args:
        atten_out_space (_type_): _description_
        inter_hidden_space (_type_): _description_
        out_hidden_space (_type_): _description_
        n_layer (int, optional): _description_. Defaults to 12.
        embedding_size (int, optional): _description_. Defaults to 768.
        model_out_hidden (int, optional): _description_. Defaults to 768.

    Returns:
        _type_: _description_
    """
    arc_config = {}
    np.random.seed(int(time.time()))  # Set the seed to the current time
    for layer in range(n_layer):
        if layer == 0:
            pre_hidden = embedding_size
        else:
            pre_hidden = arc_config[f"layer_{layer}"]["out_hidden"]

        if layer == n_layer - 1:
            out_hidden = model_out_hidden
        else:
            out_hidden = np.random.choice(out_hidden_space)
            if smallest:
                out_hidden = min(out_hidden_space)

        if smallest:
            inter_hidden = min(inter_hidden_space)
            atten_out = min(atten_out_space)

        else:
            inter_hidden = np.random.choice(inter_hidden_space)
            atten_out = np.random.choice(atten_out_space)

        arc_config[f"layer_{layer + 1}"] = {
            "atten_out": atten_out,
            "pre_hidden": pre_hidden,
            "inter_hidden": inter_hidden,
            "out_hidden": out_hidden,
        }

    return arc_config


@staticmethod
def bert_module_handler(model, arc_config):
    from transformers.models.bert.modeling_bert import (
        BertSelfAttention,
        BertSelfOutput,
        BertIntermediate,
        BertOutput,
    )
    from transformers import BertConfig

    BertLayerNorm = nn.LayerNorm

    class NewBertSelfAttention(BertSelfAttention):
        def __init__(self, config):
            super().__init__(config)

            self.num_attention_heads = config.num_attention_heads
            self.attention_head_size = config.attention_head_size
            self.all_head_size = self.num_attention_heads * self.attention_head_size

            self.query = nn.Linear(config.hidden_size, self.all_head_size)
            self.key = nn.Linear(config.hidden_size, self.all_head_size)
            self.value = nn.Linear(config.hidden_size, self.all_head_size)

            self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    class NewBertSelfOutput(BertSelfOutput):
        def __init__(self, config):
            super().__init__(config)
            self.dense = nn.Linear(
                config.attention_head_size * config.num_attention_heads,
                config.hidden_size,
            )
            self.LayerNorm = BertLayerNorm(
                config.hidden_size, eps=config.layer_norm_eps
            )

    class NewBertIntermediate(BertIntermediate):
        def __init__(self, config):
            super().__init__(config)
            self.dense = nn.Linear(config.hidden_size, config.intermediate_size)

    class NewBertOut(BertOutput):
        def __init__(self, config):
            super().__init__(config)
            self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
            self.LayerNorm = BertLayerNorm(
                config.hidden_size, eps=config.layer_norm_eps
            )

    subnetwork = copy.deepcopy(model).cpu()

    bert_layers = subnetwork.bert.encoder.layer

    for i, (layer, key) in enumerate(zip(bert_layers, arc_config)):
        arc = arc_config[key]
        new_config = BertConfig.from_dict(model.config.to_dict())
        # new_config.hidden_size = arc  # Set to the new output dimension
        new_config.attention_head_size = (
            arc["atten_out"] // new_config.num_attention_heads
        )  # Ensure it divides evenly
        new_config.intermediate_size = arc["inter_hidden"]
        new_attention_layer = NewBertSelfAttention(config=new_config)
        new_out_layer = NewBertSelfOutput(config=new_config)
        new_inter_layer = NewBertIntermediate(config=new_config)
        new_dens_out_layer = NewBertOut(config=new_config)

        load_subnet_state_dict(new_attention_layer, layer.attention.self)
        load_subnet_state_dict(new_out_layer, layer.attention.output)
        load_subnet_state_dict(new_inter_layer, layer.intermediate)
        load_subnet_state_dict(new_dens_out_layer, layer.output)

        layer.attention.self = new_attention_layer
        layer.attention.output = new_out_layer
        layer.intermediate = new_inter_layer
        layer.output = new_dens_out_layer

    total_params = calculate_params(subnetwork)

    return subnetwork, total_params


@staticmethod
def vit_module_handler(model, arc_config):
    from transformers.models.vit.modeling_vit import (
        ViTSelfAttention,
        ViTSelfOutput,
        ViTIntermediate,
        ViTOutput,
    )
    from transformers import ViTConfig

    class NewViTSelfAttention(ViTSelfAttention):
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

    class NewViTSelfOutput(ViTSelfOutput):
        def __init__(self, config: ViTConfig):
            super().__init__(config)
            self.dense = nn.Linear(
                config.attention_head_size * config.num_attention_heads,
                config.hidden_size,
            )
            self.dropout = nn.Dropout(config.hidden_dropout_prob)

    class NewViTIntermediate(ViTIntermediate):
        def __init__(self, config: ViTConfig):
            super().__init__(config)
            self.dense = nn.Linear(config.hidden_size, config.intermediate_size)

    class NewViTOutput(ViTOutput):
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
        new_attention_layer = NewViTSelfAttention(config=new_config)
        new_out_layer = NewViTSelfOutput(config=new_config)
        new_inter_layer = NewViTIntermediate(config=new_config)
        new_dens_out_layer = NewViTOutput(config=new_config)

        load_subnet_state_dict(new_attention_layer, layer.attention.attention)
        load_subnet_state_dict(new_out_layer, layer.attention.output)
        load_subnet_state_dict(new_inter_layer, layer.intermediate)
        load_subnet_state_dict(new_dens_out_layer, layer.output)

        layer.attention.attention = new_attention_layer
        layer.attention.output = new_out_layer
        layer.intermediate = new_inter_layer
        layer.output = new_dens_out_layer

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

        load_subnet_state_dict(new_attention_layer, layer.attn)
        load_subnet_state_dict(new_mlp, layer.mlp)

        layer.attn = new_attention_layer
        layer.mlp = new_mlp

    sub_model.vision_encoder = vision_encoder
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
