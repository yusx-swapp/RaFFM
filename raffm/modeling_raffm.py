""" Official Implementation
Foundation Model Scaling
"""

import os
import torch
from .model_scaling import (
    bert_module_handler,
    arc_config_sampler,
    vit_module_handler,
    vit_peft_module_handler,
)
from .param_prioritization import *
from .utils import calculate_params, save_dict_to_file, load_dict_from_file
from peft import PeftConfig, PeftModel

# __all__ = ["salient_submodel_extraction"]


class RaFFM:
    def __init__(self, model, elastic_config=None) -> None:
        self.model = model
        self.total_params = calculate_params(model=model)

        if not elastic_config:
            # set defalt search space configuration (this is defalt setting for bert)
            elastic_config = {
                "atten_out_space": [768 - i * 12 for i in range(0, 15)],
                # "atten_out_space": [768],
                "inter_hidden_space": [3072 - i * 64 for i in range(0, 20)],
                "out_hidden_space": [768 - i * 24 for i in range(0, 15)],
            }
            print(
                f"[Warning]: No elastic configuration provides. Set to the defalt elastic space {elastic_config}."
            )
        elif isinstance(elastic_config, str):
            elastic_config = load_dict_from_file(elastic_config)

        assert isinstance(
            elastic_config, dict
        ), "Invalid elastic_config, expect input a dictionary or file path"

        self.elastic_config = elastic_config

    def random_resource_aware_model(self):
        """Sample random resource awrae model from original model

        Raises:
            NotImplementedError: _description_

        Returns:
            _type_: _description_
        """
        arc_config = arc_config_sampler(**self.elastic_config)
        if "bert" == self.model.config.model_type.lower():
            subnetwork, total_params = bert_module_handler(self.model, arc_config)
        elif "vit" == self.model.config.model_type.lower():
            subnetwork, total_params = vit_module_handler(self.model, arc_config)
        else:
            raise NotImplementedError
        return subnetwork, total_params, arc_config

    def sample_smallest_model(self):
        arc_config = arc_config_sampler(**self.elastic_config, smallest=True)
        return self.resource_aware_model(arc_config), arc_config

    def resource_aware_model(self, arc_config):
        if "bert" == self.model.config.model_type.lower():
            return bert_module_handler(self.model, arc_config)
        elif "vit" == self.model.config.model_type.lower():
            return vit_module_handler(self.model, arc_config)
        else:
            raise NotImplementedError

    def salient_parameter_prioritization(self, metric=l1_norm):
        self.model = salient_parameter_prioritization(self.model, metric)

    def aggregate(self, local_models):
        """Aggregate local weights via fedavg

        Args:
            local_models (_type_): _description_
        """
        self.model.to("cpu")
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                param *= 0
                for local_model in local_models:
                    local_param = local_model.state_dict()[name].cpu()
                    if len(local_param.shape) == 2:
                        param[
                            : local_param.shape[0], : local_param.shape[1]
                        ] += local_param / len(local_models)
                    else:
                        param[: local_param.shape[0]] += local_param / len(local_models)

    def save_ckpt(self, dir):
        self.model.save_pretrained(os.path.join(dir))
        save_dict_to_file(self.elastic_config, os.path.join(dir, "elastic_space.json"))

    def load_ckpt(self, dir):
        self.model.from_pretrained(dir)

        if os.path.exists(os.path.join(dir, "elastic_space.json")):
            self.elastic_config = load_dict_from_file(
                os.path.join(dir, "elastic_space.json")
            )


class RaPEFT(RaFFM):
    def __init__(
        self, model: PeftModel, elastic_config=None, peft_config: PeftConfig = None
    ) -> None:
        super().__init__(model, elastic_config)
        if peft_config:
            self.peft_config = peft_config
        else:
            self.peft_config = model.peft_config["default"]
            print(
                f"[Warning]: No peft_config configuration provides. Set to the peft_config as the peft model {self.peft_config}."
            )

    def random_peft_model(self):
        arc_config = arc_config_sampler(**self.elastic_config)

        if "bert" == self.model.config.model_type.lower():
            raise NotImplementedError
        elif "vit" == self.model.config.model_type.lower():
            subnetwork, trainable_params = vit_peft_module_handler(
                self.model, self.peft_config, arc_config
            )
        else:
            raise NotImplementedError
        return subnetwork, trainable_params, arc_config

    def aggregate(self, local_models):
        """Aggregate local weights via fedavg

        Args:
            local_models (_type_): _description_
        """
        self.model.to("cpu")
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                if param.requires_grad:
                    param *= 0
                    for local_model in local_models:
                        local_param = local_model.state_dict()[name].cpu()
                        if len(local_param.shape) == 2:
                            param[
                                : local_param.shape[0], : local_param.shape[1]
                            ] += local_param / len(local_models)
                        else:
                            param[: local_param.shape[0]] += local_param / len(
                                local_models
                            )

    def save_ckpt(self, dir):
        super().save_ckpt(dir)
