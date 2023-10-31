""" Official Implementation
Foundation Model Scaling
"""

import os
import torch
from .model_scaling import bert_module_handler, arc_config_sampler, vit_module_handler
from .param_prioritization import *
from .utils import calculate_params

# __all__ = ["salient_submodel_extraction"]


class RaFFM:
    def __init__(self, model, elastic_config=None) -> None:
        self.model = model
        self.total_params = calculate_params(model=model)

        if not elastic_config:
            # set defalt search space configuration (this is defalt setting for bert)
            elastic_config = {
                # "atten_out_space": [768 - i * 12 for i in range(0, 15)],
                "atten_out_space": [768],
                "inter_hidden_space": [3072 - i * 64 for i in range(0, 20)],
                "out_hidden_space": [768 - i * 24 for i in range(0, 15)],
            }
            print(
                f"[Warning]: No elastic configuration provides. Set to the defalt elastic space {elastic_config}."
            )
        elif isinstance(elastic_config, str):
            elastic_config = torch.load(elastic_config)
        assert isinstance(elastic_config, dict), "Invalid elastic_config"
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
        torch.save(self.elastic_config, os.path.join(dir, "elastic.pt"))

    def load_ckpt(self, dir):
        self.model.from_pretrained(dir)
        self.elastic_config = torch.load(os.path.join(dir, "elastic.pt"))
