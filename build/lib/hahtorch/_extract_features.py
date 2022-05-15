from typing import Dict, Iterable, Callable
from collections import OrderedDict
import numpy as np

import torch
from torch import nn, Tensor
from torch.nn.functional import pad


from collections import OrderedDict

__all__ = ["ActivationExtractor"]


class ActivationExtractor(nn.Module):
    """
    Wrapper extracting a specific layer type outputs and inputs from a DNN
    Args:
        model: DNN model
        layer_type: Layer type of the interested outputs and inputs
    """

    def __init__(self, model: nn.Module, layer_type):
        super().__init__()
        self._model = model
        self.layer_type = layer_type

        self.hook_handles = {}

        self.layer_outputs = OrderedDict()
        self.layer_inputs = OrderedDict()
        self.layers_of_interest = OrderedDict()
        for layer_id, layer in model.named_modules():
            if isinstance(layer, layer_type):
                self.layer_outputs[layer_id] = torch.empty(0)
                self.layer_inputs[layer_id] = torch.empty(0)
                self.layers_of_interest[layer_id] = layer

        for layer_id, layer in model.named_modules():
            if isinstance(layer, layer_type):
                self.hook_handles[layer_id] = layer.register_forward_hook(
                    self.generate_hook_fn(layer_id))

    def generate_hook_fn(self, layer_id: str) -> Callable:
        def fn(_, input, output):
            self.layer_outputs[layer_id] = output
            self.layer_inputs[layer_id] = input[0]
        return fn

    def close(self):
        [hook_handle.remove() for hook_handle in self.hook_handles.values()]

    def forward(self, x):
        return self._model(x)

    def __getattribute__(self, name: str):
        # the last three are used in nn.Module.__setattr__
        if name in ["_model", "layers_of_interest", "layer_outputs", "layer_inputs", "hook_handles", "generate_hook_fn", "close", "__dict__", "_parameters", "_buffers", "_non_persistent_buffers_set"]:
            return object.__getattribute__(self, name)
        else:
            return getattr(self._model, name)
