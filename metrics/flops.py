import numpy as np
from torch import nn
import torch

from . import nonzero
from .abstract_flops import dense_flops, conv2d_flops, attention_flops
from ..pruning.utils import get_activations
from ..pruning import Conv2dMasked, LinearMasked


#  TODO layernorm and attention (Also Masked version of these)
def _conv2d_flops(module, activation):
    # Auxiliary func to use abstract flop computation

    # Drop batch & channels. Channels can be dropped since
    # unlike shape they have to match to in_channels
    input_shape = activation.shape[2:]
    # TODO Add support for dilation and padding size
    return conv2d_flops(in_channels=module.in_channels,
                        out_channels=module.out_channels,
                        input_shape=input_shape,
                        kernel_shape=module.kernel_size,
                        padding=module.padding_mode,
                        strides=module.stride,
                        dilation=module.dilation)


def _layer_norm_flops(module, activation):
    # Calculate flops for layer norm TODO not sure!
    return torch.prod(torch.tensor(module.normalized_shape)) * 3


def _attention_flops(module, activation):
    return attention_flops(module.embed_dim, module.num_heads)


def _linear_flops(module, activation):
    # Auxiliary func to use abstract flop computation
    return dense_flops(module.in_features, module.out_features)


def flops(model, input):
    """Compute Multiply-add FLOPs estimate from model

    Arguments:
        model {torch.nn.Module} -- Module to compute flops for
        input {torch.Tensor} -- Input tensor needed for activations

    Returns:
        tuple:
        - int - Number of total FLOPs
        - int - Number of FLOPs related to nonzero parameters
    """
    FLOP_fn = {
        nn.Conv2d: _conv2d_flops,
        nn.Linear: _linear_flops,
        Conv2dMasked: _conv2d_flops,
        LinearMasked: _linear_flops,
        nn.LayerNorm: _layer_norm_flops,
        nn.MultiheadAttention: _attention_flops,
        # TODO masked version of these
    }

    total_flops = nonzero_flops = 0
    activations = get_activations(model, input)

    # The ones we need for backprop
    for m, (act, _) in activations.items():
        if m.__class__ in FLOP_fn:
            module_flops = FLOP_fn[m.__class__](m, act)
            total_flops += module_flops
            # For our operations, all weights are symmetric so we can just
            # do simple rule of three for the estimation
            try:
                w = m.weight.detach().cpu().numpy().copy()
                nonzero_flops += module_flops * nonzero(w).sum() / np.prod(w.shape)
            except AttributeError:
                nonzero_flops = module_flops

    return total_flops, nonzero_flops
