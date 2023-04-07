"""Masked versions of common torch.nn.Modules

Implementations of most common parametric torch layers.
For vision classification networks the immense majority of the
parameters are in either Conv2d layers and Dense Layers (called
Linear in PyTorch)

Variables:
    masked_modules {dict} -- [description]
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models
from torch.nn.modules.utils import _pair


def _ensure_tensor(x):
    # Aux functions in case mask arguments are numpy arrays
    if not isinstance(x, torch.Tensor) and x is not None:
        return torch.from_numpy(x)
    return x


def _same_device(x_mask, x):
    # Aux function to ensure same device fo weight and mask
    # so _mul doesn't fail
    if x.device != x_mask.device:
        return x_mask.to(x.device)
    return x_mask


def _same_shape(x_mask, x):
    if isinstance(x, torch.Tensor):
        x = x.cpu().detach().numpy()
    return x.shape == x_mask.shape


class MaskedModule(nn.Module):
    def __init__(self, *args, **kwargs):
        super(MaskedModule, self).__init__(*args, **kwargs)


# TODO what about masking each head separately?
class AttentionMasked(MaskedModule):
    def __init__(self, attention_layer, in_proj_weight_mask, in_proj_bias_mask=None):
        super(AttentionMasked, self).__init__()
        assert isinstance(attention_layer, nn.MultiheadAttention), "Layer must be an attention layer"
        for attr in ['embed_dim', 'num_heads', 'dropout', 'add_zero_attn']:
            setattr(self, attr, getattr(attention_layer, attr))

        self.in_proj_weight = attention_layer.in_proj_weight
        self.in_proj_bias = attention_layer.in_proj_bias
        self.out_proj_weight = attention_layer.out_proj.weight
        self.out_proj_bias = attention_layer.out_proj.bias

        self.register_buffer("in_proj_weight_mask", None)
        self.register_buffer("in_proj_bias_mask", None)

        self.set_masks(in_proj_weight_mask, in_proj_bias_mask)

    def forward_pre(self):
        in_weight = self.in_proj_weight * self.in_weight_mask

        if self.in_bias_mask is not None:
            in_bias = self.in_proj_bias * self.in_bias_mask
        else:
            in_bias = self.in_proj_bias

        return in_weight, in_bias

    def set_masks(self, in_weight_mask, in_bias_mask=None):
        assert _same_shape(in_weight_mask, self.in_proj_weight)
        assert _same_shape(in_bias_mask, self.in_proj_bias)

        in_weight_mask = _ensure_tensor(in_weight_mask).to('cuda')
        self.in_proj_weight_mask = _same_device(in_weight_mask, self.in_proj_weight)
        self.in_proj_weight.data.mul_(in_weight_mask)

        if in_bias_mask is not None:
            in_bias_mask = _ensure_tensor(in_bias_mask).to('cuda')
            assert self.in_proj_bias is not None, "Provided layer must have bias for it to be masked"
            assert _same_shape(in_bias_mask, self.in_proj_bias), f"Bias Mask must match dimensions"
            self.in_proj_bias_mask = _same_device(in_bias_mask, self.in_proj_bias)
            self.in_proj_bias.data.mul_(in_bias_mask)

    def forward(
            self,
            query,
            key,
            value,
            key_padding_mask=None,
            need_weights=True,
            attn_mask=None,
            average_attn_weights=True,
            is_causal=False):
        in_weight, in_bias = self.forward_pre()
        return F.multi_head_attention_forward(query=query, key=key, value=value, embed_dim_to_check=self.embed_dim,
                                              num_heads=self.num_heads, in_proj_weight=in_weight, in_proj_bias=in_bias,
                                              bias_k=None, bias_v=None, add_zero_attn=self.add_zero_attn,
                                              dropout_p=self.dropout,
                                              out_proj_weight=self.out_proj_weight, out_proj_bias=self.out_proj_bias,
                                              training=True, key_padding_mask=key_padding_mask,
                                              # TODO not sure about training
                                              need_weights=need_weights,
                                              attn_mask=attn_mask, use_separate_proj_weight=False, q_proj_weight=None,
                                              k_proj_weight=None, v_proj_weight=None, static_k=None, static_v=None,
                                              average_attn_weights=average_attn_weights, is_causal=is_causal)

    # TODO __repr__ remained


class LinearConvMaskedModule(MaskedModule):

    def __init__(self, layer, weight_mask, bias_mask=None):
        super(LinearConvMaskedModule, self).__init__()

        self.weight = layer.weight
        self.bias = layer.bias

        # use register_buffer so model.to(device) works on fixed tensors like masks
        self.register_buffer("weight_mask", None)
        self.register_buffer("bias_mask", None)

        self.set_masks(weight_mask, bias_mask)

    def forward_pre(self):
        # Masks are pre multiplied, effectively
        # zeroing gradients to masked weights
        weight = self.weight * self.weight_mask
        if self.bias_mask is not None:
            bias = self.bias * self.bias_mask
        else:
            bias = self.bias
        return weight, bias

    def set_masks(self, weight_mask, bias_mask=None):
        assert _same_shape(weight_mask, self.weight), f"Weight Mask must match dimensions"

        # Multiply weights by masks so metrics can count nonzeros
        weight_mask = _ensure_tensor(weight_mask).to('cuda')
        self.weight_mask = _same_device(weight_mask, self.weight)
        self.weight.data.mul_(weight_mask)

        if bias_mask is not None:
            bias_mask = _ensure_tensor(bias_mask).to('cuda')
            assert self.bias is not None, "Provided layer must have bias for it to be masked"
            assert _same_shape(bias_mask, self.bias), f"Bias Mask must match dimensions"
            self.bias_mask = _same_device(bias_mask, self.bias)
            self.bias.data.mul_(bias_mask)


class LinearMasked(LinearConvMaskedModule):

    def __init__(self, linear_layer, weight_mask, bias_mask=None):
        """Masked version of a linear layer for pruning evaluation

        Constructed from an existing layer, a weight mask (and optionally
        a bias mask). By construction ensures backpropagation does not change
        masked parameters so they stay at zero.

        Arguments:
            linear_layer {torch.nn.Linear} -- Layer to mask. Not modified.
            weight_mask {numpy.ndarray} -- Mask with zero entries for weight vector

        Keyword Arguments:
            bias_mask {numpy.ndarray} -- Mask with zero entries for bias vector (default: {None})
        """
        super(LinearMasked, self).__init__(linear_layer, weight_mask, bias_mask)
        assert isinstance(linear_layer, nn.Linear), "Layer must be a linear layer"
        for attr in ['in_features', 'out_features']:
            setattr(self, attr, getattr(linear_layer, attr))

    def forward(self, input):
        weight, bias = self.forward_pre()
        return F.linear(input, weight, bias)

    def __repr__(self):
        s = f"{self.__class__.__name__}("
        s += f'in_features={self.in_features}, '
        s += f'out_features={self.out_features}, '
        s += f'bias={self.bias is not None})'
        return s


class Conv2dMasked(LinearConvMaskedModule):

    def __init__(self, conv_layer, weight_mask, bias_mask=None):
        """Masked version  of 2D convolutional layer for pruning evaluation

        Constructed from an existing layer, a weight mask (and optionally
        a bias mask). By construction ensures backpropagation does not change
        masked parameters so they stay at zero.

        [description]

        Arguments:
            linear_layer {torch.nn.Conv2d} -- Layer to mask. Not modified.
            weight_mask {numpy.ndarray} -- Mask with zero entries for weight vector

        Keyword Arguments:
            bias_mask {numpy.ndarray} -- Mask with zero entries for bias vector (default: {None})
        """
        super(Conv2dMasked, self).__init__(conv_layer, weight_mask, bias_mask)
        assert isinstance(conv_layer, nn.Conv2d), "Layer must be a Conv2d layer"
        for attr in ['in_channels', 'out_channels', 'kernel_size', 'dilation',
                     'stride', 'padding', 'padding_mode', 'groups']:
            setattr(self, attr, getattr(conv_layer, attr))

    def forward(self, input):
        weight, bias = self.forward_pre()
        if self.padding_mode == 'circular':
            expanded_padding = ((self.padding[1] + 1) // 2, self.padding[1] // 2,
                                (self.padding[0] + 1) // 2, self.padding[0] // 2)
            return F.conv2d(F.pad(input, expanded_padding, mode='circular'),
                            weight, bias, self.stride,
                            _pair(0), self.dilation, self.groups)
        return F.conv2d(input, weight, bias, self.stride,
                        self.padding, self.dilation, self.groups)

    def __repr__(self):
        s = f"{self.__class__.__name__}("
        s += ('{in_channels}, {out_channels}, kernel_size={kernel_size}'
              ', stride={stride}')
        if self.padding != (0,) * len(self.padding):
            s += ', padding={padding}'
        if self.dilation != (1,) * len(self.dilation):
            s += ', dilation={dilation}'
        if self.groups != 1:
            s += ', groups={groups}'
        if self.bias is None:
            s += ', bias=False'
        s += ')'
        return s.format(**self.__dict__)


# TODO Conv1D Conv3D ConvTranspose
# squeeze out Convs for channel pruning


masked_modules = {
    nn.Linear: LinearMasked,
    nn.modules.linear.NonDynamicallyQuantizableLinear: LinearMasked,
    nn.Conv2d: Conv2dMasked,
    nn.MultiheadAttention: AttentionMasked,
}
