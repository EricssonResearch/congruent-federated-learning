from torch.nn.modules.module import Module
from torch import Tensor
from torch.nn import functional as tf
from torch.overrides import handle_torch_function, has_torch_function
from torch.nn.parameter import Parameter
import torch
from torch.nn import init
import math


EPSILON = 1e-18


def prob_congruent_mask(in_data, mask, mask_std, weight, bias=None):
    # type: (Tensor, Tensor, Tensor, Tensor, {None, Tensor}) -> Tensor
    tens_ops = (in_data, weight)
    if not torch.jit.is_scripting():
        if any([type(t) is not Tensor for t in tens_ops]) and has_torch_function(tens_ops):
            return handle_torch_function(prob_congruent_mask, tens_ops, (in_data, mask, mask_std, weight), bias=bias)

    noise = torch.normal(0, 1, size=mask_std.shape)
    mask = mask + (mask_std * noise)
    out = torch.relu(weight * mask) + EPSILON
    updated_weight = torch.mul(torch.sign(mask), torch.sqrt(out))

    if in_data.dim() == 2 and bias is not None:
        ret = torch.addmm(bias, in_data, updated_weight.t())
    else:
        output = in_data.matmul(weight.t())
        if bias is not None:
            output += bias
        ret = output
    return ret


class ProbCongruentMask(Module):
    __constants__ = ['in_features', 'out_features']
    in_features: int
    out_features: int

    def __init__(self, in_features: int, out_features: int,
                 bias: bool = True) -> None:
        super(ProbCongruentMask, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.Tensor(out_features, in_features))
        if bias:
            self.bias = Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound)

    def forward(self, in_data) -> Tensor:
        if in_data[1] is None and in_data[2] is None:
            return tf.linear(in_data[0], self.weight, self.bias)
        else:
            return prob_congruent_mask(in_data[0], in_data[1], in_data[2], self.weight, self.bias)

    def extra_repr(self) -> str:
        return 'in_features={}, out_features={} bias={}'.format(
            self.in_features, self.out_features, self.bias is not None
        )
