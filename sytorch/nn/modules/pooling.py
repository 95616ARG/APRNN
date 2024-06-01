from __future__ import annotations
from typing import Tuple, Any
import torch.nn as nn
import torch.nn.functional as F

from sytorch.pervasives import *
from sytorch.solver import *
from sytorch.solver import lightning
from ..symbolic_mode import no_symbolic
from .module import *
from .module import T

# __all__ = [
#     # """ AdaptiveAvgPool2d """
#     "AdaptiveAvgPool2d",

#     # """ MaxPool2d """
#     "MaxPool2d",
#     "DecoupledMaxPool2d",
#     "SymbolicDecoupledMaxPool2d",
# ]

""" Adaptive Average Pooling. """

class AdaptiveAvgPool2d(LinearLayer, Layer2D, nn.AdaptiveAvgPool2d):

    def v(self:T, vpolytopes, pattern=None):
        if isinstance(vpolytopes, Tensor):
            with no_symbolic():
                return F.relu(vpolytopes)

        elif isinstance(vpolytopes, SymbolicLightningArray):
            N, V, *S = vpolytopes.shape
            out_vpolytopes = self.forward_symbolic(
                vpolytopes.reshape(N * V, *S),
                pattern = broadcast_at(pattern, 1, (V,)).reshape(N * V, *S),
            )
            out_vpolytopes = out_vpolytopes.reshape(N, V, *out_vpolytopes.shape[1:])
            return out_vpolytopes

        else:
            raise NotImplementedError

    def forward_symbolic(self, input, pattern=None):

        if isinstance(input, Tensor):
            with no_symbolic():
                return self(input)

        elif isinstance(input, SymbolicGurobiArray):
            return adaptive_avgpool2d(input, output_size=self.output_size)

        elif isinstance(input, SymbolicLightningArray):
            return lightning.adaptive_average_pool2d(input, output_size=self.output_size, executor=self.executor)

        else:
            raise NotImplementedError(
                f"unimplemented {type(self)} symbolic forward for {type(input)}"
            )


from . import functional

def _adaptive_average_pool2d_start_index(out_idx, out_len, in_len):
    """
    https://github.com/pytorch/pytorch/blob/master/aten/src/ATen/native/quantized/cpu/AdaptiveAveragePooling.cpp#L24
    """
    return (int)(np.floor((float)(out_idx * in_len) / out_len))

def _adaptive_average_pool2d_end_index(out_idx, out_len, in_len):
    """
    https://github.com/pytorch/pytorch/blob/master/aten/src/ATen/native/quantized/cpu/AdaptiveAveragePooling.cpp#L37
    """
    return (int)(np.ceil((float)((out_idx + 1) * in_len) / out_len))

def adaptive_avgpool2d(input, output_size):
    N, I =  input.shape[:2]
    O = I
    J, K = output_size

    solver = input.solver

    input_height , input_width  = input.shape[-2:]
    output_height, output_width = output_size # JK
    output = solver.reals((N, O, J, K))

    def input_window(oj, ok):
        ih0 = _adaptive_average_pool2d_start_index(oj, output_height, input_height)
        ih1 = _adaptive_average_pool2d_end_index(oj, output_height, input_height)
        iw0 = _adaptive_average_pool2d_start_index(ok, output_width, input_width)
        iw1 = _adaptive_average_pool2d_end_index(ok, output_width, input_width)
        return input[:,:,ih0:ih1,iw0:iw1]

    for j, k in np.ndindex((J, K)):
        output[:,:,j,k] = input_window(j,k).reshape(N, O, -1).mean(-1)

    return output #.alias(conservative=False)

""" Max Pooling. """

class MaxPool2d(NonLinearLayer, Layer2D, nn.MaxPool2d):
    def activation_pattern(self: T, input):
        return self.as_strided(input)\
                   .argmax(dim=-3, keepdim=True)\
                   .cpu().detach().numpy()

    def as_strided(self, input):
        if isinstance(input, Tensor):
            return _as_strided_window_views2d_torch(
                input,
                kernel_size  = self.kernel_size,
                padding      = self.padding,
                padding_mode = 'constant',
                stride       = self.stride,
                dilation     = self.dilation,
                ceil_mode    = self.ceil_mode,
                value        = -1e100,
            ).flatten(start_dim=-4, end_dim=-3)
        else:
            return flatten(
                _as_strided_window_views2d_numpy(
                    input,
                    kernel_size  = self.kernel_size,
                    padding      = self.padding,
                    padding_mode = 'constant',
                    stride       = self.stride,
                    dilation     = self.dilation,
                    ceil_mode    = self.ceil_mode,
                    value        = -1e100,
                ), start_dim=-4, end_dim=-3
            ).view(type(input)).to(input.solver)

    def forward_symbolic(self, input, pattern):

        if isinstance(input, Tensor):
            with no_symbolic():
                return self(input)

        elif isinstance(input, SymbolicLightningArray):
            assert not self.return_indices

            window_views = self.as_strided(input)

            output = np.take_along_axis(
                arr     = window_views,
                indices = pattern,
                axis    = -3
            ).squeeze(-3).view(type(input)).to(input.solver)

            constrs = window_views.argmax(axis=-3, executor=self.executor) == pattern
            handle_constrs(constrs)
            return output

        # elif isinstance(input, SymbolicGurobiArray): ...
        else:
            raise NotImplementedError(
                f"unimplemented {type(self)} symbolic forward for {type(input)}."
            )

    def decouple(self) -> SymbolicDecoupled:
        if isinstance(self, Decoupled):
            assert isinstance(self, SymbolicDecoupledMaxPool2d)
            return self
        return SymbolicDecoupledMaxPool2d(
            val = self.copy(),
            act = self.deepcopy().requires_symbolic_(False)
        )

class DecoupledMaxPool2d(DecoupledNonLinearLayer):
    def from_indices(self, inputs, indices):
        inputs = torch.flatten(inputs, 2)
        output = torch.gather(inputs, 2, torch.flatten(indices, 2))
        output = output.view(indices.shape)

        return output

    def forward_decoupled(self, input_val: Tuple, input_act: Tuple) -> Tuple[Any, Any]:

        output_act, indices = F.max_pool2d(
            input_act,
            kernel_size = self.act.kernel_size,
            stride      = self.act.stride,
            padding     = self.act.padding,
            dilation    = self.act.dilation,
            ceil_mode   = self.act.ceil_mode,
            return_indices = True
        )

        output_val = self.from_indices(input_val, indices)

        return (output_val, output_act)

class SymbolicDecoupledMaxPool2d(DecoupledMaxPool2d, NonLinearLayer): ...
