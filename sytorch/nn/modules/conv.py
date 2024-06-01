from __future__ import annotations
from typing import Iterable
import warnings
import torch.nn as nn
import torch.nn.functional as F

from sytorch.solver import *
from sytorch.solver import lightning
from ..symbolic_mode import no_symbolic

from sytorch.pervasives import *
import numpy as np
from .module import *
from .module import T

__all__ = [
    "Conv2d"
]

class Conv2d(LinearLayer, Layer2D, nn.Conv2d, ONNXCompatibleModule):

    def forward_symbolic(self:T, input: 'Tensor' | 'SymbolicArray', pattern=None):

        if self.bias is not None:
            weight = self.weight.array_if_symbolic()
            bias = self.bias.array_if_symbolic()
            configuration = tuple(map(type, (input, weight, bias)))

        else:
            weight = self.weight.array_if_symbolic()
            bias = None
            configuration = tuple(map(type, (input, weight)))

        if all(issubclass(ty, Tensor) for ty in configuration):
            with no_symbolic():
                output = self(input)

        elif any(issubclass(ty, SymbolicGurobiArray) for ty in configuration):

            if self.row_mask is not None:
                sym_output = conv2d_symbolic(
                    input,
                    weight = weight[self.row_mask,...],
                    bias = bias[self.row_mask,...] if bias is not None else None,
                    padding      = self.padding,
                    padding_mode = self.padding_mode,
                    stride       = self.stride,
                    dilation     = self.dilation,
                    ceil_mode    = False,
                    value        = 'zero'
                )
                solver = sym_output.solver
                array_type = type(sym_output)
                output = np.empty((*input.shape[:-3], weight.shape[0], *sym_output.shape[-2:]), dtype=object)
                output[...,self.row_mask,:,:] = sym_output

                with no_symbolic():
                    output[..., ~self.row_mask,:,:] = self._conv_forward(
                        input,
                        self.weight[~self.row_mask,...],
                        self.bias[~self.row_mask,...] if bias is not None else None,
                    ).cpu().detach().numpy()

                output = output.view(array_type).to(solver)
                output.row_mask = self.row_mask
                output.mask = broadcast_at(self.row_mask, output.ndim-1, output.shape[-2:], 0, output.shape[:-3])
                output._concrete_dtype = torch_dtype_to_numpy(input.dtype)
                assert output.shape == output.mask.shape

            else:
                output = conv2d_symbolic(
                    input,
                    weight = weight,
                    bias = bias,
                    padding      = self.padding,
                    padding_mode = self.padding_mode,
                    stride       = self.stride,
                    dilation     = self.dilation,
                    ceil_mode    = False,
                    groups       = self.groups,
                    value        = 'zero'
                )

        elif any(issubclass(ty, SymbolicLightningArray) for ty in configuration):
            if self.row_mask is not None and isinstance(input, Tensor):
                # print("conv row mask")

                if bias is not None:
                    sym_output = lightning.conv2d(
                        input, weight[self.row_mask,...], bias[self.row_mask,...],
                        kernel_size  = self.kernel_size,
                        padding      = self.padding,
                        padding_mode = self.padding_mode,
                        stride       = self.stride,
                        dilation     = self.dilation,
                        ceil_mode    = False,
                        groups       = self.groups,
                        executor     = self.executor,
                    )
                    solver = sym_output.solver
                    array_type = type(sym_output)
                    output = np.empty((*input.shape[:-3], weight.shape[0], *sym_output.shape[-2:]), dtype=object)
                    output[...,self.row_mask,:,:] = sym_output

                    with no_symbolic():
                        output[..., ~self.row_mask,:,:] = self._conv_forward(
                            input, self.weight[~self.row_mask,...], self.bias[~self.row_mask,...]
                        ).cpu().detach().numpy()

                else:
                    sym_output = lightning.conv2d(
                        input, weight[self.row_mask,...], bias=None,
                        kernel_size  = self.kernel_size,
                        padding      = self.padding,
                        padding_mode = self.padding_mode,
                        stride       = self.stride,
                        dilation     = self.dilation,
                        ceil_mode    = False,
                        groups       = self.groups,
                        executor     = self.executor,
                    )
                    solver = sym_output.solver
                    array_type = type(sym_output)
                    output = np.empty((*input.shape[:-3], weight.shape[0], *sym_output.shape[-2:]), dtype=object)
                    output[...,self.row_mask,:,:] = sym_output

                    with no_symbolic():
                        output[..., ~self.row_mask,:,:] = self._conv_forward(
                            input, self.weight[~self.row_mask,...], bias=None
                        ).cpu().detach().numpy()

                output = output.view(array_type).to(solver)
                output.row_mask = self.row_mask
                output.mask = broadcast_at(self.row_mask, output.ndim-1, output.shape[-2:], 0, output.shape[:-3])
                output._concrete_dtype = torch_dtype_to_numpy(input.dtype)
                assert output.shape == output.mask.shape

            else:
                output = lightning.conv2d(
                    input, weight, bias,
                    kernel_size  = self.kernel_size,
                    padding      = self.padding,
                    padding_mode = self.padding_mode,
                    stride       = self.stride,
                    dilation     = self.dilation,
                    ceil_mode    = False,
                    groups       = self.groups,
                    executor     = self.executor,
                )

        else:
            raise NotImplementedError(
                f"unimplemented {type(self)} symbolic forward for {configuration}"
            )

        return output
    

# from sytorch.nn.modules.linear import linear
from . import functional

def conv2d_symbolic(
    inp: Tensor | SymbolicGurobiArray,
    weight: Tensor | SymbolicGurobiArray,
    bias: Tensor | SymbolicGurobiArray | None,
    padding      = (0,0),
    padding_mode = 'constant',
    stride       = (1,1),
    dilation     = (1,1),
    ceil_mode    = False,
    groups       = 1,
    value        = 0,
) -> SymbolicGurobiArray:

    assert groups == 1
    O, I, W, H = weight.shape
    inp_views = _as_strided_window_views2d(
        inp,
        kernel_size = (W, H),
        padding = padding,
        padding_mode = padding_mode,
        stride = stride,
        dilation = dilation,
        ceil_mode = ceil_mode,
        value = value
    )

    N, I, W, H, J, K = inp_views.shape
    inp_views = inp_views.permute((0, 4, 5, 1, 2, 3))\
                         .reshape((N*J*K, I*W*H))
    weight = weight.reshape((O, I*W*H))

    out = functional.linear(inp_views, weight, bias)\
            .reshape(N, J, K, O)\
            .permute((0, 3, 1, 2))

    return out
