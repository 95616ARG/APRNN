from __future__ import annotations

import torch.nn as nn
import torch.nn.functional as F
from sytorch.solver import SymbolicGurobiArray, SymbolicLightningArray

from .module import *
from .module import T

from sytorch.pervasives import *
from sytorch.solver import lightning
from ...solver import *
from ..symbolic_mode import *
from .functional import *

__all__ = [
    'NormalizeInput',
    "Identity",
    "Linear",
]

class Identity(LinearLayer, nn.Identity):
    def forward_symbolic(self, input: 'Tensor' | 'SymbolicArray', pattern=None):
        return input

class NormalizeInput(LinearLayer):
    def __init__(self, mean, std):
        super().__init__()
        self.mean = nn.Parameter(mean)
        self.std = nn.Parameter(std)

    def forward(self, input):
        return (input - self.mean) / self.std

    def forward_symbolic(self, input: 'Tensor' | 'SymbolicArray', pattern=None):
        if isinstance(input, Tensor):
            return (input - self.mean) / self.std
        else:
            raise NotImplementedError

    def v(self, vpolytopes, pattern=None):
        return self(vpolytopes)

class EqualityEncoder:
    def __init__(self, encoder):
        self.encoder = encoder

    def __eq__(self, other):
        return self.encoder(other)

class Linear(LinearLayer, nn.Linear, ONNXCompatibleModule):

    def forward_symbolic(self:T, input: Tensor | SymbolicArray, pattern=None):

        if self.bias is not None:
            weight = self.weight.array_if_symbolic()
            bias = self.bias.array_if_symbolic()
            configuration = tuple(map(type, (input, weight, bias)))

        else:
            weight = self.weight.array_if_symbolic()
            bias = None
            configuration = tuple(map(type, (input, weight)))

        if all(issubclass(ty, Tensor) for ty in configuration):
            """ Concrete execution. """
            with no_symbolic():
                output = self(input)

        elif any(issubclass(ty, SymbolicGurobiArray) for ty in configuration):
            """ Symbolic gurobi execution. """
            return linear(input, weight, bias)
            # if bias is not None:
            #     output = (input @ weight.T + bias[None, :]).alias()
            # else:
            #     output = (input @ weight.T).alias()

        elif any(issubclass(ty, SymbolicLightningArray) for ty in configuration):
            """ Symbolic lightning execution. """
            if self.row_mask is not None and isinstance(input, Tensor):
                # print("linear row mask")
                output = np.empty((*input.shape[:-1], weight.shape[0]), dtype=object)
                if bias is not None:
                    sym_output = lightning.linear(input, weight[self.row_mask,:], bias[self.row_mask])
                    solver = sym_output.solver
                    array_type = type(sym_output)
                    output[..., self.row_mask] = sym_output
                    with no_symbolic():
                        output[..., ~self.row_mask] = F.linear(
                            input, self.weight[~self.row_mask,:], self.bias[~self.row_mask]
                        ).cpu().detach().numpy()

                else:
                    sym_output = lightning.linear(input, weight[self.row_mask,:], bias=None)
                    solver = sym_output.solver
                    array_type = type(sym_output)
                    output[..., self.row_mask] = sym_output
                    with no_symbolic():
                        output[..., ~self.row_mask] = F.linear(
                            input, self.weight[~self.row_mask,:], bias=None
                        ).cpu().detach().numpy()

                output = output.view(array_type).to(solver)
                output.row_mask = self.row_mask
                output.mask = broadcast_at(self.row_mask, 0, output.shape[:-1])
                output._concrete_dtype = torch_dtype_to_numpy(input.dtype)
                assert output.shape == output.mask.shape

            else:
                output = lightning.linear(input, weight, bias)

        else:
            raise NotImplementedError(
                f"unimplemented Linear symbolic forward for {configuration}"
            )

        return output
