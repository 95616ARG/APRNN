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
#     "Flatten",
# ]

class _Flatten(torch.nn.Module):
    """ torch.nn.Module implementation. """
    def __init__(self, start_dim=0, end_dim=-1):
        super().__init__()
        self.start_dim = start_dim
        self.end_dim   = end_dim

    def forward(self, x: Tensor) -> Tensor:
        return x.flatten(
            start_dim = self.start_dim,
            end_dim   = self.end_dim
        )

    def __repr__(self):
        return f"Flatten(start_dim={self.start_dim}, end_dim={self.end_dim})"

class _ONNXFlatten(torch.nn.Module, ONNXCompatibleModule):
    def forward(self, x):
        return x.flatten(1, -1)

class Flatten(LinearLayer, _Flatten):
    def to_onnx_compatible(self):
        assert self.start_dim == 1 and self.end_dim == -1
        return _ONNXFlatten()

    """ Symbolic sytorch.nn.Module implementation. """
    def forward_symbolic(self, x, *args, **kwargs):
        return flatten(x, start_dim=self.start_dim, end_dim=self.end_dim)

    def v(self, vpolytopes, **kwargs):
        start_dim, end_dim = self.start_dim, self.end_dim
        if start_dim >= 0: start_dim += 1
        if end_dim >= 0: end_dim += 1
        return flatten(vpolytopes, start_dim=start_dim, end_dim=end_dim)
