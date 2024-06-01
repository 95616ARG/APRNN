from __future__ import annotations
from torch import Tensor
import torch.nn as nn
from .module import LinearLayer

__all__ = [
    "Flatten",
]

class _Flatten(nn.Module):
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

class Flatten(LinearLayer, _Flatten):
    """ Symbolic sytorch.nn.Module implementation. """
    ...
