from __future__ import annotations

import torch.nn as nn

from sytorch.solver import SymbolicGurobiArray, SymbolicLightningArray

from .module import *
from .module import T

from sytorch.pervasives import *
from sytorch.solver import lightning
from ...solver import *
from ..symbolic_mode import *

__all__ = [
    "Dropout",
    "DecoupledDropout",
    "SymbolicDecoupledDropout",
]

class Dropout(LinearLayer, nn.Dropout):
    def decouple(self) -> SymbolicDecoupled:
        if isinstance(self, Decoupled):
            assert isinstance(self, SymbolicDecoupledDropout)
            return self
        return SymbolicDecoupledDropout(
            val = self.copy(),
            act = self.deepcopy().requires_symbolic_(False)
        )

    def forward_symbolic(self, input: 'Tensor' | 'SymbolicArray', pattern=None):
        if self.training:
            raise NotImplementedError(
                f"unimplemented symbolic forward for training dropout."
            )
        return input

class DecoupledDropout(DecoupledLayer):
    def forward_decoupled(self, input_val: Tensor, input_act: Tensor) -> Tuple[Tensor, Tensor]:
        if self.training:
            raise NotImplementedError(
                "unimplemented decoupled forward for dropout during training."
            )
        return (
            self.val(input_val),
            self.act(input_act)
        )

class SymbolicDecoupledDropout(DecoupledDropout, Layer): ...
