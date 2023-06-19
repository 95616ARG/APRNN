from __future__ import annotations
from typing import Literal, overload

from sytorch.pervasives import *
from sytorch.solver.base import *
from sytorch.solver.symbolic_array import SymbolicArray

def real(**kwargs) -> SymbolicArray:
    return Solver.default().real(**kwargs)

def reals(shape, **kwargs) -> SymbolicArray:
    return Solver.default().reals(shape=shape, **kwargs)

def reals_like(ref, **kwargs) -> SymbolicArray:
    return Solver.default().reals_like(ref=ref, **kwargs)

def add_constraints(*constraints: SymbolicArray) -> None:
    return Solver.default().add_constraints(*constraints)

def add_objectives(*objectives: SymbolicArray, mode: Literal['minimize', 'maximize']) -> None:
    return Solver.default().add_objectives(*objectives, mode=mode)

def minimize(*objectives) -> None:
    return Solver.default().add_objectives(*objectives, mode='minimize')

def maximize(*objectives) -> None:
    return Solver.default().add_objectives(*objectives, mode='maximize')

def evaluate(variables: SymbolicArray) -> 'torch.Tensor':
    return Solver.default().evaluate(variables)

def solve(*constraints: SymbolicArray, minimize=None, maximize=None) -> bool:
    return Solver.default().solve(*constraints, minimize=minimize, maximize=maximize)
