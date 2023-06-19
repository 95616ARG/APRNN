#%%

from __future__ import annotations
from typing import Any, Iterable, Literal, List, Tuple

import numpy as np

from .base import _ShapeT1, _infer_shape
import gurobipy
from gurobipy import GRB

def vars(
    model: gurobipy.Model,
    shape: _ShapeT1,
    vtype: Literal["C", "B", "I", "S", "N"],
    **kwargs
) -> gurobipy.MVar:
    if isinstance(shape, Iterable):
        shape = tuple(shape)
    out = np.array(
        model.addMVar(shape=shape, vtype=vtype, **kwargs).tolist(),
        dtype=object
    )
    model.update()
    return out

def reals(model, shape, lb=-GRB.INFINITY, ub=GRB.INFINITY, **kwargs) -> gurobipy.MVar:
    if lb is None:
        lb = -GRB.INFINITY
    if ub is None:
        ub = GRB.INFINITY
    return vars(model=model, shape=shape, vtype="C", lb=lb, ub=ub, **kwargs)

def bools(model, shape, **kwargs) -> gurobipy.MVar:
    return vars(model=model, shape=shape, vtype="B", **kwargs)

def ints(model, shape, **kwargs) -> gurobipy.MVar:
    return vars(model=model, shape=shape, vtype="I", **kwargs)

def vars_like(
    model: gurobipy.Model,
    ref,
    vtype: Literal["C", "B", "I", "S", "N"],
    **kwargs
) -> gurobipy.MVar:

    if isinstance(ref, (Tuple, List)):
        if isinstance(ref, Tuple) and all(isinstance(i, int) for i in ref):
            return vars(model=model, shape=ref, vtype=vtype, **kwargs)
        return type(ref)(vars_like(model=model, ref=ref, vtype=vtype, **kwargs) for ref in ref)

    shape = _infer_shape(ref)
    return vars(model=model, shape=shape, vtype=vtype, **kwargs)

def reals_like(model, ref, lb=-GRB.INFINITY, ub=GRB.INFINITY, **kwargs) -> gurobipy.MVar:
    if lb is None:
        lb = -GRB.INFINITY
    if ub is None:
        ub = GRB.INFINITY
    return vars_like(model=model, ref=ref, vtype="C", lb=lb, ub=ub, **kwargs)

def bools_like(model, ref, **kwargs) -> gurobipy.MVar:
    return vars_like(model=model, ref=ref, vtype="B", **kwargs)

def ints_like(model, ref, **kwargs) -> gurobipy.MVar:
    return vars_like(model=model, ref=ref, vtype="I", **kwargs)

evaluate = np.vectorize(
    lambda x: \
        x.getValue() if hasattr(x, 'getValue') else
        x.X          if hasattr(x, 'X')        else
        x
    )
