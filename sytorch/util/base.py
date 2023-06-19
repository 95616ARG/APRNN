from __future__ import annotations
from typing import Iterable, List, Tuple, overload, TypeVar

import numpy as np
from numpy import ndarray

import torch
from torch import Tensor

def flatten(input):
    if isinstance(input, ndarray):
        return ndarray.flatten()
    else:
        return np.concatenate([flatten(a) for a in input])

_variable_uid = 0
def _new_variable_uid():
    global _variable_uid
    _variable_uid += 1
    return _variable_uid -1

_ShapeT = Tuple[int, ...]
_ShapeT1 = TypeVar('_ShapeT1', bound=_ShapeT)
_ShapeT2 = TypeVar('_ShapeT2', bound=_ShapeT)
_ShapeT3 = TypeVar('_ShapeT3', bound=_ShapeT)

@overload
def _infer_shape(ref: ndarray | Tensor) -> _ShapeT: ...

@overload
def _infer_shape(ref: Iterable[ndarray | Tensor]) -> List[_ShapeT]: ...

@overload
def _infer_shape(ref: Iterable[int]) -> _ShapeT: ...

@overload
def _infer_shape(ref: int) -> _ShapeT: ...

def _infer_shape(ref):
    if isinstance(ref, np.ndarray):
        return ref.shape

    elif isinstance(ref, torch.Tensor):
        return tuple(ref.size())

    elif isinstance(ref, Iterable) and all(isinstance(i, int) for i in ref):
        return ref

    elif isinstance(ref, Iterable):
        return [_infer_shape(ref) for ref in ref]

    elif isinstance(ref, int):
        return (ref,)

    else:
        raise NotImplementedError(
            f"Failed not infer shape from ref={ref}({type(ref)})")
