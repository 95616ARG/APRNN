from __future__ import annotations
from typing import Any, Iterable, List, Tuple, overload

import numpy as np
from numpy import ndarray

import torch
from torch import Tensor

from .base import _ShapeT1, _new_variable_uid, _infer_shape
import z3

def reals(
    shape: _ShapeT1,
    naming: str="__{}",
    unique: bool=False
) -> ndarray[_ShapeT1, z3.ArithRef]:
    if unique:
        naming += f"^uid_{_new_variable_uid()}"

    size = np.prod(shape)
    ndim = len(shape)
    indices = np.indices(shape).reshape(ndim, -1)

    return np.array([
        z3.Real(naming.format(indices[:, i]))
        for i in range(size)
    ], dtype=object).reshape(shape)

def reals_like(
    ref: ndarray[_ShapeT1, Any] | Tensor | Iterable[Tensor] | _ShapeT1 | int,
    naming: str="__{}",
    unique=False
) -> np.ndarray[_ShapeT1, z3.ArithRef]:

    shapes = _infer_shape(ref)

    if isinstance(shapes, Tuple):
        return reals(shapes, naming=naming, unique=unique)

    elif isinstance(shapes, List):
        return [reals(shape, naming=naming, unique=unique) for shape in shapes]

def from_model(
    model: z3.ModelRef,
    x: np.ndarray[z3.ArithRef],
    default=0.
) -> torch.Tensor[torch.float32]:
    """
    TODOs
    =====
    1. Use torch.vmap, which is faster.
    """
    def _coerce(v):
        if isinstance(v, z3.RatNumRef):
            return float(v.as_fraction())
        elif isinstance(v, z3.BoolRef):
            return bool(v)
        else:
            raise NotImplementedError(f"TODO: Coerce z3 expr {v} to python value.")

    y = np.vectorize(lambda v: model.eval(v))(x)
    y[y!=None] = np.vectorize(_coerce)(y[y!=None])
    y[y==None] = default
    if isinstance(y.flatten()[0], float):
        y = y.astype(np.float32)
    elif isinstance(y.flatten()[0], bool):
        y = y.astype(bool)
    return torch.from_numpy(y)

def _indexing_at_dim_gen(dim, ndim):
    def indexing_at_dim(sl_dim):
        sl = [slice(None)] * ndim
        sl[dim] = sl_dim
        return tuple(sl)
    return indexing_at_dim

def gradient_wrt_dim(F: ndarray, h: int, dim: int, edge_order :int=1, drop_edge=False):
    assert(edge_order == 1)
    idx = _indexing_at_dim_gen(dim, F.ndim)

    grad = np.empty_like(F)
    grad[idx(          0)] = F[idx(1)] - F[idx(0)]
    grad[idx(slice(1,-1))] =(F[idx(slice(2,None))] - F[idx(slice(None,-2))]) / 2
    grad[idx(         -1)] = F[idx(-1)] - F[idx(-2)]
    grad /= h
    if drop_edge:
        grad = grad[idx(slice(edge_order, -edge_order))]

    return grad

def gradient(F: ndarray, spacing: ndarray, edge_order=1, drop_edge=False):
    ndims = len(spacing)
    return np.stack([
        gradient_wrt_dim(F, h, dim, edge_order=edge_order, drop_edge=drop_edge)
        for h, dim in zip(spacing, range(-ndims, 0))
    ])

def divergence(F: ndarray, spacing: ndarray):
    ndims = len(spacing)
    idx = _indexing_at_dim_gen(-ndims-1, F.ndim)
    return np.ufunc.reduce(
        np.add, [
            gradient_wrt_dim(F[idx(i)], h, dim=-1-i)
            for i, h in enumerate(spacing)
        ]
    )

def laplace(F: ndarray, spacing: ndarray):
    return divergence(gradient(F, spacing), spacing)

# def to_pointer(input: np.ndarray) -> np.ndarray:
#     return np.vectorize(lambda x: x.ast.value)(input)

# def to_ArithRef(input: np.ndarray) -> np.ndarray:
#     return np.vectorize(lambda x: z3.ArithRef(int(x)))(input)

# z3ext = ctypes.CDLL("z3utils.so")
# def matmul(A, v):
#     vp = to_pointer(v)
#     result = np.empty(A[...,0].shape, dtype=vp.dtype)
#     z3ext.matmul(
#         ctypes.c_void_p(A.ctypes.data),
#         ctypes.c_void_p(vp.ctypes.data),
#         ctypes.c_void_p(result.ctypes.data),
#         A[...,0].size,
#         v.size,
#         v[0].ctx.ctx.value
#     )
#     return to_ArithRef(result)
