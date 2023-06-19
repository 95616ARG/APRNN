from __future__ import annotations

import torch.nn as nn
import torch.nn.functional as F
from sytorch.solver import SymbolicGurobiArray, SymbolicLightningArray

from sytorch.pervasives import *
from sytorch.solver import lightning
from ...solver import *
from ..symbolic_mode import *

from tqdm.auto import tqdm

class stype_: ...
class ty_symbolic_(stype_): ...
class ty_concrete_(stype_): ...

def stype_of(arr: Optional['Tensor' | 'SymbolicGurobiArray']) -> stype_:
    if isinstance(arr, Tensor):
        return ty_concrete_

    if isinstance(arr, np.ndarray) and arr.dtype != object:
        return ty_concrete_

    elif isinstance(arr, SymbolicGurobiArray):
        return ty_symbolic_

    elif arr == None:
        return None

    else:
        raise RuntimeError(
            f"unexpected array of type {type(arr)}."
        )

def linear_scc(inp: SymbolicGurobiArray, weight: Tensor, bias: Tensor) -> SymbolicGurobiArray:
    N, I = inp.shape
    O, I = weight.shape
    solver = inp.solver
    out = solver.reals((N, O))
    rows = O
    cols = I + O
    A = np.zeros((rows, cols), dtype=np.float64)
    A[:, :I] = weight.detach().cpu().numpy()
    np.fill_diagonal(A[:, I:], -1.)
    X = np.empty((cols,), dtype=object).view(type(inp)).to(solver)
    b = -bias.detach().cpu().numpy()

    # TODO(optimize)
    try:
        for n in tqdm(range(N), desc='encoding', leave=False):
            X[:I] = inp[n]
            X[-O:] = out[n]
            solver.solver.addMConstr(A, X.mvar(), '=', b)

    except AttributeError as e:
        raise RuntimeError(
            f"symbolic array contains concrete values: {X}"
        )

    return out

def linear_scn(inp: SymbolicGurobiArray, weight: Tensor, bias: None=None) -> SymbolicGurobiArray:
    N, I = inp.shape
    O, I = weight.shape
    solver = inp.solver
    out = solver.reals((N, O))
    rows = O
    cols = I + O
    A = np.zeros((rows, cols), dtype=np.float64)
    A[:, :I] = weight.detach().cpu().numpy()
    np.fill_diagonal(A[:, I:], -1.)
    X = np.empty((cols,), dtype=object).view(type(inp)).to(solver)
    b = np.zeros((rows,), dtype=np.float64)
    for n in tqdm(range(N), desc='encoding', leave=False):
        X[:I] = inp[n]
        X[-O:] = out[n]
        solver.solver.addMConstr(A, X.mvar(), '=', b)

    return out

def linear_scs(inp: SymbolicGurobiArray, weight: Tensor, bias: SymbolicGurobiArray) -> SymbolicGurobiArray:

    N, I = inp.shape
    O, I = weight.shape
    solver = inp.solver
    out = solver.reals((N, O))
    rows = O
    cols = I + O + O # todo
    A = np.zeros((rows, cols), dtype=np.float64)
    # cw = weight.detach().cpu().numpy()
    A[:, :I] = weight.detach().cpu().numpy()
    np.fill_diagonal(A[:, I  :],  1.)
    np.fill_diagonal(A[:, I+O:], -1.)
    X = np.empty((cols,), dtype=object).view(type(inp)).to(solver).to(solver)
    X[I:I+O] = bias
    b = np.zeros((rows,), dtype=np.float64)
    # dummy_var = out.item(0)

    try:
        for n in tqdm(range(N), desc='encoding', leave=False):
            X[:I] = inp[n]
            X[-O:] = out[n]

            solver.solver.addMConstr(A, X.mvar(), '=', b)

            # # A workaround.
            # zero_indices = [i for i in range(I) if isinstance(X[i], float) and X[i] == 0.]
            # X[zero_indices] = dummy_var
            # A[:,zero_indices] = 0.
            # solver.solver.addMConstr(A, X.mvar(), '=', b)
            # A[:,zero_indices] = cw[:,zero_indices]

    except AttributeError as e:
        raise RuntimeError(
            f"symbolic array contains concrete values: {X}"
        )

    return out

def linear_csn(inp: Tensor, weight: SymbolicGurobiArray, bias: None=None) -> SymbolicGurobiArray:
    N, I = inp.shape
    O, I = weight.shape
    solver = weight.solver
    out = solver.reals((N, O))

    # (N, I) * (O, I) + (O) = (N, O)
    # (N, I) * (I) + () = (N,)

    rows = N
    cols = I + N
    A = np.zeros((rows, cols), dtype=np.float64)
    A[:,:I] = inp.detach().cpu().numpy()
    np.fill_diagonal(A[:,-N:], -1.)
    X = np.empty((cols,), dtype=object).view(type(weight)).to(solver)
    b = np.zeros((rows,), dtype=np.float64)

    for o in tqdm(range(O), desc='encoding', leave=False):
        X[:I] = weight[o,:]
        X[-N:] = out[:,o]
        solver.solver.addMConstr(A, X.mvar(), '=', b)

    return out

def linear_csc(inp: Tensor, weight: SymbolicGurobiArray, bias: Tensor) -> SymbolicGurobiArray:
    N, I = inp.shape
    O, I = weight.shape
    solver = weight.solver
    out = solver.reals((N, O))

    # (N, I) * (O, I) + (O) = (N, O)
    # (N, I) * (I) + () = (N,)

    rows = N
    cols = I + N
    A = np.zeros((rows, cols), dtype=np.float64)
    A[:,:I] = inp.detach().cpu().numpy()
    np.fill_diagonal(A[:,-N:], -1.)
    X = np.empty((cols,), dtype=object).view(type(weight)).to(solver)
    b = np.zeros((rows,), dtype=np.float64) # todo

    for o in tqdm(range(O), desc='encoding', leave=False):
        X[:I] = weight[o,:]
        X[-N:] = out[:,o]
        b[:] = -bias[o]
        solver.solver.addMConstr(A, X.mvar(), '=', b)

    return out

def decouple_output(inp: Tensor, weight: SymbolicGurobiArray, bias: Tensor) -> SymbolicGurobiArray:
    N, I = inp.shape
    O, I = weight.shape
    solver = weight.solver
    out = solver.reals((N, O))

    # (N, I) * (O, I) + (O) = (N, O)
    # (N, I) * (I) + () = (N,)

    rows = N
    cols = I + N
    A = np.zeros((rows, cols), dtype=np.float64)
    A[:,:I] = inp.detach().cpu().numpy()
    np.fill_diagonal(A[:,-N:], -1.)
    X = np.empty((cols,), dtype=object).view(type(weight)).to(solver)
    b = np.zeros((rows,), dtype=np.float64) # todo

    for o in tqdm(range(O), desc='encoding', leave=False):
        X[:I] = weight[o,:]
        X[-N:] = out[:,o]
        b[:] = -bias[o]
        solver.solver.addMConstr(A, X.mvar(), '=', b)

    return out

def linear_css(inp: Tensor, weight: SymbolicGurobiArray, bias: SymbolicGurobiArray) -> SymbolicGurobiArray:
    N, I = inp.shape
    O, I = weight.shape
    solver = weight.solver
    out = solver.reals((N, O))

    # (N, I) * (O, I) + (O) = (N, O)
    # (N, I) * (I) + () = (N,)

    rows = N
    cols = I + N + 1
    A = np.zeros((rows, cols), dtype=np.float64)
    A[:,:I] = inp.detach().cpu().numpy()
    A[:, I] = 1.
    np.fill_diagonal(A[:,-N:], -1.)
    X = np.empty((cols,), dtype=object).view(type(weight)).to(solver)
    b = np.zeros((rows,), dtype=np.float64) # todo

    for o in tqdm(range(O), desc='encoding', leave=False):
        X[:I] = weight[o,:]
        X[I] = bias[o]
        X[-N:] = out[:,o]
        solver.solver.addMConstr(A, X.mvar(), '=', b)

    return out

def linear_ccs(inp: Tensor, weight: Tensor, bias: SymbolicGurobiArray) -> SymbolicGurobiArray:
    raise NotImplementedError()

import torch.nn.functional as F

linear_kernels = {
    ty_symbolic_: {
        ty_concrete_: {
            ty_concrete_: linear_scc,
            ty_symbolic_: linear_scs,
            None        : linear_scn,
        }
    },
    ty_concrete_: {
        ty_symbolic_: {
            ty_concrete_: linear_csc,
            ty_symbolic_: linear_css,
            None        : linear_csn,
        },
        ty_concrete_: {
            ty_concrete_: F.linear,
            ty_symbolic_: linear_ccs,
            None        : F.linear,
        }
    },
}


def linear(inp, weight, bias=None) -> SymbolicGurobiArray:

    try:
        kernel = linear_kernels[stype_of(inp)][stype_of(weight)][stype_of(bias)]
    except KeyError as e:
        raise RuntimeError(
            "unexpected linear configuration {}".format(
                (stype_of(inp), stype_of(weight), stype_of(bias))
            )
        )

    return kernel(
        inp, weight, bias
    )

