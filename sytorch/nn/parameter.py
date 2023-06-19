from __future__ import annotations
from typing import Optional, Tuple, TypeVar, Union, overload
import warnings

import torch
import torch.nn as nn

from torch import Tensor

from sytorch.pervasives import Interval
from ..solver import *

__all__ = [
    'Parameter'
]

def _raise(error):
    raise error

T = TypeVar('T', bound='Parameter')
class Parameter(nn.Parameter):
    def __new__(cls, data, requires_symbolic=False, solver: Optional[Solver]=None):
        assert isinstance(solver, Solver) or solver is None
        if isinstance(data, Parameter):
            obj = super().__new__(cls, data.data, data.requires_grad)
            assert isinstance(data.solver, Solver) or data.solver is None
            solver = solver or data.solver
            requires_symbolic = requires_symbolic or data.requires_symbolic

        elif isinstance(data, nn.Parameter):
            obj = super().__new__(cls, data.data, data.requires_grad)

        elif isinstance(data, Tensor):
            obj = super().__new__(cls, data, data.requires_grad)

        else:
            raise NotImplementedError(
                f"unsupported creating symbolic.Parameter from {type(data)}"
            )

        obj._requires_symbolic = False
        obj._solver = None
        obj._lb = None
        obj._ub = None
        obj._symbolic_data = None
        obj._mask = None
        obj._concrete_cache = None
        obj._delta = None
        obj._ordered_indices = None
        obj._fused_for_argmax = dict()
        obj._name =None
        obj.to(solver=solver).requires_symbolic_(requires_symbolic)
        return obj

    def __repr__(self):
        return (
            'Symbolic Parameter containing:\n'
            f'{super(nn.Parameter, self).__repr__()},\n'
            f'requires_symbolic={self.requires_symbolic}, '
            f'solver={self.solver},'
        )

    def __deepcopy__(self, memo):
        if id(self) in memo:
            return memo[id(self)]
        else:
            result = type(self)(
                nn.Parameter(self.data.clone(memory_format=torch.preserve_format), self.requires_grad),
                requires_symbolic=self.requires_symbolic,
                solver=self.solver
            )
            result._lb = self._lb
            result._ub = self._ub
            result._mask = self._mask
            memo[id(self)] = result
            return result

    def update_(self, src=None):
        if src is None:
            src = self

        with torch.no_grad():
            if self._concrete_cache is not None:
                assert (self.cpu().detach().numpy() == self._concrete_cache).all(), \
                    "network weights is modified during repair."
            self.data[...] = src.symbolic().evaluate().to(self.data.device, self.data.dtype)
            self._concrete_cache = None
            self._delta = None

        return self

    @property
    def solver(self):
        return self._solver

    @solver.setter
    def solver(self, solver):
        if self.requires_symbolic and solver is None:
            raise NotImplementedError(
                "Please unset a parameter's requires_symbolic before reset "
                "its solver to `None`. For example: "
                "`.requires_symbolic_(False).to(solver=None)`"
            )
        self._solver = solver

    @property
    def delta(self) -> SymbolicLPArray:
        if self._delta is None:
            self._delta = self.symbolic()[self.mask] - self.concrete()[self.mask]
            # out = np.zeros(self.shape).astype(object)
            # out[self.mask] = self.symbolic()[self.mask] - self.concrete()[self.mask]
            # self._delta = out.view(type(self.symbolic())).to(self.solver)

        return self._delta

    @property
    def semi_positive_mask(self):
        return np.fromiter(map(
            (lambda v: v.is_semi_positive if isinstance(v, LightningVar) else v >= 0.),
            self.symbolic().flat
        ), dtype=bool).reshape(tuple(self.shape))

    @property
    def semi_negative_mask(self):
        return np.fromiter(map(
            (lambda v: v.is_semi_negative if isinstance(v, LightningVar) else v >= 0.),
            self.symbolic().flat
        ), dtype=bool).reshape(tuple(self.shape))

    @property
    def LBs(self):
        return np.fromiter(map(
            (lambda v: v.LB if isinstance(v, LightningVar) else v),
            self.symbolic().flat
        ), dtype=torch_dtype_to_numpy(self.dtype)).reshape(tuple(self.shape))

    @property
    def UBs(self):
        return np.fromiter(map(
            (lambda v: v.UB if isinstance(v, LightningVar) else v),
            self.symbolic().flat
        ), dtype=torch_dtype_to_numpy(self.dtype)).reshape(tuple(self.shape))

    def assert_sign(self):
        for v, c in zip(self.symbolic().flat, self.concrete().flat):
            if not isinstance(v, LightningVar):
                continue

            if c >= 0.:
                if v.LB < 0.:
                    v.UB = v.UB - v.LB
                    v.LB = 0.

            else:
                if v.UB > 0.:
                    v.UB = 0.
                    v.LB = v.LB - v.UB

        return self

    @property
    def ordered_indices(self):
        assert self._ordered_indices is not None, "please first assert_order."
        return self._ordered_indices

    @property
    def is_ordered(self):
        return self._ordered_indices is not None

    def assert_order(self):
        """ NOTE(anonymous): columns are dependent now when over-approximating.

        """
        warnings.warn("current assert order under-approx can easily cause infeasibility, consider just over-approximate for now.")
        with torch.no_grad():
            self._ordered_indices = self.sort(dim=0, stable=True, descending=False).indices
            sorted_sym = np.take_along_axis(self.symbolic(), self.ordered_indices.numpy(), axis=0)
            for smaller, bigger in zip(sorted_sym[:-1], sorted_sym[1:]):
                self.solver.add_constraints(smaller <= bigger)

    def _fuse_for_argmax(self, argmax_index, lb=None, ub=None):
        warnings.warn("lb and ub not enforced.")
        _ = self.array_if_symbolic() # sync
        if argmax_index not in self._fused_for_argmax:
            argmax_param, other_param = split_and_broadcast_for_argmax(self.array(), argmax_index, axis=0)
            out_param = (argmax_param - other_param)
            if isinstance(out_param, SymbolicArray):
                out_param = out_param.alias()
            self._fused_for_argmax[argmax_index] = out_param
        return self._fused_for_argmax[argmax_index]

    def _fuse_for_argmax_exact(self, argmax_index, lb=None, ub=None):
        warnings.warn("first attempt code.")
        if self.mask is not None:
            raise NotImplementedError(
                "I think there is a bug in masking gurobi arrays with bounds."
            )

        _ = self.symbolic() # sync
        if argmax_index not in self._fused_for_argmax:
            if self.is_ordered:
                """ First implementation where we under-approximate by sorting the params. """
                argmax_param, other_param = split_and_broadcast_for_argmax(self.symbolic(), argmax_index, axis=0)
                argmax_order, other_order = split_and_broadcast_for_argmax(self.ordered_indices.numpy(), argmax_index, axis=0)
                out_param = (argmax_param - other_param).alias()
                out_order = argmax_order - other_order
                assert not (out_order == 0).any()

                # out_order < 0: is nagative
                for prm, ord in zip(out_param.flat, out_order.flat):
                    if ord < 0:
                        make_semi_negative(prm)
                    else:
                        make_semi_positive(prm)

            else:
                self.solver.update()
                argmax_param, other_param = split_and_broadcast_for_argmax(self.symbolic(), argmax_index, axis=0)
                out_param = (argmax_param - other_param).alias()

                # Now I update it everytime.
                # # In the case we assert bounds on the delta of them, we should use tighten_bounds.
                # param_bounds = self.symbolic().bounds
                # argmax_bound, ohter_bound = split_and_broadcast_for_argmax(param_bounds, argmax_index, axis=0)
                # out_param.bounds = argmax_bound.view(Interval) - ohter_bound.view(Interval)

                self.solver.update()

            self._fused_for_argmax[argmax_index] = out_param

        out_param = self._fused_for_argmax[argmax_index]

        param_bounds = self.symbolic().bounds
        argmax_bound, ohter_bound = split_and_broadcast_for_argmax(param_bounds, argmax_index, axis=0)
        out_param.bounds = argmax_bound.view(Interval) - ohter_bound.view(Interval)
        self.solver.update()

        concrete_argmax_param, concrete_other_param = split_and_broadcast_for_argmax(self.concrete(), argmax_index, axis=0)
        concrete_out_param = (concrete_argmax_param - concrete_other_param)
        semi_positive_mask = concrete_out_param >= 0.
        negative_mask = ~semi_positive_mask

        if lb is not None or ub is not None:
            assert lb is not None and ub is not None, "for now only support setting both."

            # warnings.warn("cached fused weights once tightened can not go back.")
            # self.solver.update()

            # concrete_argmax_param, concrete_other_param = split_and_broadcast_for_argmax(self.concrete(), argmax_index, axis=0)
            # concrete_out_param = (concrete_argmax_param - concrete_other_param)

            # NOTE(anonymous): those are incomplete code. I decided to do it in another way.
            # proposed_bounds = np.zeros((*concrete_out_param.shape, 2), dtype=float)
            # semi_positive_mask = concrete_out_param >= 0.
            # negative_mask = ~semi_positive_mask
            # proposed_bounds[semi_positive_mask, 0] = 0.
            # proposed_bounds[semi_positive_mask, 1] = ub - lb # maybe better.
            # lbs = concrete_out_param + lb
            # ubs = concrete_out_param + ub

            # SHIFT
            out_param.tighten_bounds(
                (concrete_out_param + lb, concrete_out_param + ub)
            )
            self.solver.update()

        # we perhaps can even do better. we can ask gurobi to find the real tight bound to shift or check infeasibility.
        warnings.warn("shifting bounds of fused params to one side.")
        shifted_bounds = out_param.bounds
        shifted_bounds[semi_positive_mask, 1] = shifted_bounds[semi_positive_mask, 1] - shifted_bounds[semi_positive_mask, 0]
        shifted_bounds[semi_positive_mask, 0] = 0.
        shifted_bounds[negative_mask, 0] = shifted_bounds[negative_mask, 0] - shifted_bounds[negative_mask, 1]
        shifted_bounds[negative_mask, 1] = 0.
        out_param.tighten_bounds(shifted_bounds)
        self.solver.update()

        return out_param

    def fuse_for_argmax(self, indices, lb=None, ub=None):
        # indices: N, 1 or N
        indices = np.array(indices).reshape(-1)
        N, O, *trailing_shape = *indices.shape, *self.shape
        if self.requires_symbolic:
            out_param_batch = np.empty((N, O-1, *trailing_shape), dtype=object).view(type(self.symbolic())).to(self.solver)
        else:
            out_param_batch = np.empty((N, O-1, *trailing_shape), dtype=torch_dtype_to_numpy(self.dtype))

        for out_param, argmax_index in zip(out_param_batch, indices): # over N
            out_param[...] = self._fuse_for_argmax(argmax_index, lb=lb, ub=ub)

        return out_param_batch

    # def argmax_weakest_pre(self, argmax, other):
    #     coefficients = (self.symbolic()[argmax] - self.symbolic()[other]).alias()

    #     # NOTE(anonymous): No need to calculate concrete bounds. But we need some way to tell the sign.
    #     # for coeff in coefficients:
    #     #     coeff.LB, coeff.UB = coeff.overapproximate()

    #     # for coeff, a, b in zip(coefficients, self.symbolic()[argmax], self.symbolic()[other]):
    #     #     print(a.LB - b.UB, a.UB - b.LB)
    #     #     # coeff.UB = a.UB - b.LB
    #     #     # coeff.LB = a.LB - b.UB

    #     # NOTE(anonymous): I think we can just ask Gurobi to overapproximate, because
    #     # anyway one of LB/UB is dependent to others and we can not simply
    #     # compute it correctly.
    #     for coeff, argmax_order, other_order in zip(
    #         coefficients,
    #         self.ordered_indices[argmax],
    #         self.ordered_indices[other],
    #     ):
    #         if argmax_order < other_order: # argmax - other < 0
    #             coeff.make_semi_negative() # UB = 0.
    #         else:                          # argmax - other >= 0
    #             coeff.make_semi_positive() # LB = 0.

    #     coefficients.update()

    #     return coefficients

    @property
    def symbolic_data(self):
        assert self.solver is not None and self.requires_symbolic, \
            "accessing the symbolic data of a non-symbolic parameter."

        """ Create symbolic data if not exits.
        """
        if self._symbolic_data is None or self._symbolic_data.solver is not self.solver:
            self._fused_for_argmax = dict()
            self._symbolic_data = self.solver.reals(tuple(self.shape), name=self._name, mask=self.mask, lb=self.lb, ub=self.ub)

            # NOTE(anonymous): setting those bounds is super slow.
            if self.lb is not None:
                for v, c in zip(self._symbolic_data[self.mask].flat, self.concrete()[self.mask].flat):
                    v.LB += c
            if self.ub is not None:
                for v, c in zip(self._symbolic_data[self.mask].flat, self.concrete()[self.mask].flat):
                    v.UB += c

            self.solver.update()

            """ Fill non-symbolic entries with the latest concrete data. """
            if self.mask is not None:
                self._symbolic_data[~self.mask] = self.concrete()[~self._mask]

            self._delta = None

        # """ Translate to self.solver if the existing symbolic data is not on it. """
        # if self._symbolic_data.solver is not self.solver:
        #     self._symbolic_data = self._symbolic_data.to(self.solver)

        return self._symbolic_data

    def concrete(self):
        if self.device != torch.device('cpu'):
            if self._concrete_cache is None:
                self._concrete_cache = self.cpu().detach().numpy()
            if is_debugging():
                assert (self.cpu().detach().numpy() == self._concrete_cache).all()
            return self._concrete_cache
        else:
            return self.detach().numpy()

    @property
    def masked(self):
        if self.mask is not None:
            return self[torch.from_numpy(self.mask)]
        else:
            return self

    @masked.setter
    def masked(self, val):
        if self.mask is not None:
            self[torch.from_numpy(self.mask)] = val
        else:
            self[...] = val

    def symbolic(self):
        return self.symbolic_data

    def array_if_symbolic(self) -> SymbolicArray:
        if self.requires_symbolic:
            return self.symbolic_data

        return self

    def array(self):
        if self.requires_symbolic:
            return self.array_if_symbolic()

        return self.concrete()

    def to(self, *args, **kwargs) -> T:
        if len(args) == 1 and isinstance(args[0], Solver):
            self.solver = args[0]
            return self

        elif 'solver' in kwargs:
            self.solver = kwargs['solver']
            return self

        return super().to(*args, **kwargs)

    @property
    def requires_symbolic(self):
        return self._requires_symbolic

    @requires_symbolic.setter
    def requires_symbolic(self, mode):
        self._requires_symbolic = mode
        if self.requires_symbolic:
            self.to(solver=Solver.override() or self.solver or Solver.fallback() or _raise(
                RuntimeError(
                    "No available solver for `.requires_symbolic_(True)`. Try "
                    "`.to(solver).requires_symbolic_()` or to create a (override "
                    "or fallback) solver context `with solver: ...`. "
                )
            ))

    @property
    def lb(self):
        return self._lb

    # @lb.setter
    # def lb(self, value): ...
    #     if self.lb != value:
    #         self.outdated = True
    #     self._lb = value

    @property
    def ub(self):
        return self._ub

    # @ub.setter
    # def ub(self, value): ...
    #     if self.ub != value:
    #         self.outdated = True
    #     self._ub = value

    @property
    def bound(self):
        return self.lb, self.ub

    @bound.setter
    def bound(self, value):
        self.lb, self.ub = value

    @property
    def mask(self):
        return self._mask

    @mask.setter
    def mask(self, value):
        self._mask = value

    def requires_symbolic_(self: T, mode: bool=True, lb=None, ub=None, mask=None, name=None) -> T:
        """ Mark this parameter as symbolic or not. """
        self.requires_symbolic = mode
        self._lb = lb
        self._ub = ub
        self._name=name
        self.mask = mask
        return self
