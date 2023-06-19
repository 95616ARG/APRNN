from __future__ import annotations
import os
from typing import Literal, overload

from sytorch.pervasives import *
from sytorch.util import gurobi as util
from sytorch.solver.symbolic_array import *
from sytorch.solver.base import *

import gurobipy as gp
from gurobipy import GRB

class GurobiSolver(Solver):
    def __init__(self, path=None):
        # warnings.warn("TODO(anonymous): reset default gurobi lower-bounds.")
        if path is not None:
            self.solver = gp.read(path)
        else:
            self.solver = gp.Model()
        self._lazy=True
        self.lazy_(True)
        self._verbose=True
        self.verbose_(True)
        self.solver.Params.Crossover = 0
        self.solver.Params.Method = 2
        self.solver.Params.Threads = os.cpu_count()
        self.solver.Params.Presolve = 1
        self.solver.Params.OptimalityTol = 1e-6

        self._zero = self.real()
        self.add_constraints(self._zero == 0)
        self._zero = self._zero.item()

    def lazy_(self, mode=True):
        self._lazy=mode
        return self

    def verbose_(self, mode=True):
        self._verbose=mode
        self.solver._env.setParam('OutputFlag', 1 if mode else 0)
        return self

    @property
    def supported_features(self):
        return (
            'lp',
            'qp',
            'milp',
            'miqp',
            'miqcp',
        )

    @property
    def stype(self):
        return stype.gurobi

    @property
    def timeout(self):
        return self.solver.Params.TimeLimit

    @timeout.setter
    def timeout(self, value):
        self.solver.Params.TimeLimit = value

    @overload
    def real(self, lb:float=-GRB.INFINITY , ub:float=GRB.INFINITY, **kwargs): ...

    def real(self, **kwargs):
        return SymbolicLPArray(np.array(util.reals(self.solver, 1, **kwargs)[0]), self)

    @overload
    def reals(self, shape, mask=None, lb:float=-GRB.INFINITY , ub:float=GRB.INFINITY, **kwargs): ...

    def reals(self, shape, mask=None, **kwargs):
        if mask is not None:
            size = np.broadcast_to(np.empty(1), shape)[mask].size
            out = np.empty(shape, dtype=object)
            out[mask] = util.reals(self.solver, (size,), **kwargs).reshape(out[mask].shape)
            return out.view(SymbolicLPArray).to(solver=self).reshape(shape)
        else:
            return SymbolicLPArray(util.reals(self.solver, shape, **kwargs), self)

    def reals_like(self, ref, mask=None, **kwargs):
        if mask is not None:
            raise NotImplementedError
        def _f(arr):
            if isinstance(arr, np.ndarray):
                return SymbolicLPArray(arr, self)
            return type(arr)(_f(arr) for arr in arr)
        return _f(util.reals_like(self.solver, ref, **kwargs))

    def bool(self, **kwargs):
        return SymbolicLPArray(np.array(util.bools(self.solver, 1, **kwargs)[0]), self).milp()

    def bools(self, shape, mask=None, **kwargs):
        if mask is not None:
            raise NotImplementedError
        return SymbolicLPArray(util.bools(self.solver, shape, **kwargs), self).milp()

    def bools_like(self, ref, mask=None, **kwargs):
        if mask is not None:
            raise NotImplementedError
        def _f(arr):
            if isinstance(arr, np.ndarray):
                return SymbolicLPArray(arr, self)
            return type(arr)(_f(arr) for arr in arr)
        return _f(util.bools_like(self.solver, ref, **kwargs)).milp()

    def add_constraints(self, *constraints):
        for constrs in constraints:
            for constr in _iter_nested_iterable(constrs):
                self.solver.addConstr(constr)
        if not self._lazy:
            self.update()

    _objective_mode_dict = {
        'minimize': GRB.MINIMIZE,
        'maximize': GRB.MAXIMIZE
    }

    def add_objectives(self, *objectives: SymbolicGurobiArray, mode: Literal['minimize', 'maximize']):
        if mode not in self._objective_mode_dict:
            raise NotImplementedError(
                f"unsupported optimization mode '{mode}' for Gurobi solver"
                f", expecting {tuple(self._objective_mode_dict.keys())}."
            )
        mode = self._objective_mode_dict[mode]
        for objective in objectives:
            if isinstance(objective, SymbolicArray):
                objective = objective.item()
            if not isinstance(objective, gp.Var):
                objective_expr = objective
                objective = self.real().item()
                self.solver.addConstr(objective == objective_expr)

            self.solver.setObjective(objective, mode)

        if not self._lazy:
            self.update()

    def minimize(self, *objectives):
        return self.add_objectives(*objectives, mode='minimize')

    def maximize(self, *objectives):
        return self.add_objectives(*objectives, mode='maximize')

    def evaluate(self, variables: SymbolicGurobiArray) -> torch.Tensor:
        try:
            return torch.from_numpy(util.evaluate(variables).array())
        except AttributeError:
            assert self.solve()
            return torch.from_numpy(util.evaluate(variables).array())

    def solve(self, *constraints: SymbolicGurobiArray, minimize=None, maximize=None, unlink=True, **kwargs) -> bool:
        self.add_constraints(*constraints)
        if minimize is not None:
            assert maximize is None
            if not isinstance(minimize, tuple): minimize = (minimize,)
            self.minimize(*minimize)
        elif maximize is not None:
            if not isinstance(maximize, tuple): maximize = (maximize,)
            self.maximize(*maximize)
        for k, v in kwargs.items():
            setattr(self.solver.Params, k, v)
        self.solver.update()
        self.solver.optimize()
        self.status = self.solver.status
        return self.solver.status == GRB.OPTIMAL

    def write(self, *args, **kwargs):
        self.update()
        return self.solver.write(*args, **kwargs)

    def update(self: T) -> T:
        self.solver.update()
        return self

    def print(self) -> str:
        self.update()
        return self.solver.display()

    def overapproximate(self, expr, inplace=True):
        assert inplace is True

        _obj = self.solver.getObjective()
        _verbose = self._verbose

        try:
            if _verbose:
                self.verbose_(False)
            obj = expr.sum()
            assert self.solve(minimize=obj)
            lbs = expr.evaluate()
            assert self.solve(maximize=obj)
            ubs = expr.evaluate()
            return torch.stack((lbs, ubs), -1)

        finally:
            self.solver.setObjective(_obj)
            self.verbose_(_verbose)

    def computeIIS(self):
        return self.solver.computeIIS()
