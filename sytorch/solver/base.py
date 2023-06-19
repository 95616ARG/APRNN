from __future__ import annotations
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
from typing import Callable, Iterable, Literal, Optional, Union, TypeVar

import z3
import gurobipy as gp
from enum import Enum
from sytorch.pervasives import *

_SolverT = Union[z3.Solver, gp.Model]
_SolverT1 = TypeVar("_SolverT1", bound=_SolverT)
_SolverT2 = TypeVar("_SolverT2", bound=_SolverT)
_SolverT3 = TypeVar("_SolverT3", bound=_SolverT)

class stype(Enum):
    gurobi = 1
    z3 = 2
    lightning = 3

class status(Enum):
    other       = -1
    init        = 0
    sat         = 1
    unsat       = 2
    timeout     = 3
    interrupted = 4
    error       = 5

_solver_context_stack = [(None, 'fallback')]

class _MetaDefaultSolver(type):

    def default(cls) -> Solver:
        return _solver_context_stack[-1][0]

    def fallback(cls) -> Optional[Solver]:
        solver, mode = _solver_context_stack[-1]
        if mode == 'fallback':
            return solver
        return None

    def override(cls) -> Optional[Solver]:
        solver, mode = _solver_context_stack[-1]
        if mode == 'override':
            return solver
        return None

    # def temporary(cls) -> Optional[Solver]:
    #     solver, mode = _solver_context_stack[-1]
    #     if mode == 'temporary':
    #         return solver
    #     return None


""" Executor. """

def _global_sanity_check():
    if multiprocessing.current_process != 'MainProcess':
        raise RuntimeError(
            f"trying to handle constraints in child processes."
        )

""" Constraints handler. """
def _default_constrs_handler(*constrs: Iterable['SymbolicLightningArray']) -> None:
    """ The default constraint handler. It adds constraints to their solver.
    Parameters
    ==========
        constrs: an iterable of SymbolicArray.

    Returns
    =======
        None
    """
    for constr in constrs:
        constr.solver.add_constraints(constr)

_global_constrs_handler_stack = []

def push_constr_handler(constr_handler):
    global _global_constrs_handler_stack
    _global_constrs_handler_stack.append(constr_handler)

def pop_constr_handler():
    global _global_constrs_handler_stack
    return _global_constrs_handler_stack.pop()

class ConstrHandler:

    def __init__(self, handler):
        self.handler = handler
        self.lock = multiprocessing.Lock()

    def __call__(self, *args, **kwargs):
        with self.lock:
            return self.handler(*args, **kwargs)

    def __enter__(self):
        push_constr_handler(self)

    def __exit__(self, exc_type, exc_val, exc_tb):
        assert pop_constr_handler() == self

_global_constrs_handler_stack.append(ConstrHandler(_default_constrs_handler))

def _get_constr_handler_unsafe():
    global _global_constrs_handler_stack
    if len(_global_constrs_handler_stack) > 0:
        return _global_constrs_handler_stack[-1]
    else:
        return None

def handle_constrs(constrs, **kwargs):
    return _get_constr_handler_unsafe()(constrs, **kwargs)

class Solver(metaclass=_MetaDefaultSolver):

    def __init__(self, *params: Iterable['Parameter']):
        for param in params:
            param.to_(self.solver)

    def __enter__(self):
        global _solver_context_stack
        _solver_context_stack.append((self, 'fallback'))
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        global _solver_context_stack
        _solver_context_stack = _solver_context_stack[:-1]

    def as_fallback(self):
        global _solver_context_stack
        solver, _ = _solver_context_stack[-1]
        _solver_context_stack[-1] = (solver, 'fallback')

    def as_override(self):
        global _solver_context_stack
        solver, _ = _solver_context_stack[-1]
        _solver_context_stack[-1] = (solver, 'override')

    # def as_temporary(self):
    #     global _solver_context_stack
    #     solver, _ = _solver_context_stack[-1]
    #     _solver_context_stack[-1] = (solver, 'temporary')

    @property
    def supported_features(self) -> Iterable[Literal['lp', 'qp', 'milp', 'miqp', 'smt']]:
        return tuple()

    def supports(self, feature: Literal['lp', 'qp', 'milp', 'miqp', 'smt']) -> bool:
        """ Report if the solver supports the specified feature.

        Parameters
        ========
        feature(str): feature to check, accepts
            - 'lp' for Linear Programming
            - 'qp' for Quadratic Programming
            - 'milp' for Mixed Integer Linear Programming
            - 'miqp' for Mixed Integer Quadratic Programming
            - 'smt' for Satisfiability Modulo Theories
        """
        return feature in self.supported_features

    @property
    def stype(self) -> stype:
        raise NotImplementedError

    @property
    def status(self) -> status:
        return self._status

    @status.setter
    def status(self, value) -> status:
        self._status = value

    @property
    def timeout(self):
        raise NotImplementedError

    @timeout.setter
    def timeout(self, value):
        raise NotImplementedError

    def apply_(self, fn: Callable[[_SolverT1], None]) -> bool:
        """ Apply inplace opeartions on this solver. """
        raise NotImplementedError

    """ Variable creation """

    def real(self, **kwargs):
        raise NotImplementedError

    def reals(self, shape, **kwargs):
        raise NotImplementedError

    def reals_like(self, ref, **kwargs):
        raise NotImplementedError

    def bools(self, shape, **kwargs):
        raise NotImplementedError

    def bools_like(self, ref, **kwargs):
        raise NotImplementedError

    def ints(self, shape, **kwargs):
        raise NotImplementedError

    def ints_like(self, ref, **kwargs):
        raise NotImplementedError

    def vars(self, shape, **kwargs):
        raise NotImplementedError

    def vars_like(self, ref, **kwargs):
        raise NotImplementedError

    def evaluate(self, variables):
        raise NotImplementedError

    """  """

    def add_implicit_constraints(self, *args, **kwargs):
        return self.add_constraints(*args, **kwargs)

    def add_constraints(self, constraints):
        raise NotImplementedError

    def add_objective(self, objective):
        raise NotImplementedError

    def minimize(self, objective):
        raise NotImplementedError

    def maximize(self, objective):
        raise NotImplementedError

    def solve(self) -> bool:
        raise NotImplementedError

    def print(self) -> str:
        raise NotImplementedError

    def reset(self) -> None:
        raise NotImplementedError

_eps = 1e-3

def set_epsilon(eps: float) -> None:
    global _eps
    _eps = eps

def get_epsilon() -> float:
    global _eps
    return _eps

class _MetaEpsilon(type):

    @property
    def eps(cls):
        return get_epsilon()

class epsilon(metaclass=_MetaEpsilon):
    def __init__(self, eps):
        self.local_eps = eps

    def __enter__(self):
        self.prev_eps = get_epsilon()
        set_epsilon(self.local_eps)
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        set_epsilon(self.prev_eps)

    @classmethod
    def reset(cls):
        set_epsilon(1.e-3)

@add_method_to_class(gp.Var)
def is_semi_positive(self):
    if not hasattr(self, 'LB'):
        return self >= 0.
    else:
        lb = self.LB
        assert lb is not None
        return lb >= 0.

@add_method_to_class(gp.Var)
def definitely_semi_positive(self):
    assert False
    lb = self.LB
    if lb is None:
        return False
    else:
        return lb >= 0.

@add_method_to_class(gp.Var)
def is_semi_negative(self):
    if not hasattr(self, 'UB'):
        return self <= 0.
    else:
        ub = self.UB
        assert ub is not None
        return ub <= 0.

@add_method_to_class(gp.Var)
def definitely_semi_negative(self):
    assert False
    ub = self.UB
    if ub is None:
        return False
    else:
        return ub <= 0.

@add_method_to_class(gp.Var)
def semi_positivity(self):
    if not hasattr(self, 'LB'):
        if self >= 0.:
            return 1
        else:
            return -1
    else:
        lb, ub = self.LB, self.UB
        if lb >= 0.:
            return 1
        elif ub <= 0.:
            return -1
        else:
            return 0.

@add_method_to_class(gp.Var)
def make_semi_positive(self):
    if not hasattr(self, 'LB'):
        assert self >= 0.
    else:
        assert self.UB is None or self.UB >= 0.
        if self.LB is None or self.LB < 0.:
            self.LB = 0.

@add_method_to_class(gp.Var)
def make_semi_negative(self):
    if not hasattr(self, 'UB'):
        assert self <= 0.
    else:
        assert self.LB is None or self.LB <= 0.
        if self.UB is None or self.UB > 0.:
            self.UB = 0.
