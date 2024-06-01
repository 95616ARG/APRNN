from __future__ import annotations
from typing import Any, TypeVar, overload, Iterable, Literal
import warnings

import scipy
import scipy.sparse

from sytorch.pervasives import *
from sytorch.solver.base import *
import gurobipy as gp

T = TypeVar('T', bound='SymbolicArray')
class SymbolicArray(np.ndarray):

    permute = np.ndarray.transpose

    def __new__(cls, input_array: np.ndarray, solver: Solver, mask=None) -> SymbolicArray:
        obj = np.asarray(input_array).view(cls)
        obj.solver = solver
        obj.mask = mask
        return obj

    def __array_finalize__(self, obj: SymbolicArray) -> None:
        if obj is None: return
        self.solver: Solver = getattr(obj, 'solver', None)
        self.mask = getattr(obj, 'mask', None)

    # def __repr__(self) -> str:
    #     return f"{type(self).__name__}({self}),\nsolver={self.solver}"

    def update(self):
        return self.solver.update()

    def ok(self) -> bool:
        raise NotImplementedError

    """ sanity checks """
    @classmethod
    def _is_all_compatible(cls, arrays: Iterable[SymbolicArray]) -> bool:
        if len(arrays) < 2:
            return True
        return all(array.solver == arrays[0].solver for array in arrays[1:])

    @classmethod
    def _check_is_all_compatible(cls, arrays: Iterable[SymbolicArray]) -> bool:
        if not cls._is_all_compatible(arrays):
                raise TypeError(
                    f"expecting all SymbolicArrays on the same backend solver."
                )

    def _is_compatible_with(self, other):
        try:
            self._check_is_compatible_with(other)
            return True
        except Exception as e:
            return False

    def _check_is_compatible_with(self, other):
        if isinstance(other, SymbolicArray):
            if self.solver != other.solver:
                raise TypeError(
                    f"expecting SymbolicArrays on the same backend solver, but got "
                    f"{self.stype}({self.solver}) and {other.stype}({other.solver})."
                )
        else:
            return True

    @staticmethod
    def concatenate(arrays: Iterable[T], **kwargs) -> T:
        # type(arrays[0])._check_is_all_compatible(arrays)
        # return type(arrays[0])(np.concatenate(arrays, **kwargs), arrays[0].solver)

        arrays = tuple(arr.cpu().detach().numpy() if isinstance(arr, Tensor) else arr for arr in arrays)
        ty = None
        solver = None
        for arr in arrays:
            if isinstance(arr, SymbolicArray):
                if ty is None:
                    ty = type(arr)
                    solver = arr.solver
                else:
                    assert ty == type(arr)
                    assert solver == arr.solver
        return np.concatenate(arrays, **kwargs).view(ty).to(solver)

    @staticmethod
    def stack(arrays: Iterable[T], **kwargs) -> T:
        arrays = tuple(arr.cpu().detach().numpy() if isinstance(arr, Tensor) else arr for arr in arrays)
        ty = None
        solver = None
        for arr in arrays:
            if isinstance(arr, SymbolicArray):
                if ty is None:
                    ty = type(arr)
                    solver = arr.solver
                else:
                    assert ty == type(arr)
                    assert solver == arr.solver
        return np.stack(arrays, **kwargs).view(ty).to(solver)
        # return type(arrays[0])(np.stack(arrays, **kwargs), arrays[0].solver)

    def moveaxis(self, source, destination):
        return np.moveaxis(self, source, destination)

    @property
    def stype(self: T):
        return self.solver.stype

    @property
    def eps(self: T) -> float:
        return epsilon.eps

    def array(self: T) -> np.ndarray:
        """ Returns self as np.ndarray. """
        return self.view(np.ndarray)

    """ operators """
    def evaluate(self: T) -> np.ndarray:
        return self.solver.evaluate(self)

    def to(self, solver):
        assert self.solver == None or self.solver == solver, \
            f"unimplemented translation from {self.solver} to {solver}."
        self.solver = solver
        return self

    def copy(self: T) -> T:
        return self.view(ndarray).copy().view(type(self)).to(self.solver)

    def overapproximate(self, **kwargs):
        return self.solver.overapproximate(self, **kwargs)

    @property
    def LBs(self):
        return np.fromiter(
            map((lambda x: x.LB if hasattr(x, 'LB') else x), self.flat),
            count=self.size,
            dtype=float
        ).reshape(self.shape)

    @LBs.setter
    def LBs(self, value):
        for v, lb in zip(self.flat, value.flat):
            if hasattr(v, 'LB'):
                v.LB = lb
            else:
                assert v >= lb
        self.solver.update()

    @property
    def UBs(self):
        return np.fromiter(
            map((lambda x: x.UB if hasattr(x, 'UB') else x), self.flat),
            count=self.size,
            dtype=float
        ).reshape(self.shape)

    @UBs.setter
    def UBs(self, value):
        for v, ub in zip(self.flat, value.flat):
            if hasattr(v, 'UB'):
                v.UB = ub
            else:
                assert v <= ub
        self.solver.update()

    @property
    def bounds(self):
        return np.stack((self.LBs, self.UBs), axis=-1)

    @bounds.setter
    def bounds(self, value):
        if isinstance(value, tuple):
            lbs, ubs = value
        else:
            lbs, ubs = value[...,0], value[...,1]
        for v, lb, ub in zip(self.flat, lbs.flat, ubs.flat):
            if hasattr(v, 'LB'):
                v.LB = lb
                v.UB = ub
            else:
                assert v >= lb and v <= ub

        self.solver.update()

    def tighten_bounds(self, bounds):
        if isinstance(bounds, tuple):
            lbs, ubs = bounds
        else:
            lbs, ubs = bounds[...,0], bounds[...,1]
        for v, lb, ub in zip(self.flat, lbs.flat, ubs.flat):
            if hasattr(v, 'LB'):
                if lb > v.LB:
                    v.LB = lb
                if ub < v.ub:
                    v.UB = ub
            else:
                assert v >= lb and v <= ub

        self.solver.update()

    def is_semi_positive(self):
        return np.fromiter(
            map((lambda x: is_semi_positive(x)), self.flat),
            count=self.size,
            dtype=bool
        ).reshape(self.shape)

    def is_semi_negative(self):
        return np.fromiter(
            map((lambda x: is_semi_negative(x)), self.flat),
            count=self.size,
            dtype=bool
        ).reshape(self.shape)

    def semi_positivity(self):
        return np.fromiter(
            map((lambda x: semi_positivity(x)), self.flat),
            count=self.size,
            dtype=int
        ).reshape(self.shape)

    """ magic methods """

class SymbolicGurobiArray(SymbolicArray):

    def get_gurobi_indices(self, order='C'):
        out = np.empty(self.size, dtype=int, order=order)
        for i, v in enumerate(self.flat):
            if isinstance(v, gp.Var):
                out[i] = v.index
            else:
                out[i] = -1
        return out.reshape(self.shape)

    def ok(self) -> bool:
        raise NotImplementedError(
            f"unimplemented .ok() for {type(self)}."
        )

class SymbolicGurobiConstrArray(SymbolicGurobiArray):

    def ok(self) -> bool:
        """ Returns True iff all variables are alive on their solver. """
        for constr in self.flat:
            for i in range(constr._lhs.size()):
                if constr._lhs.getVar(i).index < 0:
                    return False
        return True

    def __invert__(self):
        """ Returns the negate of `gurobipy.TempConstr` elements. Specifically,
        - for `lhs >= rhs` (whose ._sense is '>'), it returns `lhs + eps <= lhs`;
        - for `lhs <= rhs` (whose ._sense is '<'), it returns `lhs - eps <= lhs`;
        NOTE(anonymous):
        - We always add/subtract `eps` to `lhs` because we also encode strict
        inequalities in this way, and ideally `eps` can be eliminated.
        - Due to the floating number weirdness, `x + eps + const - eps` may not
        give the ideal `x + const`.
        """
        assert isinstance(self.item(0), gp.TempConstr)
        return np.vectorize(
            lambda constr:\
                constr._lhs + self.eps <= constr._rhs if constr._sense == '>' else
                constr._lhs - self.eps >= constr._rhs if constr._sense == '<' else
                _raise(NotImplementedError(
                    f"unimplemented __invert__ for {constr}({type(constr)})"
                )),
            otypes=[object]
        )(self).view(type(self))

    def __eq__(self: T, other) -> T:
        assert isinstance(other, SymbolicGurobiArray)

        if isinstance(other.item(0), gp.Var) and other.item(0).VType == 'B':
            return np.stack([
                (other == 1) >> self,
                (other == 0) >> ~self
            ], axis=-1).view(type(self))

        elif isinstance(other.item(0), gp.TempConstr):
            return self == other.alias()

        raise NotImplementedError(
            f"unimplemented __eq__ (iff) for {self} and {other}."
        )

    def alias(self: T) -> 'SymbolicLPArray':
        bits = self.solver.bools_like(self)
        self.solver.add_constraints(self == bits)
        return bits

    def any(self, axis=None, keepdims=False):
        return self.alias().any(axis=axis, keepdims=keepdims) == 1


""" Subclassing `SymbolicGurobiArray` for operations in different types of
programming. For example, `SymbolicLPArray` only supports LP operations, which
doesn't include a precise encoding of `abs`, `max` and `norm`. It only supports
the upper bounds `abs_ub`, `max_ub` and `norm_ub`, which can be used to
linearize the `abs`, `max` and `norm` encoding and reformulate the certain MIP
problem to LP problem.
One can use `.milp()` to get a `SymbolicMILPArray` view of the same underlying
array, which provides precise MILP encoding of `abs`, `max` and `norm`.
"""

class SymbolicLPArray(SymbolicGurobiArray):

    def ok(self) -> bool:
        """ Returns True iff all variables are alive on their solver. """
        if isinstance(self.item(0), gp.Var):
            for var in self.flat:
                if var.index < 0:
                    return False
            return True
        else:
            for expr in self.flat:
                for i in range(expr.size()):
                    if expr.getVar(i).index < 0:
                        return False
            return True

    def mvar(self: T) -> gp.MVar:
        """ Returns self as Gurobi's matrix (or variable if self is 0d). """
        try:
            return gp.MVar(self.array())
        except AttributeError:
            return gp.MVar(self.alias().array())

    def lp(self: T) -> SymbolicLPArray:
        """ Returns self as `SymbolicLPArray`. """
        assert self.solver.supports('lp')
        return self.view(SymbolicLPArray)

    def qp(self: T) -> SymbolicQPArray:
        """ Returns self as `SymbolicQPArray`. """
        assert self.solver.supports('qp')
        return self.view(SymbolicQPArray)

    def milp(self: T) -> SymbolicMILPArray:
        """ Returns self as `SymbolicMILPArray`. """
        assert self.solver.supports('milp')
        return self.view(SymbolicMILPArray)

    def miqp(self: T) -> SymbolicMIQPArray:
        """ Returns self as `SymbolicMIQPArray`. """
        assert self.solver.supports('miqp')
        return self.view(SymbolicMIQPArray)

    @classmethod
    def from_mvar(cls, mvar, solver: Solver):
        """ Convert from Gurobi's matrix. """
        dummy = solver.reals(mvar.shape).view(cls)
        solver.add_constraints(dummy.mvar() == mvar)
        return dummy

    def alias2(self: T) -> T:
        alias = self.solver.reals_like(self)
        self.solver.add_constraints(self == alias)
        return alias

    def isnav(self):
        # nav for not-a-var
        return np.vectorize(
            lambda v: not isinstance(v, gp.Var),
            signature="()->()",
            otypes=[bool],
        )(self)

    def alias(self: T, conservative=True) -> T:
        # import warnings
        # warnings.warn('disabled aliasing.')
        # return self

        if conservative:
            mask = self.isnav()
            alias = np.empty(self.shape, dtype=object).view(type(self)).to(self.solver)
            alias[mask] = self[mask].alias(conservative=False)
            alias[~mask] = self[~mask]
            return alias

        else:
            if self.size == 0:
                return self
            alias = self.solver.reals_like(self)
            self.solver.add_constraints(self == alias)
            return alias

    """ operations """

    def argmax(self: T, axis: int) -> ArgMaxEncoder:
        """ Returns an argmax constraints encoder for self. """
        return ArgMaxEncoder(self, axis)

    def argmin(self: T, axis: int) -> ArgMinEncoder:
        """ Returns an argmin constraints encoder for self. """
        return ArgMinEncoder(self, axis)

    def gradient(self, h, dim):
        grad = type(self)(gradient(self, h, dim),
                             solver=self.solver)
        return grad

    def divergence(self, h, ndim):
        div = type(self)(divergence(self, h, ndim=ndim),
                            solver=self.solver)
        return div

    """ magic methods """

    """ comparisons """
    def __eq__(self: T, other: SymbolicGurobiArray | Tensor | np.ndarray | Any) -> T:
        self._check_is_compatible_with(other)
        if _isGurobiConstr(other):
            return other == self
        lhs, rhs = _to_operands(self, other)
        return SymbolicGurobiConstrArray(np.equal(lhs, rhs, dtype=object), self.solver)

    def __ne__(self: T, other: SymbolicGurobiArray | Tensor | np.ndarray | Any) -> T:
        raise NotImplementedError(
            f"unimplemented __ne__ for {type(self)}, perhaps try .milp()."
        )

    def __ge__(self: T, other: Any) -> T:
        self._check_is_compatible_with(other)
        if _isGurobiConstr(other):
            return other <= self
        lhs, rhs = _to_operands(self, other)
        return SymbolicGurobiConstrArray(np.greater_equal(lhs, rhs, dtype=object), self.solver)

    def __gt__(self, other):
        return self - self.eps >= other

    def __le__(self, other):
        self._check_is_compatible_with(other)
        if _isGurobiConstr(other):
            return other >= self
        lhs, rhs = _to_operands(self, other)
        return SymbolicGurobiConstrArray(np.less_equal(lhs, rhs, dtype=object), self.solver)

    def __lt__(self, other):
        return self + self.eps <= other

    """ arithmetic operators """
    def __add__(self, other: Any):
        self._check_is_compatible_with(other)
        if _isGurobiConstr(other):
            raise NotImplementedError(
                f"unimplemented __eq__ for {self} and {other}."
            )
        lhs, rhs = _to_operands(self, other)
        return type(self)(np.add(lhs, rhs, dtype=object), self.solver)

    def __radd__(self, other: Any):
        return self + other

    def __sub__(self, other: Any):
        self._check_is_compatible_with(other)
        if _isGurobiConstr(other):
            raise NotImplementedError(
                f"unimplemented __eq__ for {self} and {other}."
            )
        lhs, rhs = _to_operands(self, other)
        return type(self)(np.subtract(lhs, rhs, dtype=object), self.solver)

    def __rsub__(self, other: Any):
        self._check_is_compatible_with(other)
        if _isGurobiConstr(other):
            raise NotImplementedError(
                f"unimplemented __eq__ for {self} and {other}."
            )
        lhs, rhs = _to_operands(other, self)
        return type(self)(np.subtract(lhs, rhs, dtype=object), self.solver)

    def __matmul__(self, other: Any):
        self._check_is_compatible_with(other)
        if _isGurobiConstr(other):
            raise NotImplementedError(
                f"unimplemented __eq__ for {self} and {other}."
            )
        lhs, rhs = _to_operands(self, other)
        assert lhs.ndim >= 1 and rhs.ndim >= 1

        M_shapes = lhs.shape[:-1]
        P_shapes = rhs.shape[1:]

        lhs = lhs.reshape((-1, lhs.shape[-1]))
        rhs = rhs.reshape((rhs.shape[0], -1))

        M, N = lhs.shape
        N, P = rhs.shape
        out = np.empty((M, P), dtype=object)
        # from tqdm.auto import tqdm
        # for mi, pi in tqdm(np.ndindex(out.shape), total=np.prod(out.shape), desc=f'matmul ({M},{N}) @ ({N},{P})', leave=False):
        for mi, pi in np.ndindex(out.shape):
            out[mi, pi] = gp.quicksum(lhs[mi] * rhs[:,pi])
        out = out.view(type(self)).to(self.solver)
        out = out.reshape((*M_shapes, *P_shapes))
        return out

    def __rmatmul__(self, other) -> SymbolicArray:
        self._check_is_compatible_with(other)
        if _isGurobiConstr(other):
            raise NotImplementedError(
                f"unimplemented __eq__ for {self} and {other}."
            )
        lhs, rhs = _to_operands(other, self)
        assert lhs.ndim >= 1 and rhs.ndim >= 1

        M_shapes = lhs.shape[:-1]
        P_shapes = rhs.shape[1:]

        lhs = lhs.reshape((-1, lhs.shape[-1]))
        rhs = rhs.reshape((rhs.shape[0], -1))

        M, N = lhs.shape
        N, P = rhs.shape
        out = np.empty((M, P), dtype=object)
        # from tqdm.auto import tqdm
        # for mi, pi in tqdm(np.ndindex(out.shape), total=np.prod(out.shape), desc=f'matmul ({M},{N}) @ ({N},{P})', leave=False):
        for mi, pi in np.ndindex(out.shape):
            out[mi, pi] = gp.quicksum(lhs[mi] * rhs[:,pi])
        out = out.view(type(self)).to(self.solver)
        out = out.reshape((*M_shapes, *P_shapes))
        return out

    # def sum(self, axis=None):
    #     if axis is None:
    #         axis = 0
    #         arr = self.reshape(-1)
    #     else:
    #         arr = self

    #     return np.apply_along_axis(gp.quicksum, axis=axis, arr=arr).view(type(self)).to(solver=self.solver)
    #     # return type(self)((gp.quicksum(self.flat),), solver=self.solver).reshape(())

    def sum(self, axis=None, keepdims=False):
        assert keepdims is False
        if self.size == 0:
            return 0

        if axis == None:
            self = self.reshape(-1)
            axis = 0

        # type(self).moveaxis(self, axis, -1)
        # A = broadcast_at(np.ones(1,), 0, ())

        return np.apply_along_axis(gp.quicksum, axis, self).view(type(self)).to(solver=self.solver)

    def mean(self, axis=None, keepdims=False):
        assert keepdims is False
        if self.size == 0:
            return 0

        if axis == None:
            self = self.reshape(-1)
            axis = 0

        num = self.shape[axis]
        return (self.sum(axis=axis, keepdims=keepdims).alias() / num).alias()

    def abs_ub(self) -> SymbolicLPArray:
        """ Return variables encoded as the upper bound of self's absolute
        value. For each element `x` of self, the corresponding upper bound `u`
        of `|x|` is defined as `x <= u` and `-x <= u`.
        Such upper bound `u` can be used to replace `|x|` in minimization
        objectives when `|x|` appears without negation, or in maximization
        objectives when `|x|` appears with negation (e.g., `-|x|`).
        See this page for detail:
        https://optimization.mccormick.northwestern.edu/index.php/Optimization_with_absolute_values
        TODO(anonymous): Set lower and upper bounds of `abs_ubs`.
        See issue for details: https://github.com/95616ARG/indra/issues/806
        """
        assert self.size > 0
        abs_ubs = self.solver.reals_like(self)
        vars = type(self).concatenate((self.flatten().alias(conservative=True), abs_ubs.flatten())).mvar()
        # vars = self.flatten().tolist() + abs_ubs.flatten().tolist()

        A = scipy.sparse.diags([1., -1.], [0, self.size],
                            shape=(self.size, self.size*2),
                            dtype=np.float64, format="lil")
        b = np.zeros(self.size)
        self.solver.solver.addMConstr(A, vars, '<', b)

        A = scipy.sparse.diags([-1., -1.], [0, self.size],
                            shape=(self.size, self.size*2),
                            dtype=np.float64, format="lil")
        self.solver.solver.addMConstr(A, vars, '<', b)

        return abs_ubs

    def max_ub(self) -> SymbolicLPArray:
        max_ub = self.solver.real()
        vars = type(self).concatenate((self.flatten().alias(conservative=True), max_ub.flatten())).mvar()
        # vars = self.flatten().tolist() + max_ub[None].tolist()

        A = scipy.sparse.eye(self.size, self.size+1, dtype=np.float64, format='lil')
        A[:, -1] = -1.
        b = np.zeros(self.size)
        self.solver.solver.addMConstr(A, vars, '<', b)

        return max_ub

    def norm_ub(self, order: Literal['l1', 'l1_normalized', 'linf', 'linf+l1_normalized']='l1') -> SymbolicLPArray:
        """Returns the norm specified by order. The l1_normalized specification
        returns the l1 norm divided by the number of variables in the SymbolicArray.
        """
        assert self.ndim == 1
        # print(f"taking {order} norm over {self.size} variables.")
        if order == 'linf+l1_normalized':
            abs_ub = self.abs_ub()
            return (abs_ub.sum() / self.size) + abs_ub.max_ub()

        elif order == 'l1':
            return self.abs_ub().sum()

        elif order == 'l1_normalized':
            return self.norm_ub('l1') / self.size

        elif order == 'linf':
            return self.abs_ub().max_ub()

        raise NotImplementedError(
            f"unsupported norm order {order} for {type(self)}, perhaps try "
            f"`.milp().norm(...)`."
        )

    def any(self):
        raise NotImplementedError

    def all(self):
        raise NotImplementedError

class SymbolicMILPArray(SymbolicLPArray):

    def lp(self):
        raise TypeError(
            "unsupported conversion from SymbolicMILPArray to SymbolicLPArray."
        )

    def qp(self):
        raise TypeError(
            "unsupported conversion from SymbolicMILPArray to SymbolicQPArray."
        )

    def __ne__(self: T, other: SymbolicGurobiArray | Tensor | np.ndarray | Any) -> T:
        self._check_is_compatible_with(other)
        if _isGurobiConstr(other):
            raise NotImplementedError(
                f"unimplemented __eq__ for {self} and {other}."
            )
        lhs, rhs = _to_operands(self, other)
        return SymbolicGurobiConstrArray(
            np.stack([lhs > rhs, lhs < rhs], axis=-1).any(axis=-1) == 1, self.solver
        )

    def all(self: T, axis=None, keepdims=False):
        assert keepdims is False
        if self.size == 0:
            raise NotImplementedError
            # TODO(anonymous): return 0 or a symbolic array of 0?
            return 0

        if isinstance(self.item(0), gp.TempConstr):
            raise NotImplementedError
            bits = self.solver.bools_like(self).milp()
            self.solver.add_constraints((bits == 1) >> self)
            return bits.all(axis=axis, keepdims=keepdims)

        elif isinstance(self.item(0), gp.Var) and self.item(0).VType == 'B':
            def _all(arr):
                return gp.and_(arr.reshape(-1).tolist())

            if axis == None:
                self = self.reshape(-1)
                axis = 0

            gen_constrs = np.apply_along_axis(_all, axis, self)
            bits = self.solver.bools(gen_constrs.shape)
            self.solver.add_constraints(bits == gen_constrs)
            return bits

        else:
            raise NotImplementedError(
                f"unimplemented .any() for {self}"
            )

    def any(self: T, axis=None, keepdims=False):
        assert keepdims is False
        if self.size == 0:
            # TODO(anonymous): return 0 or a symbolic array of 0?
            return 0

        if isinstance(self.item(0), gp.TempConstr):
            bits = self.solver.bools_like(self).milp()
            self.solver.add_constraints((bits == 1) >> self)
            return bits.any(axis=axis, keepdims=keepdims)

        elif isinstance(self.item(0), gp.Var) and self.item(0).VType == 'B':
            def _any(arr):
                return gp.or_(arr.reshape(-1).tolist())

            if axis == None:
                self = self.reshape(-1)
                axis = 0

            gen_constrs = np.apply_along_axis(_any, axis, self)
            bits = self.solver.bools(gen_constrs.shape)
            self.solver.add_constraints(bits == gen_constrs)
            return bits

        else:
            raise NotImplementedError(
                f"unimplemented .any() for {self}"
            )

    def abs(self):
        """ TODO(anonymous): reuse abs. """
        vars = self.solver.reals_like(self).view(type(self))
        np.vectorize(
            lambda v, v_abs: \
                self.solver.add_constraints(v_abs == gp.abs_(v)),
            otypes=[]
        )(self.array(), vars)
        return vars

    def max(self):
        """ TODO(anonymous): 1. reuse max; 2. max along one axis """
        var_max = self.solver.real().view(type(self))
        self.solver.add_constraints(var_max.item() == gp.max_(*self.array().flatten()))
        return var_max

    def norm(self, order: Literal['l1', 'l1_normalized', 'linf']|int='l1'):
        """Returns the norm specified by order. The l1_normalized specification
        returns the l1 norm divided by the number of variables in the SymbolicArray.
        """
        assert self.ndim == 1
        if order == 'l1':
            return self.abs().sum()

        elif order == 'l1_normalized':
            return self.norm('l1') / self.size

        elif order == 'linf':
            return self.abs().max()

        warnings.warn(
            f"using Gurobi's built-in norm encoding with order {order}, "
            f"which might be slow."
        )
        norm_expr = gp.norm(self.mvar(), which=order)
        norm_var = self.solver.real().view(type(self))
        self.solver.add_constraints(norm_var.item() == norm_expr)
        return norm_var

class SymbolicQPArray(SymbolicLPArray):
    pass

class SymbolicMIQPArray(SymbolicMILPArray, SymbolicQPArray):
    pass

class ArgMaxEncoder(np.ndarray):
    def __new__(cls, input_array, axis):
        obj = np.asarray(input_array).view(cls)
        obj.axis = axis
        obj.solver = input_array.solver
        return obj

    def __array_finalize__(self, obj) -> None:
        if obj is None: return
        self.axis = getattr(obj, 'axis', None)
        self.solver = getattr(obj, 'solver', None)

    def __ge__(self, other):
        return np.vectorize(
            lambda a,b: a >= b + epsilon.eps,
            otypes=[object]
        )(self, other).array()

    def array(self):
        return self.view(np.ndarray)

    @overload
    def __eq__(self, label: int) -> np.ndarray: ...

    @overload
    def __eq__(self, label: np.ndarray) -> np.ndarray: ...

    def __eq__(self, label):
        axis = (self.axis + self.ndim) % self.ndim

        if isinstance(label, Tensor):
            label = label.cpu().detach().numpy()

        if isinstance(label, (int, np.integer)):

            idx_label = tuple(
                slice(label,label+1) if dim == axis else slice(None)
                for dim in range(self.ndim)
            )

            idx_other = tuple(
                list(range(0, label)) + list(range(label+1, self.shape[self.axis]))
                    if dim == axis
                    else slice(None)
                for dim in range(self.ndim)
            )

            # TODO(anonymous): use .addMConstr
            return SymbolicGurobiConstrArray(
                self[idx_label] >= self[idx_other],
                solver=self.solver
            )

        elif isinstance(label, np.ndarray) and \
             label.shape[-1] == 1 and \
             label.shape[:-1] == self.shape[:-1]:

            return SymbolicGurobiConstrArray(np.vectorize(
                lambda arr, lbl: arr == lbl,
                signature="(m),()->(k)",
                otypes=[object],
            )(self, label[...,0]), solver=self.solver)

        elif isinstance(label, np.ndarray) and label.ndim == 1:
            return SymbolicGurobiConstrArray(np.vectorize(
                lambda arr, lbl: arr == lbl,
                signature="(m),()->(k)",
                otypes=[object],
            )(self, label), solver=self.solver)

        raise NotImplementedError(
            f"{self.shape}.argmax({label.shape}, axis={axis})"
        )

class ArgMinEncoder(np.ndarray):
    def __new__(cls, input_array, axis):
        obj = np.asarray(input_array).view(cls)
        obj.axis = axis
        obj.solver = input_array.solver
        return obj

    def __array_finalize__(self, obj) -> None:
        if obj is None: return
        self.axis = getattr(obj, 'axis', None)
        self.solver = getattr(obj, 'solver', None)

    def __ge__(self, other):
        return np.vectorize(
            lambda a,b: a >= b + epsilon.eps,
            otypes=[object]
        )(self, other).array()

    def array(self):
        return self.view(np.ndarray)

    @overload
    def __eq__(self, label: int) -> np.ndarray: ...

    @overload
    def __eq__(self, label: np.ndarray) -> np.ndarray: ...

    def __eq__(self, label):
        axis = (self.axis + self.ndim) % self.ndim

        if isinstance(label, Tensor):
            label = label.cpu().detach().numpy()

        if isinstance(label, (int, np.integer)):

            idx_label = tuple(
                slice(label,label+1) if dim == axis else slice(None)
                for dim in range(self.ndim)
            )

            idx_other = tuple(
                list(range(0, label)) + list(range(label+1, self.shape[self.axis]))
                    if dim == axis
                    else slice(None)
                for dim in range(self.ndim)
            )

            return SymbolicGurobiConstrArray(
                self[idx_other] >= self[idx_label],
                solver=self.solver
            )

        elif isinstance(label, np.ndarray) and label.ndim == 1:
            return SymbolicGurobiConstrArray(np.vectorize(
                lambda arr, lbl: arr == lbl,
                signature="(m),()->(k)",
                otypes=[object],
            )(self, label), solver=self.solver)

        raise NotImplementedError

def _to_operand(x):
    """ Convert any ndarray- or Tensor-like objects to ndarray, and pass others
    as-is.
    """
    if isinstance(x, np.ndarray):
        return np.asarray(x)

    if isinstance(x, Tensor):
        x = x.cpu().detach().numpy()

    return x

def _to_operands(*xs):
    return tuple(_to_operand(x) for x in xs)

def _isGurobiConstr(x):
    if isinstance(x, np.ndarray):
        x = x.item(0)
    return isinstance(x, (gp.Constr, gp.GenConstr, gp.MConstr, gp.QConstr, gp.TempConstr))

def _isGurobiExpr(x):
    if isinstance(x, np.ndarray):
        x = x.item(0)
    return isinstance(x, (gp.GenExpr, gp.LinExpr, gp.MLinExpr, gp.MQuadExpr, gp.QuadExpr))

def indexing_at_dim_gen(dim, ndim):
    def indexing_at_dim(sl_dim):
        sl = [slice(None)] * ndim
        sl[dim] = sl_dim
        return tuple(sl)
    return indexing_at_dim

def gradient(F, h, dim):
    idx = indexing_at_dim_gen(dim, F.ndim)

    grad = np.empty_like(F)
    grad[idx(          0)] = F[idx(1)] - F[idx(0)]
    grad[idx(slice(1,-1))] =(F[idx(slice(2,None))] - F[idx(slice(None,-2))]) / 2
    grad[idx(         -1)] = F[idx(-1)] - F[idx(-2)]
    grad /= h

    return grad

def divergence(F, h, ndim):
    assert F.ndim in [ndim+1, ndim+2]
    assert F.shape[-ndim-1] == ndim

    return np.ufunc.reduce(
        np.add, [
            gradient(F[...,i,:,:], h[i], dim=-1-i)
            for i in range(2)
        ]
    )
