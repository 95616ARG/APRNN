# pylint: disable=import-error
try:
    from external.bazel_python.pytest_helper import main
    IN_BAZEL = True
except ImportError:
    IN_BAZEL = False

import itertools
import numpy as np
import sytorch
from sytorch.solver import *
from gurobipy import GRB

nshape = 3

def test_argmax():
    for shape, seed in itertools.product(
        (tuple(range(ndim+1, 1, -1)) for ndim in range(1, nshape+1)),
        range(3)
    ):
        solver = GurobiSolver()

        x = solver.reals(shape)
        y = np.random.default_rng(seed).random(shape)
        axis = np.random.default_rng(seed).choice(len(shape)*2) - len(shape)
        label = np.random.default_rng(seed).choice(shape[axis])

        solver.add_constraints((x + y).argmax(axis=axis) == label)
        assert solver.solve()
        assert ((x.evaluate().numpy() + y).argmax(axis=axis) == label).all()

def test_argmin():
    for shape, seed in itertools.product(
        (tuple(range(ndim+1, 1, -1)) for ndim in range(1, nshape+1)),
        range(3)
    ):
        solver = GurobiSolver()

        x = solver.reals(shape)
        y = np.random.default_rng(seed).random(shape)
        axis = np.random.default_rng(seed).choice(len(shape)*2) - len(shape)
        label = np.random.default_rng(seed).choice(shape[axis])

        solver.add_constraints((x + y).argmin(axis=axis) == label)
        assert solver.solve()
        assert ((x.evaluate().numpy() + y).argmin(axis=axis) == label).all()

def _np_norm_wrapper(array, order):
    if order == 'l1':
        return np.linalg.norm(array, ord=1)
    if order == 'l1_normalized':
        return np.linalg.norm(array, ord=1) / array.size
    if order == 'linf':
        return np.linalg.norm(array, ord=np.inf)
    raise RuntimeError(f"unsupported {order} norm.")

def test_abs_ub():
    for shape, seed in itertools.product(
        ((2,), (2, 3), (2, 3, 4)),
        range(3)
    ):
        solver = sytorch.GurobiSolver()
        x = solver.reals(shape)
        x_abs_ub = x.abs_ub()

        c = np.random.default_rng(seed).random(shape) * 1e3
        c_abs = np.abs(c)

        assert solver.solve(x == c, minimize=x_abs_ub.sum())
        assert (x_abs_ub.evaluate().numpy() >= c_abs).all()
        assert np.allclose(x_abs_ub.evaluate().numpy(), c_abs)

def test_abs():
    for shape, seed in itertools.product(
        ((2,), (2, 3), (2, 3, 4)),
        range(3)
    ):
        solver = sytorch.GurobiSolver()
        x = solver.reals(shape).milp()
        x_abs = x.abs()

        c = np.random.default_rng(seed).random(shape) * 1e3
        c_abs = np.abs(c)

        assert solver.solve(x == c, minimize=x_abs.sum())
        assert np.allclose(x_abs.evaluate().numpy(), c_abs)

def test_max_ub():
    for shape, seed in itertools.product(
        ((2,), (2, 3), (2, 3, 4)),
        range(3)
    ):
        solver = sytorch.GurobiSolver()
        x = solver.reals(shape)
        x_max_ub = x.max_ub()

        c = (np.random.default_rng(seed).random(shape) - .5) * 1e3
        c_max = np.max(c)

        assert solver.solve(x == c, minimize=x_max_ub.sum())
        assert (x_max_ub.evaluate().numpy() >= c_max).all()
        assert np.allclose(x_max_ub.evaluate().numpy(), c_max)

def test_max():
    for shape, seed in itertools.product(
        ((2,), (2, 3), (2, 3, 4)),
        range(3)
    ):
        solver = sytorch.GurobiSolver()
        x = solver.reals(shape).milp()
        x_max = x.max()

        c = (np.random.default_rng(seed).random(shape) - .5) * 1e3
        c_max = np.max(c)

        assert solver.solve(x == c, minimize=x_max.sum())
        assert np.allclose(x_max.evaluate().numpy(), c_max)

def test_norm_ub():
    for which_norm, shape, seed in itertools.product(
        ('l1', 'l1_normalized', 'linf'),
        ((5,),),
        range(3)
    ):
        solver = GurobiSolver()

        x = solver.reals(shape)
        x_norm_ub = x.norm_ub(order=which_norm).alias()

        c = np.random.default_rng(seed).random(shape)
        c_norm = _np_norm_wrapper(c, order=which_norm)

        # Assert the minimized norm upper-bound >= the real norm.
        assert solver.solve(x == c, minimize=x_norm_ub)
        assert (x_norm_ub.evaluate().numpy() + solver.solver.Params.FeasibilityTol >= c_norm)

        # Assert the norm upper-bound can not be smaller than the real norm.
        assert not solver.solve(x_norm_ub < c_norm)

def test_norm():
    for which_norm, shape, seed in itertools.product(
        ('l1', 'l1_normalized', 'linf'),
        ((5,),),
        range(3)
    ):
        solver = GurobiSolver()

        x = solver.reals(shape).milp()
        x_norm = x.norm(order=which_norm).alias()

        c = np.random.default_rng(seed).random(shape)
        c_norm = _np_norm_wrapper(c, order=which_norm)

        # Assert the solved norm is close to the real norm.
        assert solver.solve(x == c)
        assert np.allclose(x_norm.evaluate().numpy(), c_norm)

def test_symbolic_nd_add_concrete_0d_real():
    for shape, seed in itertools.product(
        (tuple(range(ndim+1, 1, -1)) for ndim in range(1, nshape+1)),
        range(3)
    ):
        with GurobiSolver():
            x = reals(shape)
            y = np.random.default_rng(seed).random(1)[0]
            b = np.random.default_rng(seed).random(shape) + y

            assert solve(x + y == b)
            assert np.allclose(x.evaluate().numpy() + y, b)

def test_symbolic_nd_add_concrete_nd_real():
    for shape, seed in itertools.product(
        (tuple(range(ndim+1, 1, -1)) for ndim in range(1, nshape+1)),
        range(3)
    ):
        solver = GurobiSolver()

        x = solver.reals(shape)
        y = np.random.default_rng(seed).random(shape)
        b = np.random.default_rng(seed).random(shape)

        solver.add_constraints(x + y == b)
        assert solver.solve()
        assert np.allclose(x.evaluate().numpy() + y, b)

def test_concrete_nd_radd_symbolic_nd_real():
    for shape, seed in itertools.product(
        (tuple(range(ndim+1, 1, -1)) for ndim in range(1, nshape+1)),
        range(3)
    ):
        solver = GurobiSolver()

        x = np.random.default_rng(seed).random(shape)
        y = solver.reals(shape)
        b = np.random.default_rng(seed).random(shape)

        solver.add_constraints(x + y == b)
        assert solver.solve()
        assert np.allclose(x + y.evaluate().numpy(), b)

def test_symbolic_nd_add_symbolic_nd_real():
    for shape, seed in itertools.product(
        (tuple(range(ndim+1, 1, -1)) for ndim in range(1, nshape+1)),
        range(3)
    ):
        solver = GurobiSolver()

        x = solver.reals(shape)
        y = solver.reals(shape)
        b = np.random.default_rng(seed).random(shape)

        solver.add_constraints(x + y == b)
        assert solver.solve()
        assert np.allclose(x.evaluate().numpy() + y.evaluate().numpy(), b)

def test_concrete_nd_rmatmul_symbolic_1d_real():
    for shape, seed in itertools.product(
        (tuple(range(ndim+1, 1, -1)) for ndim in range(1, nshape+1)),
        range(3)
    ):
        solver = GurobiSolver()

        A = np.random.default_rng(seed).random(shape)
        s = np.random.default_rng(seed).random(shape[-1])
        b = A @ s

        x = solver.reals_like(s)
        solver.add_constraints(A @ x == b)
        assert solver.solve()
        assert np.allclose(A @ x.evaluate().numpy(), b)

if IN_BAZEL:
    main(__name__, __file__)
