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

def test_epsilon():
    """Test the epsilon context manager `sytorch.solver.epsilon`"""

    default_eps = epsilon.eps

    for local_eps in np.random.rand(10):
        with epsilon(local_eps):
            assert epsilon.eps == local_eps
        assert epsilon.eps == default_eps

    for local_eps in np.random.rand(1):
        with epsilon(local_eps):
            assert epsilon.eps == local_eps
            epsilon.reset()
            assert epsilon.eps == default_eps
        assert epsilon.eps == default_eps

def test_bound():
    """ test default bounds """
    # with GurobiSolver(): assert not solve(real() < -GRB.INFINITY)
    # with GurobiSolver(): assert not solve(real() >  GRB.INFINITY)

    """ test bounds """
    with epsilon(1e-5):
        for lb in (np.random.default_rng(seed=0).random(size=10) - .5) * 1e5:
            with GurobiSolver(): assert     solve(real(lb=lb, ub=lb+1.) == lb)
            with GurobiSolver(): assert not solve(real(lb=lb, ub=lb+1.) <  lb)
        for ub in (np.random.default_rng(seed=0).random(size=10) - .5) * 1e5:
            with GurobiSolver(): assert     solve(real(lb=ub-1., ub=ub) == ub)
            with GurobiSolver(): assert not solve(real(lb=ub-1., ub=ub) >  ub)

def test_bools():
    solver = GurobiSolver()

    # Create scalar of bool variable.
    assert solver.bool().item().VType == 'B'

    for shape in ((3,), (3, 4), (3, 4, 5)):
        # Create array of bool variables from shape.
        bits = solver.bools(shape)
        assert bits.shape == shape
        assert bits.item(0).VType == 'B'

        # Create array of bool variables from a reference array.
        bits_like = solver.bools_like(bits)
        assert bits_like.shape == shape
        assert bits_like.item(0).VType == 'B'

def test_constraints_contradiction():
    with epsilon(1e-5), GurobiSolver():
        x = real()
        assert not solve(1 < x, x < 1)

def test_optimize():
    for lb in (np.random.default_rng(seed=0).random(size=10) - .5) * 1e5:
        with GurobiSolver():
            x = real(lb=lb-1., ub=lb+1.)
            assert solve(x >= lb, minimize=x)
            assert np.allclose(x.evaluate(), lb)

    for ub in (np.random.default_rng(seed=0).random(size=10) - .5) * 1e5:
        with GurobiSolver():
            x = real(lb=ub-1., ub=ub+1.)
            assert solve(x <= ub, maximize=x)
            assert np.allclose(x.evaluate(), ub)

if IN_BAZEL:
    main(__name__, __file__)
