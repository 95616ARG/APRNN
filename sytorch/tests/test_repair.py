# pylint: disable=import-error
try:
    from external.bazel_python.pytest_helper import main
    IN_BAZEL = True
except ImportError:
    IN_BAZEL = False

import numpy as np
import sytorch
import sytorch.nn as nn
import random
import torch

from sytorch.solver.base import ConstrHandler

def set_seed(seed=0):
    sytorch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

def test_ddnn_repair():
    set_seed(29)
    dtype = sytorch.float64
    axis = 1
    label = 0

    solver = sytorch.GurobiSolver()
    dnn = nn.Sequential(
        nn.Linear(2, 2), nn.ReLU(),
        nn.Linear(2, 2), nn.ReLU(),
        nn.Linear(2, 2),
    ).to(dtype).to(solver)
    ddnn = dnn.decouple().repair()
    ddnn[0].val.requires_symbolic_()

    x = sytorch.ones(1,2).to(dtype)

    """ Solves it using the local default solver. """
    assert solver.solve(ddnn(x).argmax(axis) == label)

    """ Check whether the encoded argmax is correct. """
    assert ddnn(x).evaluate().argmax(axis) == label

    """ Patch the network. """
    ddnn.update_()

    """ Temporarily disable symbolic execution. Now `ddnn(x)` returns a
    concrete Tensor instead of a SymbolicArray. One can also use
    `ddnn.repair(False)` to disable the symbolic execution of ddnn.
    """
    with sytorch.no_symbolic():

        """ check whether the actual argmax is correct """
        assert ddnn(x).argmax(axis) == label

def test_dnn_repair():
    set_seed(1)
    dtype = sytorch.float64
    axis = 1
    label = 0

    solver = sytorch.GurobiSolver().verbose_()
    with ConstrHandler(solver.add_constraints):
        dnn = nn.Sequential(
            nn.Linear(2, 2), nn.ReLU(),
            nn.Linear(2, 2), nn.ReLU(),
            nn.Linear(2, 2),
        ).to(dtype).to(solver).repair()
        dnn[0].requires_symbolic_()

        x = sytorch.ones(1,2).to(dtype)
        ap = dnn.activation_pattern(x)

        """ Solves it using the local default solver. """
        sy = dnn(x, ap)
        assert solver.solve(sy.argmax(axis) == label)

    """ Check whether the encoded argmax is correct. """
    assert sy.evaluate().argmax(axis) == label

    """ Patch the network. """
    dnn.update_()

    """ Temporarily disable symbolic execution. Now `dnn(x)` returns a
    concrete Tensor instead of a SymbolicArray. One can also use
    `dnn.repair(False)` to disable the symbolic execution of ddnn.
    """
    with sytorch.no_symbolic():

        """ check whether the actual argmax is correct """
        assert dnn(x).argmax(axis) == label

def test_infeasible_dnn_repair():
    dtype = sytorch.float64
    solver = sytorch.GurobiSolver().verbose_()
    with ConstrHandler(solver.add_constraints):
        dnn = nn.Sequential(
            nn.Linear(2, 2),
            nn.ReLU(),
            nn.Linear(2, 2),
        ).to(dtype).to(solver).repair()

        # Test infeasible repair
        with torch.no_grad():
            dnn[0].weight[:] = torch.eye(2)
            dnn[0].bias[:] = torch.zeros(2)
            dnn[2].weight[:] = torch.Tensor([[1., 0.], [1., 0.]])
            dnn[2].bias[:] = torch.zeros(2)
        dnn[0].requires_symbolic_()

        x = torch.Tensor([[1.0, 0.5]]).to(dtype)
        ap = dnn.activation_pattern(x)
        sy = dnn(x, ap)
        assert not solver.solve(sy.argmax(axis=1) == 1)

def test_paper_examples():
    # Initialize example DNN
    dtype = sytorch.float64
    dnn = nn.Sequential(
        nn.Linear(1, 2),
        nn.ReLU(),
        nn.Linear(2, 2),
        nn.ReLU(),
        nn.Linear(2, 2),
        nn.ReLU(),
        nn.Linear(2, 2),
        nn.ReLU(),
        nn.Linear(2, 1)
    )

    with torch.no_grad():
        dnn[0].weight[:] = torch.Tensor([[1.], [-1.]])
        dnn[0].bias[:]   = torch.Tensor([[1., 0.]])
        dnn[2].weight[:] = torch.Tensor([[-1., 1.], [1., -1.]])
        dnn[2].bias[:]   = torch.Tensor([[-1., 1.]])
        dnn[4].weight[:] = torch.Tensor([[1., 1.], [-1., 1.]])
        dnn[4].bias[:]   = torch.Tensor([[0., 1.]])
        dnn[6].weight[:] = torch.Tensor([[-1., 1.], [1., -1.]])
        dnn[6].bias[:]   = torch.Tensor([[2., 0.]])
        dnn[8].weight[:] = torch.Tensor([[1., 1.]])
        dnn[8].bias[:]   = torch.Tensor([[1.]])

    dnn = dnn.to(dtype)
    x1 = torch.Tensor([-2.]).to(dtype)
    x2 = torch.Tensor([ 1.]).to(dtype)
    assert(dnn(x1) == 3.)
    assert(dnn(x2) == 4.)

    # Test single-layer repair example
    solver = sytorch.GurobiSolver().verbose_()
    with ConstrHandler(solver.add_constraints):
        dnn1 = dnn.deepcopy().to(solver).repair()
        dnn1[6].requires_symbolic_()

        sy1 = dnn1(x1[None,...], dnn1.activation_pattern(x1[None,...]))
        sy2 = dnn1(x2[None,...], dnn1.activation_pattern(x2[None,...]))
        solver.add_constraints(sy1 == 4.)
        solver.add_constraints(sy2 == 9.)

        assert solver.solve()
        dnn1.update_()
        with sytorch.no_symbolic():
            assert(dnn1(x1) == 4.)
            assert(dnn1(x2) == 9.)

    # Test single-layer repair example with PRDNN
    assert(dnn(x1) == 3.)
    assert(dnn(x2) == 4.)
    solver = sytorch.GurobiSolver().verbose_()
    dnn1_prdnn = dnn.deepcopy().decouple().to(solver).repair()
    dnn1_prdnn[6].val.requires_symbolic_()
    solver.add_constraints(dnn1_prdnn(x1[None,...]) == 4.)
    solver.add_constraints(dnn1_prdnn(x2[None,...]) == 9.)
    assert solver.solve()
    dnn1_prdnn.update_()
    with sytorch.no_symbolic():
        assert(dnn1_prdnn(x1) == 4.)
        assert(dnn1_prdnn(x2) == 9.)

    # Test single-layer plus biases repair example
    assert(dnn(x1) == 3.)
    assert(dnn(x2) == 4.)
    solver = sytorch.GurobiSolver().verbose_()
    with ConstrHandler(solver.add_constraints):
        dnn2 = dnn.deepcopy().to(solver).repair()
        dnn2[4].requires_symbolic_()
        dnn2[6].bias.requires_symbolic_()
        dnn2[8].bias.requires_symbolic_()

        sy1 = dnn2(x1[None,...], dnn2.activation_pattern(x1[None,...]))
        sy2 = dnn2(x2[None,...], dnn2.activation_pattern(x2[None,...]))
        solver.add_constraints(sy1 == 4.)
        solver.add_constraints(sy2 == 9.)

        assert solver.solve()
        dnn2.update_()
        with sytorch.no_symbolic():
            assert(dnn2(x1) == 4.)
            assert(dnn2(x2) == 9.)

    # Test consecutive-layer repair example
    assert(dnn(x1) == 3.)
    assert(dnn(x2) == 4.)
    dnn3 = dnn.deepcopy()
    with torch.no_grad():
        dnn3[4].weight[:] = torch.Tensor([[1.5 , 1.], [-1., 1.]])
        dnn3[6].weight[:] = torch.Tensor([[-1., 2.], [1., -1.]])
        dnn3[8].bias[:] = torch.Tensor([[2.]])

    assert(dnn3(x1) == 4.)
    assert(dnn3(x2) == 9.)

    # Test multi-layer repair example
    assert(dnn(x1) == 3.)
    assert(dnn(x2) == 4.)
    dnn4 = dnn.deepcopy()
    with torch.no_grad():
        dnn4[2].weight[:] = torch.Tensor([[-1., 2.], [1., -1.]])
        dnn4[4].weight[:] = torch.Tensor([[1., -1.], [-1., 1.]])
        dnn4[8].weight[:] = torch.Tensor([[4./3., 1.]])

    assert(torch.allclose(dnn4(x1), torch.Tensor([4.]).to(dtype)))
    assert(torch.allclose(dnn4(x2), torch.Tensor([9.]).to(dtype)))


if IN_BAZEL:
    main(__name__, __file__)
