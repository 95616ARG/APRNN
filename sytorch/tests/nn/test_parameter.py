# pylint: disable=import-error
try:
    from external.bazel_python.pytest_helper import main
    IN_BAZEL = True
except ImportError:
    IN_BAZEL = False

import numpy as np
import sytorch
import sytorch.nn as nn
import itertools
import copy

def test_new_parameter_default():
    for data, requires_grad in itertools.product(
        # """ data """
        (
            sytorch.randn((3, 4)),                 # from `sytorch.Tensor`
            nn.Parameter(sytorch.randn(3, 4)),     # from `nn.Parameter`
            nn.Parameter(sytorch.randn((3, 4))),   # from `nn.Parameter`
        ),

        # """ requires_grad """
        (True, False),
    ):
        data.requires_grad_(requires_grad)
        param = nn.Parameter(data)
        assert param.data_ptr()        == data.data_ptr()
        assert param.requires_grad     == data.requires_grad
        assert param.requires_symbolic == False
        assert param.solver            == None

def test_new_parameter():
    for data, requires_grad, requires_symbolic, solver in itertools.product(
        # """ data """
        (
            sytorch.randn((3, 4)),                 # from `sytorch.Tensor`
            nn.Parameter(sytorch.randn(3, 4)),     # from `nn.Parameter`
            nn.Parameter(sytorch.randn((3, 4))),   # from `nn.Parameter`
        ),

        # """ requires_grad """
        (True, False),

        # """ required_symbolic """
        (True, False),

        # """ solver """
        (sytorch.GurobiSolver(),),
    ):
        data.requires_grad_(requires_grad)

        """ default """
        param = nn.Parameter(data)
        assert param.data_ptr()        == data.data_ptr()
        assert param.requires_grad     == data.requires_grad
        assert param.requires_symbolic == False
        assert param.solver            == None

        """ with kwargs """
        param = nn.Parameter(
            data=data,
            requires_symbolic=requires_symbolic,
            solver=solver
        )
        assert param.data_ptr()        == data.data_ptr()
        assert param.requires_grad     == data.requires_grad
        assert param.requires_symbolic == requires_symbolic
        assert param.solver            == solver

        """ default with-in scope """
        with sytorch.GurobiSolver() as solver_in_context:
            param = nn.Parameter(data)
            assert param.data_ptr()        == data.data_ptr()
            assert param.requires_grad     == data.requires_grad
            assert param.requires_symbolic == False
            assert param.solver            == None

            param = nn.Parameter(data, requires_symbolic=True)
            assert param.data_ptr()        == data.data_ptr()
            assert param.requires_grad     == data.requires_grad
            assert param.requires_symbolic == True
            assert param.solver            == solver_in_context

            """ with kwargs """
            param = nn.Parameter(
                data=data,
                requires_symbolic=requires_symbolic,
                solver=solver
            )
            assert param.data_ptr()        == data.data_ptr()
            assert param.requires_grad     == data.requires_grad
            assert param.requires_symbolic == requires_symbolic
            assert param.solver            == solver

def test_to():
    pass

def test_update():
    pass

def test_deepcopy():
    for requires_grad, requires_symbolic, solver in itertools.product(
        # """ requires_grad """
        (True, False),

        # """ required_symbolic """
        (True, False),

        # """ solver """
        # TODO(anonymous): cases with solver=None
        (sytorch.GurobiSolver(),),
    ):
        param_src = nn.Parameter(sytorch.randn((3, 4)))\
            .requires_grad_(requires_grad)\
            .to(solver=solver)\
            .requires_symbolic_(requires_symbolic)

        param_dst = copy.deepcopy(param_src)

        """ The deepcopy retains the original's data's value. """
        assert sytorch.all(param_src == param_dst)

        """ The deepcopy's data does not reference to the original's data. """
        assert param_src.data_ptr()        != param_dst.data_ptr()

        """ The deepcopy retains the original's `.requires_grad` property. """
        assert param_src.requires_grad     == param_dst.requires_grad

        """ The deepcopy retains the original's `.requires_symbolic` property. """
        assert param_src.requires_symbolic == param_dst.requires_symbolic

        """ The deepcopy is on the same solver as the original. """
        assert param_src.solver            == param_dst.solver

        """ The deepcopy does not share the same variables with the original. """
        if requires_symbolic:
            assert all(map(lambda x: x[0] is not x[1], zip(param_src.delta.flat, param_dst.delta.flat)))

if IN_BAZEL:
    main(__name__, __file__)
