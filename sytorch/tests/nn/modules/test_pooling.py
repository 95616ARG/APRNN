# pylint: disable=import-error

import sytorch as st
from sytorch import nn
from sytorch.pervasives import *
from sytorch.solver.symbolic_array import SymbolicArray

dtype = torch.float64
device = torch.device('cpu')

def _get_kwargs(**kwargs):
    return kwargs

adaptive_average_pool2d_specs = (
    _get_kwargs(output_size = (1, 1)),
    _get_kwargs(output_size = (3, 4)),
)

def test_adaptive_average_pool2d():
    # NOTE(anonymous): GurobiSolver version is no loner maintained.
    for Solver, spec in itertools.product(
        (st.GurobiSolver, st.LightningSolver,),
        adaptive_average_pool2d_specs
    ):
        st.set_all_seed(0)
        with torch.no_grad():
            solver = Solver()
            dnn = nn.AdaptiveAvgPool2d(**spec).to(solver,device,dtype).symbolic_()
            input = st.randn(2, 3, 12, 11).to(device,dtype)
            symbolic_input = solver.reals(input.shape)

            with st.no_symbolic():
                output = dnn(input)

            """ Test concrete execution. """
            with st.no_symbolic():
                output = dnn(input)
                assert (output == torch.nn.functional.adaptive_avg_pool2d(input, dnn.output_size)).all()

            """ Test symbolic input. """
            pattern = dnn.activation_pattern(input)
            symbolic_output = dnn(symbolic_input, pattern=pattern)
            assert isinstance(symbolic_output, SymbolicArray)
            assert solver.solve(symbolic_input == input)
            assert torch.allclose(symbolic_output.evaluate().to(device,dtype), output)

max_pool2d_specs = (
    _get_kwargs(
        kernel_size  = (3, 3),
        stride = (1, 1),
        padding = (0, 0),
        dilation = (1, 1),
        return_indices = False,
        ceil_mode = False,
    ),
    _get_kwargs(
        kernel_size  = (3, 4),
        stride = (1, 2),
        padding = (1, 2),
        dilation = (1, 1),
        return_indices = False,
        ceil_mode = False,
    ),
    _get_kwargs(
        kernel_size  = (3, 4),
        stride = (1, 2),
        padding = (1, 2),
        dilation = (1, 1),
        return_indices = False,
        ceil_mode = True,
    ),
)
def test_max_pool2d():
    # NOTE(anonymous): GurobiSolver version is no loner maintained.
    for Solver, spec in itertools.product(
        (st.LightningSolver,),
        max_pool2d_specs
    ):
        st.set_all_seed(0)
        with torch.no_grad():
            solver = Solver()
            dnn = nn.MaxPool2d(**spec).to(solver,device,dtype).symbolic_()
            input = st.randn(2, 3, 12, 11).to(device,dtype)
            symbolic_input = solver.reals(input.shape)

            with st.no_symbolic():
                output = dnn(input)

            """ Test concrete execution. """
            with st.no_symbolic():
                output = dnn(input)
                assert (output == torch.nn.functional.max_pool2d(
                                    input, dnn.kernel_size, dnn.stride,
                                    dnn.padding, dnn.dilation, dnn.ceil_mode,
                                    dnn.return_indices)).all()

            """ Test symbolic input. """
            pattern = dnn.activation_pattern(input)
            symbolic_output = dnn(symbolic_input, pattern=pattern)
            assert isinstance(symbolic_output, SymbolicArray)
            assert solver.solve(symbolic_input == input)
            assert torch.allclose(symbolic_output.evaluate().to(device,dtype), output)

def test_max_pool2d_negative_infinity_padding():
    # NOTE(anonymous): GurobiSolver version is no loner maintained.
    for Solver, spec in itertools.product(
        (st.LightningSolver,),
        max_pool2d_specs
    ):
        st.set_all_seed(0)
        with torch.no_grad():
            solver = Solver()
            dnn = nn.MaxPool2d(**spec).to(solver,device,dtype).symbolic_()
            input = st.randn(2, 3, 12, 11).to(device,dtype) - 1e10
            assert (input < 0.).all()
            symbolic_input = solver.reals(input.shape)

            with st.no_symbolic():
                output = dnn(input)

            """ Test concrete execution. """
            with st.no_symbolic():
                output = dnn(input)
                assert (output == torch.nn.functional.max_pool2d(
                                    input, dnn.kernel_size, dnn.stride,
                                    dnn.padding, dnn.dilation, dnn.ceil_mode,
                                    dnn.return_indices)).all()

            """ Test symbolic input. """
            pattern = dnn.activation_pattern(input)
            symbolic_output = dnn(symbolic_input, pattern=pattern)
            assert isinstance(symbolic_output, SymbolicArray)
            assert solver.solve(symbolic_input == input)
            assert torch.allclose(symbolic_output.evaluate().to(device,dtype), output)

try:
    from external.bazel_python.pytest_helper import main
    main(__name__, __file__)
except ImportError:
    pass
