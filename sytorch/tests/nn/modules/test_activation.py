# pylint: disable=import-error

import sytorch as st
from sytorch import nn
from sytorch.pervasives import *
from sytorch.solver.symbolic_array import SymbolicArray

dtype = torch.float64
device = torch.device('cpu')

def test_relu():
    for Solver in (st.GurobiSolver, st.LightningSolver):
        st.set_all_seed(0)
        with torch.no_grad():
            solver = Solver()
            dnn = nn.ReLU().to(solver,device,dtype).symbolic_()
            input = st.randn(2, 3).to(device,dtype)
            symbolic_input = solver.reals(input.shape)

            """ Test concrete execution. """
            with st.no_symbolic():
                output = dnn(input)
                assert (output == torch.nn.functional.relu(input)).all()

            """ Test symbolic input. """
            pattern = dnn.activation_pattern(input)
            symbolic_output = dnn(symbolic_input, pattern=pattern)
            assert isinstance(symbolic_output, SymbolicArray)
            assert np.vectorize(lambda a, b: a is b)(
                symbolic_output[pattern], symbolic_input[pattern]).view(ndarray).all()
            assert (symbolic_output[~pattern].view(ndarray) == 0.).all()
            assert solver.solve(symbolic_input == input)
            assert torch.allclose(symbolic_output.evaluate().to(device,dtype), output)

def test_linear_relu_mask():
    for Solver in (st.LightningSolver,):
        st.set_all_seed(0)
        with torch.no_grad():
            solver = Solver()
            dnn = nn.Sequential(
                nn.Linear(3, 4),
                nn.ReLU(),
            ).to(solver,device,dtype).symbolic_()
            dnn[0].requires_symbolic_(row_mask=[1,3])

            input = st.randn(2, 3).to(device,dtype)
            symbolic_input = solver.reals(input.shape)

            """ Test concrete execution. """
            with st.no_symbolic():
                output = dnn(input)

            """ Test symbolic input. """
            pattern = dnn.activation_pattern(input)
            symbolic_output = dnn(input, pattern=pattern)
            assert isinstance(symbolic_output, SymbolicArray)
            assert solver.solve(
                dnn.parameter_deltas(concat=True) == 0.,
                symbolic_input == input
            )
            assert torch.allclose(symbolic_output.evaluate().to(device,dtype), output)

def test_conv2d_relu_mask():
    for Solver in (st.LightningSolver,):
        st.set_all_seed(0)
        with torch.no_grad():
            solver = Solver()
            dnn = nn.Sequential(
                nn.Conv2d(3, 4, 3),
                nn.ReLU(),
            ).to(solver,device,dtype).symbolic_()
            dnn[0].requires_symbolic_(row_mask=[1,3])

            input = st.randn(2, 3, 7, 7).to(device,dtype)
            symbolic_input = solver.reals(input.shape)

            """ Test concrete execution. """
            with st.no_symbolic():
                output = dnn(input)

            """ Test symbolic input. """
            pattern = dnn.activation_pattern(input)
            symbolic_output = dnn(input, pattern=pattern)
            assert isinstance(symbolic_output, SymbolicArray)
            assert solver.solve(
                dnn.parameter_deltas(concat=True) == 0.,
                symbolic_input == input
            )
            assert torch.allclose(symbolic_output.evaluate().to(device,dtype), output)

try:
    from external.bazel_python.pytest_helper import main
    main(__name__, __file__)
except ImportError:
    pass
