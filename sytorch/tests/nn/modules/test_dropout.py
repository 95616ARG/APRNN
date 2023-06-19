# pylint: disable=import-error

import sytorch as st
from sytorch import nn
from sytorch.pervasives import *
from sytorch.solver.symbolic_array import SymbolicArray

dtype = torch.float64
device = torch.device('cpu')

def test_dropout():
    for Solver in (st.GurobiSolver, st.LightningSolver):
        st.set_all_seed(0)
        with torch.no_grad():
            solver = Solver()
            dnn = nn.Dropout().to(solver,device,dtype).requires_symbolic_().symbolic_()
            input = st.randn(2, 3).to(device,dtype)
            symbolic_input = solver.reals(input.shape)

            """ Test concrete execution. """
            with st.no_symbolic():
                output = dnn(input)
                assert (output == input).all()

            """ Test symbolic execution. """
            symbolic_output = dnn(symbolic_input)
            assert isinstance(symbolic_output, SymbolicArray)
            assert solver.solve(symbolic_input == input)
            assert torch.allclose(symbolic_output.evaluate().to(device,dtype), output)

def test_dropout_training():
    for Solver in (st.GurobiSolver, st.LightningSolver):
        st.set_all_seed(0)
        with torch.no_grad():
            solver = Solver()
            dnn = nn.Dropout().to(solver,device,dtype).requires_symbolic_().symbolic_().train()
            input = st.randn(2, 3).to(device,dtype)
            symbolic_input = solver.reals(input.shape)

            """ Test symbolic execution raises a NotImplementedErrer in training mode. """
            try:
                symbolic_output = dnn(symbolic_input)
                assert False
            except NotImplementedError:
                pass

try:
    from external.bazel_python.pytest_helper import main
    main(__name__, __file__)
except ImportError:
    pass
