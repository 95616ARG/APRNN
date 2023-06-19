# pylint: disable=import-error

import sytorch as st
from sytorch import nn
from sytorch.pervasives import *
from sytorch.solver.lightning import LightningVar
from sytorch.solver.symbolic_array import SymbolicArray

dtype = torch.float64
device = torch.device('cpu')

def _get_kwargs(**kwargs):
    return kwargs

linear_specs = (
    _get_kwargs(
        in_features  = 3,
        out_features = 4,
        bias = True,
    ),
    _get_kwargs(
        in_features  = 3,
        out_features = 4,
        bias = False,
    ),
)
# TODO(anonymous): bias=False case.
def test_linear():
    for Solver, spec in itertools.product(
        (st.GurobiSolver, st.LightningSolver),
        linear_specs,
    ):
        st.set_all_seed(0)
        with torch.no_grad():
            solver = Solver()
            dnn = nn.Linear(**spec).to(solver,device,dtype).symbolic_()
            input = st.randn(2, 3).to(device,dtype)
            symbolic_input = solver.reals(input.shape)

            """ Test concrete execution. """
            with st.no_symbolic():
                output = dnn(input)
                assert (output == torch.nn.functional.linear(input, dnn.weight, dnn.bias)).all()

            """ Test symbolic input + concrete parameters. """
            symbolic_output = dnn(symbolic_input)
            assert isinstance(symbolic_output, SymbolicArray)
            assert solver.solve(symbolic_input == input)
            assert torch.allclose(symbolic_output.evaluate().to(device,dtype), output)

            """ Test symbolic input + symbolic bias. """
            if dnn.bias is not None:
                dnn.bias.requires_symbolic_()
                symbolic_output = dnn(symbolic_input)
                assert isinstance(symbolic_output, SymbolicArray)
                assert solver.solve(
                    dnn.bias.symbolic() == dnn.bias
                )
                assert torch.allclose(symbolic_output.evaluate().to(device,dtype), output)

            """ Test concrete input + symbolic parameters. """
            dnn.weight.requires_symbolic_()
            symbolic_output = dnn(input)
            assert isinstance(symbolic_output, SymbolicArray)
            assert solver.solve(
                dnn.weight.symbolic() == dnn.weight,
            )
            assert torch.allclose(symbolic_output.evaluate().to(device,dtype), output)

def test_linear_repair():
    for Solver, spec in itertools.product(
        (st.GurobiSolver, st.LightningSolver),
        linear_specs,
    ):
        st.set_all_seed(0)
        with torch.no_grad():
            solver = Solver()
            dnn = nn.Linear(**spec).to(solver,device,dtype).symbolic_()
            input = st.randn(2, 3).to(device,dtype)
            symbolic_input = solver.reals(input.shape)

            """ Test concrete execution. """
            with st.no_symbolic():
                output = dnn(input)
                assert (output == torch.nn.functional.linear(input, dnn.weight, dnn.bias)).all()

            """ Test symbolic input + concrete parameters. """
            symbolic_output = dnn(symbolic_input)
            assert isinstance(symbolic_output, SymbolicArray)
            assert solver.solve(symbolic_input == input)
            assert torch.allclose(symbolic_output.evaluate().to(device,dtype), output)

            """ Test symbolic input + symbolic bias. """
            if dnn.bias is not None:
                dnn.bias.requires_symbolic_()
                symbolic_output = dnn(symbolic_input)
                assert isinstance(symbolic_output, SymbolicArray)
                assert solver.solve(
                    dnn.bias.symbolic() == st.randn(*dnn.bias.shape)
                )
                repaired_output = dnn.deepcopy().update_(src=dnn).repair(False)(input)
                assert torch.allclose(symbolic_output.evaluate().to(device,dtype), repaired_output)

            """ Test concrete input + symbolic parameters. """
            dnn.weight.requires_symbolic_()
            symbolic_output = dnn(input)
            assert isinstance(symbolic_output, SymbolicArray)
            assert solver.solve(
                dnn.weight.symbolic() == st.randn(*dnn.weight.shape),
            )
            repaired_output = dnn.deepcopy().update_(src=dnn).repair(False)(input)
            assert torch.allclose(symbolic_output.evaluate().to(device,dtype), repaired_output)

def test_linear_row_mask():
    for Solver, spec in itertools.product(
        (st.LightningSolver, ),
        linear_specs,
    ):
        st.set_all_seed(0)
        with torch.no_grad():
            solver = Solver()
            dnn = nn.Linear(**spec).to(solver,device,dtype).symbolic_()
            input = st.randn(2, 3).to(device,dtype)
            symbolic_input = solver.reals(input.shape)

            row_mask = [1, 3]

            """ Test concrete execution. """
            with st.no_symbolic():
                output = dnn(input)
                assert (output == torch.nn.functional.linear(input, dnn.weight, dnn.bias)).all()

            """ Test symbolic input + concrete parameters. """
            symbolic_output = dnn(symbolic_input)
            assert isinstance(symbolic_output, SymbolicArray)
            assert solver.solve(symbolic_input == input)
            assert torch.allclose(symbolic_output.evaluate().to(device,dtype), output)

            """ Test concrete input + symbolic parameters. """
            dnn.requires_symbolic_(row_mask=row_mask)
            assert (dnn.row_mask == [False, True, False, True]).all()
            symbolic_output = dnn(input)
            assert all(map(lambda v: isinstance(v, LightningVar), symbolic_output[...,dnn.row_mask].flat))
            assert (symbolic_output[...,~dnn.row_mask].view(ndarray) == output[...,~dnn.row_mask].cpu().numpy()).all()
            assert isinstance(symbolic_output, SymbolicArray)
            assert solver.solve(
                dnn.bias.symbolic() == dnn.bias if dnn.bias is not None else (),
                dnn.weight.symbolic() == dnn.weight,
            )
            assert torch.allclose(symbolic_output.evaluate().to(device,dtype), output)

def test_identity():
    for Solver in (st.GurobiSolver, st.LightningSolver):
        st.set_all_seed(0)
        with torch.no_grad():
            solver = Solver()
            dnn = nn.Identity().to(solver,device,dtype).requires_symbolic_().symbolic_()
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

try:
    from external.bazel_python.pytest_helper import main
    main(__name__, __file__)
except ImportError:
    pass
