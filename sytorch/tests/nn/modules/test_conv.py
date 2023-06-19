# pylint: disable=import-error

import sytorch as st
from sytorch import nn
from sytorch.pervasives import *
from sytorch.solver.lightning import LightningVar
from sytorch.solver.symbolic_array import SymbolicArray
import gurobipy as gp

dtype = torch.float64
npdtype = np.float64
device = torch.device('cpu')

def _get_kwargs(**kwargs):
    return kwargs

conv2d_specs = (
    _get_kwargs(
        in_channels  = 3,
        out_channels = 4,
        kernel_size  = (3, 3),
        stride = (1, 1),
        padding = (0, 0),
        dilation = (1, 1),
        groups = 1,
        padding_mode = 'zeros',
        bias = False,
    ),
    _get_kwargs(
        in_channels  = 3,
        out_channels = 4,
        kernel_size  = (3, 4),
        stride = (1, 2),
        padding = (2, 1),
        dilation = (1, 1),
        groups = 1,
        padding_mode = 'zeros',
        bias = True,
    ),
    _get_kwargs(
        in_channels  = 3,
        out_channels = 4,
        kernel_size  = (3, 4),
        stride = (1, 2),
        padding = (2, 1),
        dilation = (1, 1),
        groups = 1,
        padding_mode = 'replicate',
        bias = True,
    ),
    _get_kwargs(
        in_channels  = 3,
        out_channels = 4,
        kernel_size  = (3, 4),
        stride = (1, 2),
        padding = (2, 1),
        dilation = (1, 1),
        groups = 1,
        padding_mode = 'reflect',
        bias = True,
    ),
    _get_kwargs(
        in_channels  = 3,
        out_channels = 4,
        kernel_size  = (3, 4),
        stride = (1, 2),
        padding = (2, 1),
        dilation = (1, 1),
        groups = 1,
        padding_mode = 'circular',
        bias = True,
    ),
)

def test_conv2d():
    # NOTE(anonymous): GurobiSolver version is no loner maintained.
    for Solver, conv2d_spec in itertools.product(
        (st.GurobiSolver, st.LightningSolver,),
        conv2d_specs
    ):
        st.set_all_seed(0)
        with torch.no_grad():
            solver = Solver()
            dnn = nn.Conv2d(**conv2d_spec).to(solver,device,dtype).symbolic_()
            input = st.randn(2, 3, 12, 11).to(device,dtype)
            symbolic_input = solver.reals(input.shape)

            with st.no_symbolic():
                output = dnn(input)

            """ Test symbolic input + concrete parameters. """
            symbolic_output = dnn(symbolic_input)
            assert isinstance(symbolic_output, SymbolicArray)
            assert solver.solve(symbolic_input == input)
            assert torch.allclose(symbolic_output.evaluate().to(device,dtype), output)

            if dnn.bias is not None:
                """ Test symbolic input + symbolic bias. """
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

def test_conv2d_repair():
    for Solver, conv2d_spec in itertools.product(
        (st.LightningSolver, st.GurobiSolver),
        conv2d_specs
    ):
        st.set_all_seed(0)
        with torch.no_grad():
            solver = Solver()
            dnn = nn.Conv2d(**conv2d_spec).to(solver,device,dtype).symbolic_()
            input = st.randn(2, 3, 12, 11).to(device,dtype)
            symbolic_input = solver.reals(input.shape)

            with st.no_symbolic():
                output = dnn(input)

            """ Test symbolic input + concrete parameters. """
            symbolic_output = dnn(symbolic_input)
            assert isinstance(symbolic_output, SymbolicArray)
            assert solver.solve(symbolic_input == input)
            assert torch.allclose(symbolic_output.evaluate().to(device,dtype), output)

            if dnn.bias is not None:
                """ Test symbolic input + symbolic bias. """
                dnn.bias.requires_symbolic_()
                symbolic_output = dnn(symbolic_input)
                assert isinstance(symbolic_output, SymbolicArray)
                assert solver.solve(
                    dnn.bias.symbolic() == st.randn(*dnn.bias.shape),
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

def test_conv2d_row_mask():
    # NOTE(anonymous): GurobiSolver version is no loner maintained.
    for Solver, conv2d_spec in itertools.product(
        (st.LightningSolver, st.GurobiSolver),
        conv2d_specs
    ):
        st.set_all_seed(0)
        with torch.no_grad():
            solver = Solver()
            dnn = nn.Conv2d(**conv2d_spec).to(solver,device,dtype).symbolic_()
            input = st.randn(2, 3, 12, 11).to(device,dtype)
            symbolic_input = solver.reals(input.shape)

            row_mask = [1, 3]

            with st.no_symbolic():
                output = dnn(input)

            """ Test symbolic input + concrete parameters. """
            symbolic_output = dnn(symbolic_input)
            assert isinstance(symbolic_output, SymbolicArray)
            assert solver.solve(symbolic_input == input)
            assert torch.allclose(symbolic_output.evaluate().to(device,dtype), output)

            """ Test concrete input + symbolic parameters. """
            dnn.requires_symbolic_(row_mask=row_mask)
            assert (dnn.row_mask == [False, True, False, True]).all()
            symbolic_output = dnn(input)
            assert all(map(lambda v: isinstance(v, (LightningVar, gp.Var)), symbolic_output[...,dnn.row_mask,:,:].flat))
            assert np.allclose(symbolic_output[...,~dnn.row_mask,:,:].view(ndarray).astype(npdtype), output[...,~dnn.row_mask,:,:].cpu().numpy())
            assert isinstance(symbolic_output, SymbolicArray)
            assert solver.solve(
                dnn.bias.symbolic() == dnn.bias if dnn.bias is not None else (),
                dnn.weight.symbolic() == dnn.weight,
            )
            assert torch.allclose(symbolic_output.evaluate().to(device,dtype), output)

try:
    from external.bazel_python.pytest_helper import main
    main(__name__, __file__)
except ImportError:
    pass
