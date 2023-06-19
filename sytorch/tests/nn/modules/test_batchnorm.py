# pylint: disable=import-error

import sytorch as st
from sytorch import nn
from sytorch.pervasives import *
from sytorch.solver.symbolic_array import SymbolicArray

dtype = torch.float64
device = torch.device('cpu')

def _get_kwargs(**kwargs):
    return kwargs

batch_norm2d_specs = (
    _get_kwargs(
        num_features = 3,
        eps = 1e-5,
        momentum = 0.1,
        affine = True,
        track_running_stats=True,
    ),
    _get_kwargs(
        num_features = 3,
        eps = 1e-4,
        momentum = 0.1,
        affine = False,
        track_running_stats=True,
    ),
)

seeds = range(2)

def test_batch_norm2d():
    for Solver, spec, seed in itertools.product(
        (st.GurobiSolver,),
        batch_norm2d_specs,
        seeds,
    ):
        st.set_all_seed(seed)
        with torch.no_grad():
            solver = Solver()
            dnn = nn.BatchNorm2d(**spec).to(solver,device,dtype).symbolic_()
            dnn.running_mean[:] = st.randn((dnn.num_features,)).to(device,dtype)
            dnn.running_var[:] = st.rand((dnn.num_features,)).to(device,dtype)
            if dnn.affine:
                dnn.weight[:] = st.randn((dnn.num_features,)).to(device,dtype)
                dnn.bias[:] = st.randn((dnn.num_features,)).to(device,dtype)

            input = st.randn(2, 3, 9, 8).to(device,dtype)
            symbolic_input = solver.reals(input.shape)

            """ Test concrete execution. """
            with st.no_symbolic():
                output = dnn(input)

            """ Test symbolic input + concrete parameters. """
            symbolic_output = dnn(symbolic_input)
            assert isinstance(symbolic_output, SymbolicArray)
            assert solver.solve(symbolic_input == input)
            assert torch.allclose(symbolic_output.evaluate().to(device,dtype), output)

            if dnn.affine:
                """ Test symbolic input + symbolic bias. """
                dnn.bias.requires_symbolic_()
                symbolic_output = dnn(symbolic_input)
                assert isinstance(symbolic_output, SymbolicArray)
                assert solver.solve(
                    symbolic_input == input,
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

def test_batch_norm2d_repair():
    for Solver, spec, seed in itertools.product(
        (st.GurobiSolver,),
        batch_norm2d_specs,
        seeds,
    ):
        st.set_all_seed(seed)
        with torch.no_grad():
            solver = Solver()
            dnn = nn.BatchNorm2d(**spec).to(solver,device,dtype).symbolic_()
            dnn.running_mean[:] = st.randn((dnn.num_features,)).to(device,dtype)
            dnn.running_var[:] = st.rand((dnn.num_features,)).to(device,dtype)
            if dnn.affine:
                dnn.weight[:] = st.randn((dnn.num_features,)).to(device,dtype)
                dnn.bias[:] = st.randn((dnn.num_features,)).to(device,dtype)

            input = st.randn(2, 3, 9, 8).to(device,dtype)
            symbolic_input = solver.reals(input.shape)

            """ Test concrete execution. """
            with st.no_symbolic():
                output = dnn(input)

            """ Test symbolic input + concrete parameters. """
            symbolic_output = dnn(symbolic_input)
            assert isinstance(symbolic_output, SymbolicArray)
            assert solver.solve(symbolic_input == input)
            assert torch.allclose(symbolic_output.evaluate().to(device,dtype), output)

            if dnn.affine:
                """ Test symbolic input + symbolic bias. """
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

try:
    from external.bazel_python.pytest_helper import main
    main(__name__, __file__)
except ImportError:
    pass
