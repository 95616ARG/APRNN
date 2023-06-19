# pylint: disable=import-error

import sytorch as st
from sytorch import nn
from sytorch.pervasives import *
from sytorch.solver.symbolic_array import SymbolicArray

dtype = torch.float64
device = torch.device('cpu')

def test_sequential_slicing():
    with ProcessPoolExecutor() as e:
        solver = st.LightningSolver()
        net = nn.Sequential(
            nn.Linear(2, 2), nn.ReLU(),
            nn.Linear(2, 2), nn.ReLU(),
        ).to(solver).to(e)

        sliced = net[:2]

        assert net.executor == sliced.executor
        assert net.symbolic_mode == sliced.symbolic_mode

def test_parallel_slicing():
    with ProcessPoolExecutor() as e:
        solver = st.LightningSolver()
        net = nn.Parallel(
            nn.Linear(2, 2),
            nn.Linear(2, 2),
            nn.Linear(2, 2),
            nn.Linear(2, 2),
            mode = 'cat',
            dim = -1,
        ).to(solver).to(e)

        sliced = net[:2]

        assert net.executor == sliced.executor
        assert net.symbolic_mode == sliced.symbolic_mode
        assert net.mode == sliced.mode
        assert net.dim == sliced.dim

def create_sequential_1(in_channels, out_channels, solver):
    def f(dnn):
        dnn[2].requires_symbolic_()
        return dnn

    return nn.Sequential(
        nn.Linear(in_channels, 3),
        nn.ReLU(),
        nn.Linear(3, 3),
        nn.ReLU(),
        nn.Linear(3, out_channels),
    ).to(solver).symbolic_(), f

def create_sequential_2(in_channels, out_channels, solver):
    def f(dnn):
        dnn[2][1][1].requires_symbolic_()
        return dnn

    return nn.Sequential(
        nn.Linear(in_channels, 3),
        nn.ReLU(),
        nn.Sequential(
            nn.Linear(3, 3),
            nn.Sequential(
                nn.ReLU(),
                nn.Linear(3, 3),
            ),
            nn.ReLU(),
        ),
        nn.Sequential(
            nn.Linear(3, 3),
            nn.ReLU(),
            nn.Linear(3, 3),
        ),
        nn.Linear(3, 3),
        nn.ReLU(),
        nn.Linear(3, out_channels),
    ).to(solver).symbolic_(), f

def create_parallel_add(in_channels, out_channels, solver):
    def f(dnn):
        dnn[0].requires_symbolic_()
        dnn[1][0].requires_symbolic_()
        dnn[2].requires_symbolic_()
        return dnn

    return nn.Parallel(
        nn.Linear(in_channels, out_channels),
        nn.Sequential(
            nn.Linear(in_channels, 3),
            nn.Linear(3, 3),
            nn.ReLU(),
            nn.Linear(3, out_channels),
        ),
        nn.Linear(in_channels, out_channels),
        mode = 'add',
    ).to(solver).symbolic_(), f

def create_parallel_cat(in_channels, out_channels, solver):
    def f(dnn):
        for m in dnn:
            m.requires_symbolic_()
        return dnn

    return nn.Parallel(
        *(
            nn.Linear(in_channels, 1).to(solver)
            for _ in range(out_channels)
        ),
        mode = 'cat',
        dim = -1,
    ).to(solver).symbolic_(), f

def test_container():
    # NOTE(anonymous): GurobiSolver version is no loner maintained.
    for Solver, dnn_creator in itertools.product(
        (st.LightningSolver,),
        (
            create_sequential_1,
            create_sequential_2,
            create_parallel_add,
            create_parallel_cat,
        ),
    ):
        st.set_all_seed(0)
        with torch.no_grad():
            solver = Solver()
            dnn, make_symbolic = dnn_creator(3, 4, solver)
            dnn.to(device,dtype)
            input = st.randn(2, 3).to(device,dtype)
            symbolic_input = solver.reals(input.shape)

            with st.no_symbolic():
                output = dnn(input)

            """ Test symbolic input. """
            pattern = dnn.activation_pattern(input)
            symbolic_output = dnn(symbolic_input, pattern=pattern)
            assert isinstance(symbolic_output, SymbolicArray)
            assert solver.solve(symbolic_input == input)
            assert torch.allclose(symbolic_output.evaluate().to(device,dtype), output)

            """ Test symbolic parameters. """
            dnn = make_symbolic(dnn)
            pattern = dnn.activation_pattern(input)
            symbolic_output = dnn(input, pattern=pattern)
            assert isinstance(symbolic_output, SymbolicArray)
            assert solver.solve(dnn.parameter_deltas(concat=True) == 0.)
            assert torch.allclose(symbolic_output.evaluate().to(device,dtype), output)

try:
    from external.bazel_python.pytest_helper import main
    main(__name__, __file__)
except ImportError:
    pass
