import warnings; warnings.filterwarnings("ignore")

from experiments import mnist
from experiments.base import *
import sytorch as st
import argparse
import pandas as pd
from timeit import default_timer as timer

""" Extend beyond this experiment.
    ==============================

    1. Different buggy network.
       - Change `network` to load a different buggy network.
         Specifically, the following command loads a torch DNN `torch_dnn_object`:

         ```
         network = st.nn.from_torch(torch_dnn_object)
         ```

         And the following command loads an ONNX DNN `onnx_dnn_path` from file:

         ```
         network = st.nn.from_file(onnx_dnn_path)
         ```

    2. Different dataset.
       - Change the `images` and `labels` tensors to load expected buggy inputs
         and the correct labels, in the same way as training a PyTorch DNN.

    3. Different repair parameters:
       - Change the `k` parameters.
       - Change `lb` and `ub`.
"""

parser = argparse.ArgumentParser(description='')
parser.add_argument('--net', type=str, dest='net', action='store', required=True,
                    help='3x100, 9x100, 9x200 or path to .onnx file')
parser.add_argument('--num', type=int, dest='num', action='store', default=100,
                    help='number of points to repair')
parser.add_argument('--k', type=str, dest='k', action=ParseIndex, default=None,
                    help='repair parameter k')
parser.add_argument('--input_shape', dest='input_shape', action='store',
                    default=(784,),
                    help='repair parameter k')
parser.add_argument('--device', type=str, dest='device', action='store', default='cpu',
                    help='device to use, e.g., cuda, cuda:0, cpu. (default=cpu).')
parser.add_argument('--use_artifact', dest='use_artifact', action='store_true',
                    help='use authors\' repaired DNN.')
args = parser.parse_args()

device = get_device(args.device)
dtype = st.float64

if args.k is None:
    k = {
        '3x100': 2,
        '9x100': 10,
        '9x200': 12,
    }[args.net]
else:
    k = args.k

n_points = args.num

if args.net in ('3x100', '9x100', '9x200'):
    network = mnist.model(args.net).to(dtype=dtype, device=device)
else:
    network = st.nn.from_file(args.net)

corruption = 'fog'
GeneralizationSet = mnist.GeneralizationSet(corruption)[n_points:].reshape(*args.input_shape)
DrawdownSet = mnist.DrawdownSet().reshape(*args.input_shape)
RepairSet = mnist.RepairSet(corruption).reshape(*args.input_shape)

if not args.use_artifact:

    """ Load the buggy input points from the repair set. """
    images, labels = RepairSet.load(n_points)
    images = images.to(dtype=dtype, device=device)

    start = timer()

    """ Create a new solver. """
    solver = st.GurobiSolver()

    """ Attach the decoupled DNN to the solver.
        =============================
        - `.deepcopy()` returns a deepcopy of the DNN to repair. This is optional.
        - `.decouple()` decouples a given DNN into a DDNN.
        - `.to(solver)` attaches the DNN to the solver, just like how you attach a
        DNN to a device in PyTorch.
        - `.repair()` turns on the repair mode and enables symbolic execution. It
        works just like `.train()` in PyTorch, which turns on the training mode.
    """
    N = network.deepcopy().decouple().to(solver).repair()

    """ Specify the symbolic weights.
        ==============================
        `N[k].val.weight.requires_symbolic_(lb=-3., ub=3.)` makes the
        k-th layer weight of the value network symbolic.
    """
    N[k].val.weight.requires_symbolic_(lb=-3., ub=3.)

    """ Encode the symbolic output.
        ===========================
        `N(images)` symbolically forwards `images` through `N`.
    """
    symbolic_output = N(images)

    """ Calculate the original output for minimization.
        ==============================================
        - The `st.no_symbolic()` context turns off the symbolic execution.
    """
    with st.no_grad(), st.no_symbolic():
        original_output = network(images)

    """ Add output specification.
        =========================
        - `symbolic_output.argmax(-1) == labels` encodes constraints saying
          the classification (argmax of `symbolic_output`'s last dimension)
          should be the correct `labels`.
        - `solver.add_constraints(...)` adds the constraints.
    """
    solver.add_constraints(symbolic_output.argmax(-1) == labels)

    """ Construct the minimization objective.
        ====================================
        - `output_deltas` is the symbolic output delta flatten into 1D array.
        - `param_deltas` is the concatenation of all parameter delta in a 1D array.
        - `all_deltas` is the concatenation of `output_deltas` and `param_deltas`.
        - `.alias()` creates a corresponding array of variables that equals to
          the given array of symbolic expressions.
        - `.norm_ub('linf+l1_normalized')` encodes the (upper-bound) of
           the sum of L^inf norm and normalized L^1 norm.
        - `solver.minimize(obj)` sets the minimization objective.
    """
    output_deltas = (symbolic_output - original_output).flatten()
    param_deltas = N.parameter_deltas(concat=True)
    all_deltas = st.cat([output_deltas, param_deltas]).alias()
    obj = all_deltas.norm_ub('linf+l1_normalized')
    solver.minimize(obj)

    """ Solve the constraints while minimizing the objective. """
    solver.solve()
    time = timer() - start

    """ Update `N` with new parameters. """
    N.update_().repair(False)

    result_path = (get_results_root() / 'eval_1' / f'prdnn_{args.net}').as_posix()

else:
    N = network.deepcopy().decouple()
    N.load((get_artifact_root() / 'eval_1' / f'prdnn_{args.net}.pth'))
    time = None

    result_path = (get_results_root() / 'eval_1' / f'artifact_prdnn_{args.net}').as_posix()

d0 = DrawdownSet.accuracy(network)
d1 = DrawdownSet.accuracy(N)

g0 = GeneralizationSet.accuracy(network)
g1 = GeneralizationSet.accuracy(N)

result = {
    args.net: {
        ('PRDNN', 'D'): d0 - d1,
        ('PRDNN', 'G'): g1 - g0,
        ('PRDNN', 'T'): 'N/A' if time is None else f'{int(time)}s',
    }
}

np.save(result_path+".npy", result, allow_pickle=True)

if args.use_artifact:

    print_msg_box(
        f"Experiment 1 for MNIST {args.net} using PRDNN SUCCEED.\n"
        f"Saved result to {result_path}.npy"
    )

else:

    print_msg_box(
        f"Experiment 1 for MNIST {args.net} using PRDNN SUCCEED.\n"
        f"Saved result to {result_path}.npy"
    )
