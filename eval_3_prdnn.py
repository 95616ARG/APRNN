import warnings; warnings.filterwarnings("ignore")

from experiments import mnist
from experiments.base import *
import sytorch as st
import numpy as np
import sys, argparse
from timeit import default_timer as timer

""" Extend beyond this experiment.
    ==============================

    1. Different buggy network.
       - Change `net` to load a different buggy network.
         Specifically, the following command loads a torch DNN `torch_dnn_object`:

         ```
         network = st.nn.from_torch(torch_dnn_object)
         ```

         And the following command loads an ONNX DNN `onnx_dnn_path` from file:

         ```
         network = st.nn.from_file(onnx_dnn_path)
         ```

    2. Different dataset.
       - Change `npolys` to repair a different number of vpolytopes.
       - Change `repair_label` to repair images of a different label.
       - Change `degs=np.linspace(-30., 30., 7)` to rotate images by different angles.
       - Sample `points` with `labels` from the resulting vpolytopes.

    3. Different repair parameters:
       - Change the `s` and `k` parameters.
       - Change `lb` and `ub`.
"""

parser = argparse.ArgumentParser(description='')
parser.add_argument('--net', '-n', type=str, dest='net', action='store',
                    default='9x100',
                    choices=['9x100'],
                    help='Networks to repair in experiments (if applicable).')
parser.add_argument('--device', type=str, dest='device', action='store', default='cpu',
                    help='device to use, e.g., cuda, cuda:0, cpu. (default=cpu).')
parser.add_argument('--use_artifact', dest='use_artifact', action='store_true',
                    help='use authors\' repaired DNN.')

args = parser.parse_args()

device = get_device(args.device)
dtype = st.float64

npolys = 5

dnn = mnist.model('9x100').to(device,dtype)

testset = mnist.datasets.Dataset('identity', 'test').reshape(784).to(device,dtype)
mnist_c = mnist.datasets.MNIST_C(corruption='fog', split='test').reshape(784).to(device,dtype)

_, rotset0 = mnist_c.filter_label(8).misclassified(dnn)
rotset = rotset0.rotate(degs=np.linspace(-30., 30., 7), scale=1.)
generalization_set = rotset[npolys:]
g2 = rotset0.rotate(degs=np.arange(-30., 30.1, .1), scale=1.)[:npolys]

def vpoly_repair(dnn, points, labels, k, lb=-3., ub=3., Method=-1):

    """ Calculate the original output for minimization.
        ==============================================
        - The `st.no_symbolic()` context turns off the symbolic execution.
    """
    with st.no_symbolic(), st.no_grad():
        ref_out = dnn(points)

    """ Create a new solver. """
    solver = st.LightningSolver()

    """ Attach the decoupled DNN to the solver.
        =============================
        - `.deepcopy()` returns a deepcopy of the DNN to repair. This is optional.
        - `.decouple()` decouples a given DNN into a DDNN.
        - `.to(solver)` attaches the DNN to the solver, just like how you attach a
        DNN to a device in PyTorch.
        - `.repair()` turns on the repair mode and enables symbolic execution. It
        works just like `.train()` in PyTorch, which turns on the training mode.
    """
    N = dnn.deepcopy().decouple().to(solver).repair()

    """ Specify the symbolic weights.
        ==============================
        `N[k].val.weight.requires_symbolic_(lb=-3., ub=3.)` makes the
        k-th layer weight of the value network symbolic.
    """
    N[k].val.weight.requires_symbolic_(lb=lb, ub=ub)

    """ Encode the symbolic output.
        ===========================
        `N(images)` symbolically forwards `images` through `N`.
    """
    sy_points = N(points)

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
    param_delta = N.parameter_deltas(concat=True)
    output_delta = (sy_points - ref_out).reshape(-1)
    delta = param_delta.norm_ub('linf+l1_normalized') + output_delta.norm_ub('linf+l1_normalized')

    """ Solve the constraints while minimizing the objective.
       ======================================================
       - `sy_points.argmax(-1) == labels` encodes constraints saying
         the classification (argmax of `symbolic_output`'s last dimension)
         should be the correct `labels`.
       - `minimize=delta` sets the minimization objective.
       - `Method=Method` sets the gurobipy method.
    """
    solver.solve(sy_points.argmax(-1) == labels, minimize=delta, Method=Method)

    """ Update `N` with new parameters.
        ===============================
        - `.update_()` inplace update `N` with new parameters.
        - `.requires_symbolic_(False)` removes symbolic parameters.
        - `.to(None)` detaches the solver.
        - `.repair(False)` turns off the repair (symbolic) mode.
    """
    N.update_().requires_symbolic_(False).to(None).repair(False)

    return N

if not args.use_artifact:
    points = st.from_numpy(np.load(get_datasets_root() / 'eval_3_prdnn_points.npy')).to(device,dtype)
    labels = st.from_numpy(np.load(get_datasets_root() / 'eval_3_prdnn_labels.npy'))

    start = timer()
    N = vpoly_repair(dnn, points, labels, k=10, lb=-3., ub=3., Method=2)
    time = timer() - start
    result_path = (get_results_root() / 'eval_3' / f'prdnn_{args.net}').as_posix()

else:
    N = dnn.deepcopy().decouple()
    N.load((get_artifact_root() / 'eval_3' / f'prdnn.pth'))
    time = None
    result_path = (get_results_root() / 'eval_3' / f'artifact_prdnn_{args.net}').as_posix()

acc0 = testset.accuracy(dnn)
acc1 = testset.accuracy(N)
D = acc0 - acc1
# print(f"PRDNN Drawdown: {D:.2%} ({acc0:.2%} -> {acc1:.2%}).")
Dstr = f"{D:.2%} ({acc0:.2%} -> {acc1:.2%})"

acc0 = generalization_set.accuracy(dnn)
acc1 = generalization_set.accuracy(N)
G1 = acc1 - acc0
# print(f"PRDNN Generalization on S1: {G1:.2%} ({acc0:.2%} -> {acc1:.2%}).")
G1str = f"{G1:.2%} ({acc0:.2%} -> {acc1:.2%})"

acc0 = g2.accuracy(dnn)
acc1 = g2.accuracy(N)
G2 = acc1 - acc0
# print(f"PRDNN Generalization on S2: {G2:.2%} ({acc0:.2%} -> {acc1:.2%}).")
G2str = f"{G2:.2%} ({acc0:.2%} -> {acc1:.2%})"

result = {
    'PRDNN': {
        'D' : Dstr,
        'G1': G1str,
        'G2': G2str,
        'T': 'N/A' if time is None else f'{int(time)}',
    }
}

np.save(result_path+".npy", result, allow_pickle=True)

print_msg_box(
    f"Experiment 3 using PRDNN SUCCEED.\n"
    f"Saved result to {result_path}.npy"
)
