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

""" Number of vpolytopes to repair. """
npolys = 5
repair_label = 8

""" Load the buggy DNN to repair. """
dnn = mnist.model('9x100').to(device,dtype)

""" Repair parameters as introduced in the paper. """
s = ((0, 4), (4, 6), (6, 8), (8, 10), (10, 12), (12, 14), (14, 16), (16, 18))
k = 16

testset = mnist.datasets.Dataset('identity', 'test').reshape(784).to(device,dtype)
mnist_c = mnist.datasets.MNIST_C(corruption='fog', split='test').reshape(784).to(device,dtype)

_, rotset0 = mnist_c.filter_label(repair_label).misclassified(dnn)
rotset = rotset0.rotate(degs=np.linspace(-30., 30., 7), scale=1.)
generalization_set = rotset[npolys:]
g2 = rotset0.rotate(degs=np.arange(-30., 30.1, .1), scale=1.)[:npolys]

def symbolic_forward_vpolytopes(N, vpolytopes, lb=-3., ub=3.):

    """ Detach the DNN slice `N` from any previous solvers.
        ===============================
        - `.requires_symbolic_(False)` removes symbolic parameters.
        - `.to(None)` detaches the solver.
        - `.repair(False)` turns off the repair (symbolic) mode.
    """
    N.requires_symbolic_(False).to(None).repair(False)

    """ Create a new solver. """
    solver = st.GurobiSolver()

    """ Attach the DNN to the solver.
        =============================
        - `.deepcopy()` returns a deepcopy of the DNN to repair. This is optional.
        - `.to(solver)` attaches the DNN to the solver, just like how you attach a
          DNN to a device in PyTorch.
        - `.repair()` turns on the repair mode and enables symbolic execution. It
          works just like `.train()` in PyTorch, which turns on the training mode.
        - `.requires_symbolic_weight_and_bias(lb=lb, ub=ub)` makes the first layer
          weight and all layers' bias symbolic.
    """
    N.to(solver).repair().requires_symbolic_weight_and_bias(lb=lb, ub=ub)

    # `N.v(vpolytopes)` symbolically forward `vpolytopes` through `N`.
    sy_vpolytopes = N.v(vpolytopes)

    # Returns the symbolic output vpolytopes and the solver.
    return sy_vpolytopes, solver

def vpoly_repair(dnn, vpolytopes, labels, s, k, lb=-3., ub=3., Method=-1):
    """ Create a deepcopy of `dnn` to repair. """
    N = dnn.deepcopy()

    for l0, l1 in s:
        """ Forward `vpolytopes` throught `N`
            =================================
            - Concretely forward `vpolytopes` through `N[:l0]` (`N[:l0].v(vpolytopes)`).
            - Then symbolically forward `N[:l0].v(vpolytopes)` through `N[l0:l1]`.
        """
        sy_vpolytopes, solver = symbolic_forward_vpolytopes(N[l0:l1], N[:l0].v(vpolytopes), lb=lb, ub=ub)

        """ Construct the minimization objective.
            ====================================
            - `output_deltas` is the symbolic output delta flatten into 1D array.
            - `param_deltas` is the concatenation of all parameter delta in a 1D array.
            - `all_deltas` is the concatenation of `output_deltas` and `param_deltas`.
            - `.norm_ub('linf+l1_normalized')` encodes the (upper-bound) of
            the sum of L^inf norm and normalized L^1 norm.
        """
        param_delta = N[l0:l1].parameter_deltas(concat=True)
        output_delta = (sy_vpolytopes - dnn[:l1].v(vpolytopes)).reshape(-1)
        delta = param_delta.norm_ub('linf+l1_normalized') + output_delta.norm_ub('linf+l1_normalized')

        """ Solve the Shift & Assert constraints.
            =====================================
            - `minimize=delta` sets the minimization objective.
            - `Method=Method` sets the gurobipy method.
        """
        solver.solve(minimize=delta, Method=Method, TimeLimit=7200)

        # Update parameters.
        N[l0:l1].update_().requires_symbolic_(False).to(None).repair(False)

    # Concretely forward `vpolytopes` through `N[:k]` (`N[:k].v(vpolytopes)`),
    # then symbolically forward `N[:k].v(vpolytopes)` through `N[k:]`.
    sy_vpolytopes, solver = symbolic_forward_vpolytopes(N[k:], N[:k].v(vpolytopes), lb=lb, ub=ub)

    # Construct parameter delta and output delta
    param_delta = N[k:].parameter_deltas(concat=True)
    output_delta = (sy_vpolytopes - dnn.v(vpolytopes)).reshape(-1)
    delta = param_delta.norm_ub('linf+l1_normalized') + output_delta.norm_ub('linf+l1_normalized')

    """ Solve the Shift & Assert constraints.
        =====================================
        - `sy_vpolytopes.argmax(-1) == labels` encodes constraints saying
          the classification (argmax of `sy_vpolytopes`'s last dimension)
          should be the correct `labels`.
        - `minimize=delta` sets the minimization objective.
        - `Method=Method` sets the gurobipy method.
    """
    solver.solve(sy_vpolytopes.argmax(-1) == labels, minimize=delta, Method=Method)

    """ Update `N` with new parameters.
        ===============================
        - `.update_()` inplace update `N` with new parameters.
        - `.requires_symbolic_(False)` removes symbolic parameters.
        - `.to(None)` detaches the solver.
        - `.repair(False)` turns off the repair (symbolic) mode.
    """
    N[k:].update_().requires_symbolic_(False).to(None).repair(False)

    return N

imgs, labels = rotset.load(npolys)
imgs = imgs.to(device,dtype)

if not args.use_artifact:
    start = timer()
    N = vpoly_repair(dnn, imgs, labels, s=s, k=k, lb=-3, ub=3, Method=-1)
    time = timer() - start
    result_path = (get_results_root() / 'eval_3' / f'aprnn_{args.net}').as_posix()

else:
    N = dnn.deepcopy()
    N.load((get_artifact_root() / 'eval_3' / f'aprnn.pth'))
    time = None
    result_path = (get_results_root() / 'eval_3' / f'artifact_aprnn_{args.net}').as_posix()

acc0 = testset.accuracy(dnn)
acc1 = testset.accuracy(N)
D = acc0 - acc1
# print(f"APRNN Drawdown: {D:.2%} ({acc0:.2%} -> {acc1:.2%}).")
Dstr = f"{D:.2%} ({acc0:.2%} -> {acc1:.2%})"

acc0 = generalization_set.accuracy(dnn)
acc1 = generalization_set.accuracy(N)
G1 = acc1 - acc0
# print(f"APRNN Generalization on S1: {G1:.2%} ({acc0:.2%} -> {acc1:.2%}).")
G1str = f"{G1:.2%} ({acc0:.2%} -> {acc1:.2%})"

acc0 = g2.accuracy(dnn)
acc1 = g2.accuracy(N)
G2 = acc1 - acc0
# print(f"APRNN Generalization on S2: {G2:.2%} ({acc0:.2%} -> {acc1:.2%}).")
G2str = f"{G2:.2%} ({acc0:.2%} -> {acc1:.2%})"

result = {
    'APRNN': {
        'D' : Dstr,
        'G1': G1str,
        'G2': G2str,
        'T': 'N/A' if time is None else f'{int(time)}',
    }
}

np.save(result_path+".npy", result, allow_pickle=True)

print_msg_box(
    f"Experiment 3 using APRNN SUCCEED.\n"
    f"Saved result to {result_path}.npy"
)
