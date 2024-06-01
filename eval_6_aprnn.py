import warnings; warnings.filterwarnings("ignore")

from experiments import mnist
from experiments.base import *
import sytorch as st
import argparse
from timeit import default_timer as timer

parser = argparse.ArgumentParser(description='')
parser.add_argument('--net', type=str, dest='net', action='store', required=True,
                    help='3x100, 9x100 or 9x200')
parser.add_argument('--npoints', '-n', type=int, dest='npoints', action='store', required=True,
                    help='npoints')
parser.add_argument('--device', type=str, dest='device', action='store', default='cpu',
                    help='device to use, e.g., cuda, cuda:0, cpu. (default=cpu).')
parser.add_argument('--use_artifact', dest='use_artifact', action='store_true',
                    help='use authors\' repaired DNN.')
args = parser.parse_args()

device = get_device(args.device)
dtype = st.float64

k = {
    '3x100_gelu': 2,
    '3x100_hardswish': 2,
}[args.net]

n_points = args.npoints
network = mnist.model(args.net).to(dtype=dtype, device=device)

seed = -1

corruption = 'fog'
GeneralizationSet = mnist.GeneralizationSet(corruption).reshape(784)
DrawdownSet = mnist.DrawdownSet().reshape(784)
RepairSet = mnist.RepairSet(corruption).reshape(784).misclassified(network)[1].shuffle(seed)[:n_points]

print(RepairSet.accuracy(network))

if args.use_artifact:
    print("Will rerun this experiment because We don't have artifact for this experiment/configuration yet.")
    args.use_artifact = True

if not args.use_artifact:

    images, labels = RepairSet.load(n_points)
    images = images.to(dtype=dtype, device=device)

    with st.no_grad():
        y0 = network(images)

    start = timer()
    solver = st.GurobiSolver()
    solver.solver.Params.Method=2
    N = network.deepcopy().to(solver).repair()
    N[k:].requires_symbolic_weight_and_bias(lb=-10., ub=10.)
    print(N[k:])
    sy = N(images)
    solver.add_constraints(sy.argmax(-1) == labels)
    output_deltas = (sy - y0).flatten()
    param_deltas = N.parameter_deltas(concat=True)
    all_deltas = st.cat([output_deltas, param_deltas]).alias()
    obj = all_deltas.norm_ub('linf+l1_normalized')
    solver.minimize(obj)
    assert solver.solve()
    time = timer() - start
    N.update_().repair(False)

    result_path = (get_results_root() / 'eval_6' / f'aprnn_{args.net}_{n_points}').as_posix()

else:
    assert False

d0 = DrawdownSet.accuracy(network)
d1 = DrawdownSet.accuracy(N)

g0 = GeneralizationSet.accuracy(network)
g1 = GeneralizationSet.accuracy(N)

result = {
    str(n_points): {
        (args.net, 'D'): f'{d0 - d1:.2%}',
        (args.net, 'G'): f'{g1 - g0:.2%}',
        (args.net, 'T'): 'N/A' if time is None else f'{int(time)}s',
    }
}

print(RepairSet.accuracy(N))

print(result)

np.save(result_path+".npy", result, allow_pickle=True)

if args.use_artifact:
    print_msg_box(
        f"Experiment 6 for MNIST {args.net} using APRNN SUCCEED.\n"
        f"Saved result to {result_path}.npy"
    )
else:
    print_msg_box(
        f"Experiment 6 for MNIST {args.net} using APRNN SUCCEED.\n"
        f"Saved result to {result_path}.npy"
    )
