import warnings; warnings.filterwarnings("ignore")

from experiments import acas
from experiments.base import *
import sytorch as st
import torch
import numpy as np
import sys, argparse
from tqdm.auto import tqdm
from timeit import default_timer as timer

""" Extend beyond this experiment.
    ==============================

    1. Different partitioning schemes:
       - Change the box size `h` parameter.

    2. Different AXAS Xu networks:
       - Change `aprev` and `tau` parameters to load a different network.
       - Change `applicable_properties` and `repair_properties` accordingly to
         the list of applicable properties.

    3. Different repair parameters:
       - Change the `s` and `k` parameters as introduced in the paper.
"""

parser = argparse.ArgumentParser(description='')
parser.add_argument('--net', '-n', type=str, dest='net', action='store',
                    default='n29',
                    choices=['n29'],
                    help='Networks to repair in experiments (if applicable).')
parser.add_argument('--device', type=str, dest='device', action='store', default='cpu',
                    help='device to use, e.g., cuda, cuda:0, cpu. (default=cpu).')
parser.add_argument('--use_artifact', dest='use_artifact', action='store_true',
                    help='use authors\' repaired DNN.')

args = parser.parse_args()

device = get_device(args.device)
dtype = st.float64

network, _norm, _denorm = acas.models.acas(a_prev=2, tau=9)
network = network.to(device,dtype)

h = 0.05 # box size
seed = 0
dissection = False
label="N29"

applicable_properties = (1,2,3,4,8,)
props = {
    prop: acas.property(prop)\
        .partition_and_classify_(_norm, network,
            h=h, gap=0., h_sample=0.005, label=label)
    for prop in tqdm(applicable_properties, desc="loading dataset", leave=False)
}

# N_2,9 only violates properties 2 and 8.
p2_repair, p2_gen = props[2].split(num_repair=4, shuffle=True, seed=seed)
p8_repair, p8_gen = props[8].split(num_repair=20, shuffle=True, seed=seed)

def same_argmin(outputs, axis=-1):
    return st.SymbolicMILPArray.stack(tuple(
        (outputs.argmin(axis=axis) == label).alias().all()
        for label in range(outputs.shape[axis])
    )).sum() == 1

def eval_drawdown(dnn, props):
    # Because property 1 and 2 shares the same input polytope, N_2,9 satisfies
    # property 1 but violates 2, props[2].satisfy_accuracy considers points that
    # originally satisfies both properties 1 and 2.
    _, n_sat, n_total = props[2].satisfy_accuracy(dnn, other_prop=props[1])
    for p in (3, 4, 8):
        _, n_prop_sat, n_prop_total = props[p].satisfy_accuracy(dnn)
        n_sat += n_prop_sat
        n_total += n_prop_total

    drawdown = 1. - float(n_sat / n_total)
    return drawdown, n_sat, n_total

def eval_generalizataion(dnn, datasets):
    n_sat, n_total = 0, 0
    for dataset in datasets:
        _, n_dataset_sat, n_dataset_total = dataset.violate_accuracy(dnn)
        n_sat += n_dataset_sat
        n_total += n_dataset_total
    generalization = float(n_sat / n_total)
    return generalization, int(n_sat), int(n_total)


repair_properties = (p2_repair, p8_repair)

k = -1

if not args.use_artifact:

    st.set_epsilon(1e-6)

    def param_setter(solver):
        # solver.solver.Params.TimeLimit = 3600
        pass

    start_time = timer()

    """ Create a new solver. """
    solver = st.GurobiSolver().verbose_(True)

    """ Attach the decoupled DNN to the solver.
        =============================
        - `.deepcopy()` returns a deepcopy of the DNN to repair. This is optional.
        - `.decouple()` decouples a given DNN into a DDNN.
        - `.to(solver)` attaches the DNN to the solver, just like how you attach a
        DNN to a device in PyTorch.
        - `.repair()` turns on the repair mode and enables symbolic execution. It
        works just like `.train()` in PyTorch, which turns on the training mode.
    """
    N = network.deepcopy().decouple().to(solver).repair(True)

    """ Specify the symbolic weights.
        ==============================
        `N[k].val.weight.requires_symbolic_(lb=-5., ub=5.)` makes the
        k-th layer weight of the value network symbolic.
    """
    N[k].val.weight.requires_symbolic_(lb=-5., ub=5.)

    """ Calculate parameter deltas. """
    param_deltas = N.parameter_deltas(concat=True).alias().reshape(-1)
    deltas = [param_deltas]

    """ Symbolically forward samples in boxes of each property. """
    for p in repair_properties:
        all_points = []
        """ Sample `points` in each box of property `p`. """
        for box in tqdm(p.violate_hboxes, desc=f"{p}"):
            with st.no_symbolic():
                center = st.center_of_hboxes(box[None]).to(device,dtype)
                points = st.sample_hbox(box, h=0.016).to(device,dtype)
                points = torch.cat((center, points))
                all_points.append(points)
        all_points = torch.cat(all_points, dim=0)
        # all_points = torch.stack(all_points, dim=0)
        p.all_points = all_points
        shape = all_points.shape

        """ Calculate the original output for minimization. """
        with st.no_symbolic():
            ref_output = network(all_points)

        """ Symbolically forward sampled points. """
        symbolic_output = N(all_points.reshape(-1, 5)).reshape(shape)

        """ Calculate the output deltas. """
        deltas.append((ref_output - symbolic_output).alias().reshape(-1))

        p.symbolic_output = symbolic_output

    """ Set gurobipy solver method. """
    solver.solver.Params.Method = -1

    """ Add constraints. """
    for p in repair_properties:
        """ Add output specifications. """
        solver.add_constraints(p(p.symbolic_output.reshape(-1, 5)))

        """ Add (approximate) local-robustness constraints. """
        for sy in p.symbolic_output:
            solver.add_constraints(same_argmin(sy, axis=-1))

    """ Set the minimization objective. """
    deltas = type(deltas[0]).concatenate(deltas).to(solver)
    solver.minimize(deltas.norm_ub('linf') + deltas.norm_ub('l1_normalized'))

    solver.solve()

    """ Update `post` with new parameters. """
    N = N.update_().requires_symbolic_(False).to(None).repair(False)

    time = timer() - start_time
    result_path = (get_results_root() / 'eval_4' / f'prdnn_{args.net}').as_posix()

else:

    N = network.deepcopy().decouple()
    N.load((get_artifact_root() / 'eval_4' / f'prdnn.pth'))
    time = None
    result_path = (get_results_root() / 'eval_4' / f'artifact_prdnn_{args.net}').as_posix()

d, _, _ = eval_drawdown(N, props)
g, _, _ = eval_generalizataion(N, (p2_gen, p8_gen))

result = {
    'PRDNN': {
        'D': f'{d:.2%}',
        'G': f'{g:.2%}',
        'T': 'N/A' if time is None else f'{int(time)}',
    }
}

np.save(result_path+".npy", result, allow_pickle=True)

print_msg_box(
    f"Experiment 4 using PRDNN SUCCEED.\n"
    f"Saved result to {result_path}.npy"
)
