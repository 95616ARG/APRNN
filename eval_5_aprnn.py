import warnings; warnings.filterwarnings("ignore")

from experiments import acas
from experiments.base import *
import sytorch as st
from sytorch.pervasives import calculate_vertices_and_pattern
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

h = 0.13         # box size
gap = 0.0        # no gap between boxes.
h_sample = 0.01  # sample spacing
lb = -5.0        # upper-bound of parameter delta
ub = 5.0         # lower-bound of parameter delta
seed = 0
s = ((0, 6, -1), (6, 8, -1), (8, 10, -1), (10, 12, -1))
k = 12
dissection = True # Dissect boxes into disjoint simplices.

""" Load ACAS Xu N_{2,9}. """
aprev = 2
tau = 9
network, _norm, _denorm = acas.models.acas(aprev, tau)
network = network.to(device,dtype)

""" Load all applicable properties for ACAS Xu N_{2,9}. """
applicable_properties = (1,2,3,4,8,)

props = {
    prop: acas.property(prop)\
        .partition_and_classify_(_norm, network,
            h=h, gap=gap, h_sample=h_sample, label="N29")
    for prop in tqdm(applicable_properties, desc="loading dataset", leave=False)
}

repair_properties = tuple(prop for _, prop in props.items())

def repair_polytope(
    pre, post, net_ref,
    repair_props,
    lb, ub, out_spec=False, param_setter=None,
    dissection = True, Method = None
):
    """ Parameters
        ==========
        - `pre` is the leading concrete layers.
        - `post` is the following symbolic layers.
        - `net_ref` is the reference (original) network for minimization.
        - `repair_props` are the ACAS Xu properties to repair.
        - `lb` and `ub` are the lower and upper bounds of parameter deltas.
        - `out_spec` is whether to add output specifications.
        - `param_setter` is a function that sets additional custom solver parameters.
        - `dissection` is whether to dissect boxes into disjoint simplices.
        - `Method` is the Gurobi solver method.

        Returns
        =======
        True if the repair is successful with `post` repaired, False otherwise.
    """
    assert dissection is True

    """ Create a new solver. """
    solver = st.GurobiSolver().verbose_(True)

    """ Attach the DNN to the solver.
        =============================
        - `.deepcopy()` returns a deepcopy of the DNN to repair. This is optional.
        - `.to(solver)` attaches the DNN to the solver, just like how you attach a
        DNN to a device in PyTorch.
        - `.repair()` turns on the repair mode and enables symbolic execution. It
        works just like `.train()` in PyTorch, which turns on the training mode.
    """
    post.to(solver).requires_symbolic_(False).repair()

    """ Specify the symbolic weights.
        ==============================
        `post.requires_symbolic_weight_and_bias(lb=-3., ub=3.)` makes the
        first layer weight and all layers' bias symbolic.
    """
    post.requires_symbolic_weight_and_bias(lb=lb, ub=ub)

    """ Calculate parameter deltas. """
    param_deltas = post.parameter_deltas(concat=True).alias().reshape(-1)
    deltas = [param_deltas]

    """ Symbolically forward boxes of each property. """
    for prop in repair_props:
        # Property 1 and 2 shares the same input polytope, hence we skip one of them.
        if isinstance(prop, acas.Property_1):
            # print('skip prop1, subsumed in prop2.')
            continue

        """ Calculates the unique vertices of boxes.
            ========================================
            - `pre_vertices` are the vertices before `post`, after `pre`.
            - `vertices_pattern` are the activation patterns of `pre_vertices`.
            - `ref_output` are the reference outputs of `pre_vertices` for minimization.
        """
        with st.no_symbolic(), torch.no_grad():
            pre_vertices, vertices_pattern, ref_output = calculate_vertices_and_pattern(
                prop.hboxes, pre, post, net_ref=net_ref, dissection=dissection, local_rubustness=False)

        """ Symbolically forward `pre_vertices` through `post`. """
        post_symbolic_output = post(pre_vertices, pattern=vertices_pattern)
        prop.symbolic_output = post_symbolic_output

        """ Calculate the output deltas. """
        output_deltas = (ref_output - post_symbolic_output).alias().reshape(-1)
        deltas.append(output_deltas)

    """ Set additional custom solver parameters. """
    if param_setter is not None: param_setter(solver)
    if Method is not None: solver.solver.Params.Method = Method

    """ Add output specifications if `out_spec=True`. """
    for prop in repair_props:
        if isinstance(prop, acas.Property_1):
            # print('skip prop1, subsumed in prop2.')
            continue
        if out_spec:
            solver.add_constraints(prop(prop.symbolic_output))
            if isinstance(prop, acas.Property_2):
                solver.add_constraints(acas.Property_1()(prop.symbolic_output))

        prop.symbolic_output = None

    """ Set the minimization objective. """
    deltas = type(deltas[0]).concatenate(deltas)
    solver.minimize(deltas.norm_ub('linf') + deltas.norm_ub('l1_normalized'))

    if solver.solve():
        """ Update `post` with new parameters. """
        post.update_().requires_symbolic_(False).to(None).repair(False)
        return True
    else:
        post.requires_symbolic_(False).to(None).repair(False)
        return False

if not args.use_artifact:

    st.set_epsilon(1e-7)

    def param_setter(solver):
        pass

    N = network.deepcopy().to(device,dtype).eval()

    start_time = timer()

    """ Shift linear regions. """
    for start, end, method in s:
        assert repair_polytope(
            pre=N[:start], post=N[start:end], net_ref=network[:end],
            repair_props=repair_properties,
            lb=lb, ub=ub, out_spec=False, param_setter = param_setter,
            dissection = dissection, Method=method
        )

    """ Assert the output specifications. """
    start, end = k, 13
    repair_polytope(
        pre=N[:start], post=N[start:end], net_ref=network[:end],
        repair_props=repair_properties,
        lb=lb, ub=ub, out_spec=True, param_setter = param_setter,
        dissection = dissection, Method = -1
    )
    time = timer() - start_time
    result_path = (get_results_root() / 'eval_5' / f'aprnn_{args.net}').as_posix()

else:
    N = network.deepcopy()
    N.load((get_artifact_root() / 'eval_5' / f'aprnn.pth'))
    time = None
    result_path = (get_results_root() / 'eval_5' / f'artifact_aprnn_{args.net}').as_posix()

result = {
    'APRNN': {
        'T': 'N/A' if time is None else f'{int(time)}',
    }
}

np.save(result_path+".npy", result, allow_pickle=True)

print_msg_box(
    f"Experiment 5 using APRNN SUCCEED.\n"
    f"Saved result to {result_path}.npy"
)

