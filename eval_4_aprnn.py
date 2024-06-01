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

""" Load ACAS Xu N_{2,9}. """
network, _norm, _denorm = acas.models.acas(a_prev=2, tau=9)
network = network.to(device,dtype)

h = 0.05 # box size
seed = 0
dissection = False # Don't dissect boxes into disjoint simplices.
label="N29"

""" Load all applicable properties for ACAS Xu N_{2,9}. """
applicable_properties = (1,2,3,4,8,)
props = {
    prop: acas.property(prop)\
        .partition_and_classify_(_norm, network,
            h=0.05, gap=0., h_sample=0.005, label=label)
    for prop in tqdm(applicable_properties, desc="loading dataset", leave=False)
}

# N_{2,9} only violates properties 2 and 8.
# Note: due to numerical issue, on some hardware, the input polytope might
# be partitioned into different amount of boxes, hence the split with the same
# seed might produce different results.
p2_repair, p2_gen = props[2].split(num_repair=4, shuffle=True, seed=seed)
p8_repair, p8_gen = props[8].split(num_repair=20, shuffle=True, seed=seed)

def same_argmin(outputs, axis=-1):
    """ A helper function to encode the local-robustness constraint. """
    return st.SymbolicMILPArray.stack(tuple(
        (outputs.argmin(axis=axis) == label).alias().all()
        for label in range(outputs.shape[axis])
    )).sum() == 1

def repair_polytope(
    pre, post, net_ref,
    repair_props,
    lb, ub, out_spec=False, param_setter=None,
    dissection = False, method=None
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
    assert dissection == False

    """ Create a new solver. """
    solver = st.GurobiSolver().verbose_(True)
    if method is not None:
        solver.solver.Params.Method = method

    """ Attach the DNN to the solver.
        =============================
        - `.deepcopy()` returns a deepcopy of the DNN to repair. This is optional.
        - `.to(solver)` attaches the DNN to the solver, just like how you attach a
        DNN to a device in PyTorch.
        - `.repair()` turns on the repair mode and enables symbolic execution. It
        works just like `.train()` in PyTorch, which turns on the training mode.
    """
    post.to(solver).requires_symbolic_(False).repair()
    post.requires_symbolic_weight_and_bias(lb=lb, ub=ub)

    """ Calculate parameter deltas. """
    param_deltas = post.parameter_deltas(concat=True).alias().reshape(-1)
    deltas = [param_deltas]

    """ Symbolically forward boxes of each property. """
    for prop in repair_props:
        # Property 1 and 2 shares the same input polytope, hence we skip one of them.
        if isinstance(prop, acas.Property_1):
            # subsumed in prop2.
            continue

        """ Calculates the unique vertices of boxes.
            ========================================
            - `pre_vertices` are the vertices before `post`, after `pre`.
            - `vertices_pattern` are the activation patterns of `pre_vertices`.
            - `ref_output` are the reference outputs of `pre_vertices` for minimization.
        """
        with st.no_symbolic(), torch.no_grad():
            box_vertices_unique, pre_vertices, vertices_pattern, ref_output = st.calculate_vertices_and_pattern(
                    prop.violate_hboxes, pre, post, net_ref=net_ref,
                    dissection=dissection, local_rubustness=False)

        """ Symbolically forward `pre_vertices` through `post`. """
        post_symbolic_output = post(pre_vertices, pattern=vertices_pattern)

        """ Find unique vertices. """
        if out_spec:
            sy_dict = dict()
            for box_vert, box_vert_sy in zip(
                    box_vertices_unique.reshape(-1, 5),
                    post_symbolic_output.reshape(-1, 5)
                ):
                keyable = tuple(box_vert.tolist())
                assert keyable not in sy_dict
                sy_dict[keyable] = box_vert_sy

            prop.symbolic_output = post_symbolic_output
            prop.sy_dict = sy_dict

    """ Set additional custom solver parameters. """
    if param_setter is not None: param_setter(solver)

    """ Add output specifications if `out_spec=True`. """
    if out_spec:
        """ Add local-robustness constraints. """
        for prop in repair_props:
            prop.sy_dict = { k: v for k, v in prop.sy_dict.items() }
            for box in prop.violate_hboxes:
                box_vertices = st.hboxes_to_vboxes(box[None])
                box_vertices_sy = st.SymbolicLPArray.stack(tuple(
                    prop.sy_dict[tuple(box_vert.tolist())]
                    for box_vert in box_vertices.reshape(-1, 5)
                ))
                solver.add_constraints(same_argmin(box_vertices_sy, axis=-1))

        """ Add output specifications. """
        for prop in repair_props:
            if isinstance(prop, acas.Property_1):
                # subsumed in prop2
                continue
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

st.set_epsilon(1e-6)

def param_setter(solver):
    # solver.solver.Params.TimeLimit = 3600
    pass

repair_properties = (p2_repair, p8_repair)

""" NOTE: Because Gurobi by default uses a *non-deterministic* concurrent method
    to optimize a LP problem with various methods and uses the one that finishes
    first, it may not produce the same result in different runs. To improve
    reproducibility, we specify the LP methods Gurobi finally used in our run.
    However, it's difficult to control how Gurobi solves MILP problems in
    parallel.
"""

if not args.use_artifact:

    N = network.deepcopy().to(device,dtype).eval()

    start_time = timer()
    for start, end, method in (
        ( 0,  2, -1),
        ( 2,  4, -1),
        ( 4,  6, -1),
        ( 6,  8, -1),
        ( 8, 10, -1),
        (10, 12, -1),
    ):
        assert repair_polytope(
            pre=N[:start], post=N[start:end], net_ref=network[:end],
            repair_props=repair_properties,
            lb=-5., ub=5., out_spec=False, param_setter = param_setter,
            dissection = dissection, method = method
        )

    start, end, method = 12, 13, -1
    assert repair_polytope(
        pre=N[:start], post=N[start:end], net_ref=network[:end],
        repair_props=repair_properties,
        lb=-5., ub=5., out_spec=True, param_setter = param_setter,
        dissection = dissection, method = method
    )
    time = timer() - start_time
    result_path = (get_results_root() / 'eval_4' / f'aprnn_{args.net}').as_posix()

else:

    N = network.deepcopy()
    N.load((get_artifact_root() / 'eval_4' / f'aprnn.pth'))
    time = None
    result_path = (get_results_root() / 'eval_4' / f'artifact_aprnn_{args.net}').as_posix()

d, _, _ = eval_drawdown(N, props)
g, _, _ = eval_generalizataion(N, (p2_gen, p8_gen))

result = {
    'APRNN': {
        'D': f'{d:.2%}',
        'G': f'{g:.2%}',
        'T': 'N/A' if time is None else f'{int(time)}',
    }
}

np.save(result_path+".npy", result, allow_pickle=True)

print_msg_box(
    f"Experiment 4 using APRNN SUCCEED.\n"
    f"Saved result to {result_path}.npy"
)
