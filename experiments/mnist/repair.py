from __future__ import annotations
from typing import List, Iterable, Tuple, Literal
from timeit import default_timer as timer
from tqdm.auto import tqdm
import sytorch
import sytorch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
from .datasets import Dataset
from .evaluation import evaluate
import torch

__all__ = [
    "repair_pointwise",
]

""" Repair helpers. """

class _timeit:
    def __init__(self, out_dict, key):
        self.out_dict = out_dict
        self.key = key

    def __enter__(self):
        self.start = timer()

    def __exit__(self, exc_type, exc_value, exc_traceback):
        self.out_dict[self.key] = timer() - self.start

def repair_pointwise(
    network: nn.Module,
    dataset: Dataset,
    layer: int,
    mode: Literal['decouple', 'quotient'],
    bounds: Tuple[float, float] = (-3., 3.),
    l1_weight=1.,
    linf_weight=1.,
):

    """ Batch repair. """
    if not isinstance(network, nn.Module) \
    or not isinstance(layer, int) \
    or not isinstance(mode, str):

        if isinstance(network, nn.Module):
            network = [network]

        if isinstance(layer, int):
            layer = [layer]

        if isinstance(mode, str):
            mode = [mode]

        return pd.concat(np.vectorize(
            lambda network, layer, mode:\
                repair_pointwise(
                network    = network,
                dataset     = dataset,
                layer      = layer,
                mode       = mode,
                bounds      = bounds,
                l1_weight   = l1_weight,
                linf_weight = linf_weight,
            ),
            otypes=[object]
        )(*np.meshgrid(
            np.array(network + [None], dtype=object)[:-1],
            layer,
            mode
        )).flatten())

    else:
        """ Base case. """
        _timestamps = dict()

        with _timeit(_timestamps, "init"):
            lb, ub = bounds
            solver = sytorch.GurobiSolver()
            if mode == 'decouple':
                repaired_network = network.deepcopy().decouple().to(solver).repair().requires_symbolic_(False)
                repaired_network[layer].val.requires_symbolic_(lb=lb, ub=ub)
            elif mode == 'quotient':
                repaired_network = network.deepcopy().to(solver).repair().requires_symbolic_(False)
                repaired_network[layer].requires_symbolic_(lb=lb, ub=ub)

        with _timeit(_timestamps, "constr"):
            for image, label in tqdm(DataLoader(dataset), leave=False, desc="Adding constraints."):
                activation_pattern = repaired_network.activation_pattern(image[None,:])
                symbolic_output, constrs = repaired_network(image[None,:], activation_pattern)
                solver.add_constraints(*constrs, symbolic_output.argmax(axis=0) == label)

        with _timeit(_timestamps, "obj"):
            solver.minimize((l1_weight * repaired_network.parameter_deltas(concat=True).norm_ub(order="l1_normalized"))
                        + (linf_weight * repaired_network.parameter_deltas(concat=True).norm_ub(order="linf")))

        with _timeit(_timestamps, "solve"):
            feasible = solver.solve()

        if feasible:
            repaired_network.update_().repair(False)
            eval_repaired = evaluate(repaired_network, dataset) - evaluate(network, dataset)
        else:
            repaired_network = None
            return pd.DataFrame(
            {
                **{
                    ("timing", event): [time]
                    for event, time in _timestamps.items()
                },
                ("performance", "efficacy"): [0.0],
                ("performance", "generalization"): [0.0],
                ("performance", "drawdown"): [0.0],
                ("artifact", "network"): [network],
                ("artifact", "repaired"): [repaired_network],
            },
            index = pd.MultiIndex.from_tuples([(layer, mode)], names=['layer', 'mode'])
        )

        return pd.DataFrame(
            {
                **{
                    ("timing", event): [time]
                    for event, time in _timestamps.items()
                },
                **eval_repaired.loc[0].to_dict(),
                ("artifact", "network"): [network],
                ("artifact", "repaired"): [repaired_network],
            },
            index = pd.MultiIndex.from_tuples([(layer, mode)], names=['layer', 'mode'])
        )


def box_repair(
    network: nn.Module,
    dataset: Dataset,
    layer: int,
    bounds: Tuple[float, float] = (-3., 3.),
    cutoff=None,
    l1_weight=1.,
    linf_weight=1.
):
    """Boxifies a network's activation constraints, then repairs the network."""

    # Batch repair.
    if not isinstance(network, nn.Module) \
    or not isinstance(layer, int):

        if isinstance(network, nn.Module):
            network = [network]

        if isinstance(layer, int):
            layer = [layer]

        return pd.concat(np.vectorize(
            lambda network, layer:\
                box_repair(
                network    = network,
                dataset     = dataset,
                layer      = layer,
                bounds      = bounds,
                cutoff      = cutoff,
                l1_weight   = l1_weight,
                linf_weight = linf_weight,
            ),
            otypes=[object]
        )(*np.meshgrid(
            np.array(network + [None], dtype=object)[:-1],
            layer
        )).flatten())

    else:
        # Base case.
        _timestamps = dict()
        lb, ub = bounds
        solver = sytorch.GurobiSolver()
        assert(isinstance(network[layer], nn.Linear))

        # Set the designated repair layer to symbolic, as well as biases for all subsequent layers.
        repaired_network = network.deepcopy().to(solver).repair().requires_symbolic_(False)
        repaired_network[layer].requires_symbolic_(lb=lb, ub=ub)
        if layer+1 == len(network):
            print("No activation constraints to boxify since this is the last layer.")
            return None
        index = layer
        while index < len(network):
            if isinstance(repaired_network[index], nn.Linear):
                repaired_network[index].bias.requires_symbolic_(lb=lb, ub=ub)
            index+=1

        with _timeit(_timestamps, "boxify"):
            print("Boxify")
            if cutoff is None or cutoff < layer:
                # Add activation pattern constraints to the solver.
                for data in DataLoader(dataset):
                    image, label = data
                    _, act_constrs = repaired_network(image[None,:], repaired_network.activation_pattern(image[None,:]))
                    solver.add_constraints(*act_constrs)

                # Boxify the activation pattern constraints, then remove them from the solver.
                lower_bounds, upper_bounds = nn.boxify(solver.solver.getA().toarray(), solver.solver.getAttr('Sense', solver.solver.getConstrs()), 
                                                        solver.solver.getAttr('RHS', solver.solver.getConstrs()), lb=lb, ub=ub, min_range=0)
                if lower_bounds is None or upper_bounds is None:
                    return None
                solver.solver.remove(solver.solver.getConstrs())

                # Add the box constraints to the solver.
                for d, l, u in zip(repaired_network.parameter_deltas(concat=True), lower_bounds, upper_bounds):
                    solver.add_constraints(d <= u)
                    solver.add_constraints(d >= l)
            else:
                back_constrs = []
                # Add activation pattern constraints to the solver.
                print("Add activation pattern constraints to solver.")
                for data in DataLoader(dataset):
                    image, label = data
                    pattern = repaired_network.activation_pattern(image)
                    _, act_constrs = repaired_network(image, pattern)
                    back_constrs.append(act_constrs[cutoff:])
                    solver.add_constraints(*act_constrs[:cutoff])

                # Boxify the activation pattern constraints, then remove them from the solver.
                lower_bounds, upper_bounds = nn.boxify(solver.solver.getA().toarray(), solver.solver.getAttr('Sense', solver.solver.getConstrs()), 
                                                        solver.solver.getAttr('RHS', solver.solver.getConstrs()), lb=lb, ub=ub, min_range=0)
                if lower_bounds is None or upper_bounds is None:
                    return None
                solver.solver.remove(solver.solver.getConstrs())

                print("Adding box constraints to solver.")
                # Add the box constraints to the solver.
                for d, l, u in zip(repaired_network.parameter_deltas(concat=True), lower_bounds, upper_bounds):
                    solver.add_constraints(d <= u)
                    solver.add_constraints(d >= l)

                print("Adding additional activation constraints.")
                # Add the rest of the activation constraints.
                for constr in back_constrs:
                    solver.add_constraints(*constr)


        with _timeit(_timestamps, "out_constr"):
            print("Adding output constraints.")
            # Add the output constraints to the solver.
            for data in DataLoader(dataset):
                image, label = data
                sym_out, _ = repaired_network(image[None,:], repaired_network.activation_pattern(image[None,:]))
                out_constrs = sym_out.argmax(axis=0) == label
                solver.add_constraints(*out_constrs)

        with _timeit(_timestamps, "obj and solve"):
            print("Solving.")
            # Minimize objective and solve.
            solver.minimize((l1_weight * repaired_network.parameter_deltas(concat=True).norm_ub(order="l1_normalized"))
                        + (linf_weight * repaired_network.parameter_deltas(concat=True).norm_ub(order="linf")))
            feasible = solver.solve()

        # Report performance.
        if feasible:
            repaired_network.update_().repair(False)
            eval_repaired = evaluate(repaired_network, dataset) - evaluate(network, dataset)
            return pd.DataFrame(
                {
                    **{
                    ("timing", event): [time]
                    for event, time in _timestamps.items()
                    },
                    **eval_repaired.loc[0].to_dict(),
                }, index = [layer])
        else:
            return pd.DataFrame(
                {
                    **{
                    ("timing", event): [time]
                    for event, time in _timestamps.items()
                    },
                    ("performance", "efficacy"): [0.0],
                    ("performance", "generalization"): [0.0],
                    ("performance", "drawdown"): [0.0],
                }, index = [layer])
    
