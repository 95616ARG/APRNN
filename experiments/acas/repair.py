from __future__ import annotations
from typing import List, Iterable, Tuple, Literal
from timeit import default_timer as timer
from tqdm.auto import tqdm
import sytorch
import sytorch.nn as nn
import numpy as np
import pandas as pd
from .evaluation import evaluate

__all__ = [
    "repair_pointwise",
    "repair_all_properties",
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
    repair_set,
    drawdown_set,
    gen_set,
    property,
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
                repair_set = repair_set,
                drawdown_set = drawdown_set,
                gen_set    = gen_set,
                property   = property,
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

        with _timeit(_timestamps, "constr"):
            lb, ub = bounds
            solver = sytorch.GurobiSolver().verbose_()
            if mode == 'decouple':
                repaired_network = network.deepcopy().decouple().to(solver).repair().requires_symbolic_(False)
                repaired_network[layer].val.requires_symbolic_(lb=lb, ub=ub)
                l = layer
                while l < len(repaired_network):
                    if isinstance(l, nn.Linear):
                        repaired_network[layer].val.bias.requires_symbolic_(lb=lb, ub=ub)
                    l+=1
                for input in tqdm(repair_set.dataloader(), leave=False, desc="Adding constraints."):
                    if property.applicable(input):
                        output = repaired_network(input.float())
                        solver.add_constraints(property(output))

            elif mode == 'quotient':
                repaired_network = network.deepcopy().to(solver).repair().requires_symbolic_(False)
                repaired_network[layer].requires_symbolic_(lb=lb, ub=ub)
                l = layer
                while l < len(repaired_network):
                    if isinstance(l, nn.Linear):
                        repaired_network[layer].bias.requires_symbolic_(lb=lb, ub=ub)
                    l+=1
                for input in tqdm(repair_set.dataloader(), leave=False, desc="Adding constraints."):
                    if property.applicable(input):
                        input = input.float()
                        output, constrs = repaired_network(input, network.activation_pattern(input))
                        solver.add_constraints(*constrs)
                        solver.add_constraints(property(output))

        with _timeit(_timestamps, "obj"):
            parameter_deltas = repaired_network.parameter_deltas(concat=True)
            solver.minimize((l1_weight * parameter_deltas.norm_ub(order="l1_normalized"))
                        + (linf_weight * parameter_deltas.norm_ub(order="linf")))

        with _timeit(_timestamps, "solve"):
            feasible = solver.solve()

        if feasible:
            repaired_network.update_().repair(False)
            print("Mode: ", mode)
            print("Repaired Network ")
            repaired = evaluate(repaired_network, repair_set, drawdown_set, gen_set)
            print("Original Network")
            original = evaluate(network, repair_set, drawdown_set, gen_set)
            eval_repaired = repaired - original
        else:
            print("Infeasible")
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

def repair_all_properties(
    network: nn.Module,
    repair_set,
    drawdown_set,
    gen_set,
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
                repair_all_properties(
                network    = network,
                repair_set = repair_set,
                drawdown_set = drawdown_set,
                gen_set    = gen_set,
                layer      = layer,
                mode       = mode,
                bounds     = bounds,
                l1_weight  = l1_weight,
                linf_weight= linf_weight,
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

        with _timeit(_timestamps, "constr"):
            lb, ub = bounds
            solver = sytorch.GurobiSolver()
            if mode == 'decouple':
                repaired_network = network.deepcopy().decouple().to(solver).repair().requires_symbolic_(False)
                repaired_network[layer].val.requires_symbolic_(lb=lb, ub=ub)
                l = layer
                while l < len(repaired_network):
                    if isinstance(l, nn.Linear):
                        repaired_network[layer].val.bias.requires_symbolic_(lb=lb, ub=ub)
                    l+=1
                for input in tqdm(repair_set.dataloader(), leave=False, desc="Adding constraints."):
                    input = input.float()
                    output = repaired_network(input)
                    for p in repair_set.valid_properties:
                        if p.applicable(input):
                            solver.add_constraints(p(output))
            elif mode == 'quotient':
                repaired_network = network.deepcopy().to(solver).repair().requires_symbolic_(False)
                repaired_network[layer].requires_symbolic_(lb=lb, ub=ub)
                l = layer
                while l < len(repaired_network):
                    if isinstance(l, nn.Linear):
                        repaired_network[layer].bias.requires_symbolic_(lb=lb, ub=ub)
                    l+=1
                for input in tqdm(repair_set.dataloader(), leave=False, desc="Adding constraints."):
                    input = input.float()
                    output, constrs = repaired_network(input, network.activation_pattern(input))
                    solver.add_constraints(*constrs)
                    for p in repair_set.valid_properties:
                        if p.applicable(input):
                            solver.add_constraints(p(output))

        with _timeit(_timestamps, "obj"):
            parameter_deltas = repaired_network.parameter_deltas(concat=True)
            solver.minimize((l1_weight * parameter_deltas.norm_ub(order="l1_normalized"))
                        + (linf_weight * parameter_deltas.norm_ub(order="linf")))

        with _timeit(_timestamps, "solve"):
            feasible = solver.solve()

        if feasible:
            repaired_network.update_().repair(False)
            print("Mode: ", mode)
            print("Layer: ", layer)
            print("Repaired network")
            repaired_results = evaluate(repaired_network, repair_set, drawdown_set, gen_set)
            print("Original network")
            original_results = evaluate(network, repair_set, drawdown_set, gen_set)
            eval_repaired = repaired_results - original_results
        else:
            print("Infeasible")
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
            index = pd.MultiIndex.from_tuples([(layer, mode)], names=['layer', 'mode']))

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
            index = pd.MultiIndex.from_tuples([(layer, mode)], names=['layer', 'mode']))
