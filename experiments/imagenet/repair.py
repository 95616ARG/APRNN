from __future__ import annotations
from concurrent.futures import ProcessPoolExecutor
import itertools
import os
from typing import List, Iterable, Tuple, Literal, overload
from timeit import default_timer as timer
from tqdm.auto import tqdm
import sytorch
import sytorch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
from .datasets import Dataset
from .evaluation import evaluate

__all__ = [
    "repair_pointwise",
]

""" Repair helpers. """

class _timeit:
    def __init__(self, out_dict, key):
        self.out_dict = out_dict
        self.key = key

    def __enter__(self):
        # print(self.key, "...")
        self.start = timer()

    def __exit__(self, exc_type, exc_value, exc_traceback):
        self.out_dict[self.key] = timer() - self.start
        # print(f"{self.out_dict[self.key]:.3f}s")

def _to_array(item, ty):
    if isinstance(item, ty):
        item = [item]
    if not isinstance(item, np.ndarray):
        item = np.asarray(item + [None], dtype=object)[:-1]
    return item

def repair_pointwise_decoupled(
    network: nn.Module,
    dataset: Dataset,
    dtype,
    device,
    executor,
    bounds,
    linearization,
    l1_weight=1.,
    linf_weight=1.,
    verbose=True,
):
    with sytorch.no_grad():
        _timestamps = dict()

        with _timeit(_timestamps, "prepare"):
            lb, ub = bounds
            N = network\
                .deepcopy()\
                .decouple()\
                .to(dtype).to(device)\
                .to(executor)\
                .eval()\
                .repair('decouple')\
                .requires_symbolic_(False)

            for idx in linearization:
                idx = idx[:-1] + ('val',) + idx[-1:]
                N[idx].requires_symbolic_(lb=lb, ub=ub)

            """ Prepare repair set. """
            images, labels = dataset.load('all')
            images = images.to(dtype=dtype, device=device)

        with _timeit(_timestamps, "encode"):
            symbolic_outputs = N(images)

        with _timeit(_timestamps, "config"):
            """ Create model from LP file. """
            solver.solver.Params.Threads = 64
            if solver.solver.isMIP:
                # Root barrier for LP
                # Root concurrent + node barrier for MIP

                # MIP
                # solver.solver.Params.DegenMoves = 0
                # solver.solver.Params.MIPFocus = 1
                # solver.solver.Params.ConcurrentMIP = 16
                # no root relaxation work, migth help
                # solver.solver.Params.NoRelHeurTime = 30

                # concurrent method at root
                solver.solver.Params.Method = 3

            else:
                # barrier at root
                solver.solver.Params.Method = 2

            # THIS SAVED MY DAY!
            solver.solver.Params.NodeMethod = 2
            solver.solver.Params.Crossover = 0
            # solver.solver.Params.Presolve = 2

            # 0 is practical
            solver.solver.Params.BarOrder = 0

            solver.solver.Params.ScaleFlag = 3

            solver.solver.Params.FeasibilityTol = 1e-2
            solver.solver.Params.OptimalityTol = 1e-2

            solver.solver.Params.IntFeasTol = 1e-1

            solver.solver.Params.BarConvTol = 1e0
            # solver.solver.Params.BarHomogeneous = 1
            # solver.solver.Params.NumericFocus = 3

        with _timeit(_timestamps, "constr+obj"):
            solver.add_constraints(symbolic_outputs.argmax(axis=-1) == labels)

            all_deltas = N.parameter_deltas(concat=True)
            solver.minimize((l1_weight * all_deltas.norm_ub(order="l1_normalized"))
                        + (linf_weight * all_deltas.norm_ub(order="linf")))

            # seed = 0
            # percent = .1
            # solver.add_constraints(
            #     np.random.default_rng(seed).choice(
            #         all_deltas.flatten(),
            #         int(all_deltas.size * (1-percent))
            #     ) == 0.
            # )

        with _timeit(_timestamps, "solve"):
            feasible = solver.solve()

        if feasible:
            N.update_().repair(False)
        else:
            N = None

        return pd.DataFrame(
            {
                **{
                    ("timing", event): [time]
                    for event, time in _timestamps.items()
                },
                ("performance", "efficacy"): dataset.accuracy(N, device=device, dtype=dtype),
                # **eval_repaired.loc[0].to_dict(),
                ("artifact", "network"): [network],
                ("artifact", "repaired"): [N],
                ("artifact", "sy"): [symbolic_outputs],
                # ("artifact", "solver"): [solver],
            },
            index = pd.MultiIndex.from_tuples([(linearization, 'quotient')], names=['linearization', 'mode'])
        )


def repair_pointwise_quotient(
    network: nn.Module,
    dataset: Dataset,
    dtype,
    device,
    bounds,
    linearization,
    lp_file,
    executor,
    l1_weight=1.,
    linf_weight=1.,
    verbose=False,
    param_setter = None,
):
    # if num_workers is None:
    #     num_workers = os.cpu_count()

    with sytorch.no_grad():
        _timestamps = dict()

        with _timeit(_timestamps, "init"):
            """ Init for this prototype. """
            sytorch.reset_variable_counter()
            lb, ub = bounds
            """ Prepare Network N to repair. """
            N = network\
                .deepcopy()\
                .to(device=device, dtype=dtype, executor=executor)\
                .eval()\
                .repair('quotient')\
                .requires_symbolic_(False)\
                .requires_symbolic_(lb=lb, ub=ub, linearization=linearization)

            """ Prepare repair set. """
            images, labels = dataset.load('all')
            images = images.to(dtype=dtype, device=device)

        with _timeit(_timestamps, "AP"):
            patterns = N.activation_pattern(images)

        with _timeit(_timestamps, "encode"):
            symbolic_outputs, constrs = N(images, patterns)

        with _timeit(_timestamps, "write"):
            """ Write to LP file. """
            sytorch.write_lp(lp_file, constrs, N.constraints(), variables=sytorch.all_variables())

        with _timeit(_timestamps, "read"):
            """ Create model from LP file. """
            solver = sytorch.GurobiSolver(path=lp_file).verbose_(mode=verbose)
            solver.solver.Params.Threads = executor._max_workers
            if solver.solver.isMIP:
                # Root barrier for LP
                # Root concurrent + node barrier for MIP

                # MIP
                # solver.solver.Params.DegenMoves = 0
                # solver.solver.Params.MIPFocus = 1
                # solver.solver.Params.ConcurrentMIP = 16
                # no root relaxation work, migth help
                # solver.solver.Params.NoRelHeurTime = 30

                # concurrent method at root
                solver.solver.Params.Method = 3

            else:
                # barrier at root
                solver.solver.Params.Method = 2

            # THIS SAVED MY DAY!
            solver.solver.Params.NodeMethod = 2
            solver.solver.Params.Crossover = 0
            # solver.solver.Params.Presolve = 2

            # 0 is practical
            # solver.solver.Params.BarOrder = 0

            # solver.solver.Params.ScaleFlag = 3

            # solver.solver.Params.FeasibilityTol = 1e-2
            # solver.solver.Params.OptimalityTol = 1e-2

            # solver.solver.Params.IntFeasTol = 1e-1

            # solver.solver.Params.BarConvTol = 1e0
            # solver.solver.Params.BarHomogeneous = 1
            # solver.solver.Params.NumericFocus = 3

            if param_setter:
                param_setter(solver)

        with _timeit(_timestamps, "constr+obj"):
            N.to(solver)
            symbolic_outputs = symbolic_outputs.to(solver)
            # print(symbolic_outputs.shape, labels.shape)
            solver.add_constraints(symbolic_outputs.argmax(axis=-1) == labels)

            all_deltas = N.parameter_deltas(concat=True)
            solver.minimize((l1_weight * all_deltas.norm_ub(order="l1_normalized"))
                        + (linf_weight * all_deltas.norm_ub(order="linf")))

            # seed = 0
            # percent = .1
            # solver.add_constraints(
            #     np.random.default_rng(seed).choice(
            #         all_deltas.flatten(),
            #         int(all_deltas.size * (1-percent))
            #     ) == 0.
            # )

        with _timeit(_timestamps, "double checking bounds"):
            for delta in all_deltas.flat:
                assert delta.UB == ub and delta.LB == lb

        with _timeit(_timestamps, "solve"):
            feasible = solver.solve()

        if feasible:
            N.update_().repair(False)
        else:
            N = None

        return pd.DataFrame(
            {
                **{
                    ("timing", event): [time]
                    for event, time in _timestamps.items()
                },
                ("performance", "efficacy"): dataset.accuracy(N, device=device, dtype=dtype),
                # **eval_repaired.loc[0].to_dict(),
                # ("artifact", "network"): [network],
                # ("artifact", "repaired"): [N],
                # ("artifact", "sy"): [symbolic_outputs],
                # ("artifact", "solver"): [solver],
            },
            index = pd.MultiIndex.from_tuples([(linearization, 'quotient')], names=['linearization', 'mode'])
        )

@overload
def repair_pointwise(
    network: nn.Module,
    dataset: Dataset,
    dtype,
    device,
    bounds,
    linearization,
    mode,
    lp_file,
    num_workers = 'os.cpu_count()',
    l1_weight=1.,
    linf_weight=1.,
    verbose=True,
): ...

def repair_pointwise(
    networks: nn.Module,
    dataset: Dataset,
    linearizations,
    modes,
    **kwargs,
):

    results = []
    for network, linearization, mode in itertools.product(
        networks,
        linearizations,
        modes
    ):
        if mode == 'quotient':
            result = repair_pointwise_quotient(
                network     = network,
                dataset     = dataset,
                linearization = linearization,
                **kwargs
            )
            results.append(result)

        elif mode == 'decouple':
            result = repair_pointwise_decoupled(
                network     = network,
                dataset     = dataset,
                linearization = linearization,
                **kwargs
            )
            results.append(result)

        raise NotImplementedError(
            f"unimplemented mode {mode}."
        )

    report = pd.concat(results)
    return report
