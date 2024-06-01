from __future__ import annotations
from typing import List, Iterable, Tuple
from timeit import default_timer as timer
from tqdm.auto import tqdm
import sytorch.nn as nn
import pandas as pd
import numpy as np

__all__ = [
    "evaluate",
]

""" Evaluate helpers. """
def evaluate(
    network: nn.Module,
    repair_set,
    drawdown_set,
    gen_set,
) -> pd.DataFrame:

    repair_acc = repair_set.accuracy(network)
    gen_acc = gen_set.accuracy(network)
    drawdown_acc = drawdown_set.accuracy(network)
    # print("Repair Spec Efficacies: ", repair_acc)
    # print("Generalizations: ", gen_acc)
    # print("Drawdowns: ", drawdown_acc)

    return pd.DataFrame().join([
        pd.DataFrame(data={('performance', 'efficacy'): repair_acc}),
        pd.DataFrame(data={('performance', 'generalization'): gen_acc}),
        pd.DataFrame(data={('performance', 'drawdown'): drawdown_acc}),
    ], how='outer')
