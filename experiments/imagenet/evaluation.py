from __future__ import annotations
from typing import List, Iterable, Tuple
from timeit import default_timer as timer
from tqdm.auto import tqdm
import sytorch
import sytorch.nn as nn
import numpy as np
import pandas as pd
from .datasets import Dataset

__all__ = [
    "evaluate_efficacy",
    # "evaluate_generalization",
    # "evaluate_drawdown",
    "evaluate",
]

""" Evaluate helpers. """

def evaluate_efficacy(
    network: nn.Module | Iterable[nn.Module],
    dataset: Dataset
) -> pd.DataFrame:
    """ Evaluate the efficacy of network(s) on the given dataset. """
    return pd.DataFrame(data={('performance', 'efficacy'): [dataset.accuracy(network)]})

# def evaluate_generalization(
#     network: nn.Module | Iterable[nn.Module],
#     corruption: str,
# ) -> pd.DataFrame:
#     """ Evaluate the generalization of network(s) on the specified MNIST corruption. """
#     dataset = Dataset(corruption=corruption, split='test')
#     return pd.DataFrame(data={('performance', 'generalization'): [dataset.accuracy(network)]})

# def evaluate_drawdown(
#     network: nn.Module | Iterable[nn.Module],
# ) -> pd.DataFrame:
#     """ Evaluate the drawdown of network(s). """
#     dataset = Dataset(corruption='identity', split='test')
#     return pd.DataFrame(data={('performance', 'drawdown'): [dataset.accuracy(network)]})

def evaluate(
    network: nn.Module | Iterable[nn.Module],
    dataset: Dataset
) -> pd.DataFrame:
    return pd.DataFrame().join([
        evaluate_efficacy(network, dataset),
        # evaluate_generalization(network, dataset.corruption),
        # evaluate_drawdown(network)
    ], how='outer')
