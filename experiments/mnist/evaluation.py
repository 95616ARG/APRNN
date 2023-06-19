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
    "compute_accuracy",
    "evaluate_efficacy",
    "evaluate_generalization",
    "evaluate_drawdown",
    "evaluate",
]

""" Evaluate helpers. """

def compute_accuracy(
    networks: nn.Module | Iterable[nn.Module],
    dataset: Dataset
) -> List[float]:
    """ Evaluate the accuracy of network(s) on the given dataset. """
    if isinstance(networks, nn.Module):
        networks = [networks]
    return [
        (network(dataset.images).argmax(dim=1) == dataset.labels)\
            .count_nonzero().item() / len(dataset)
        for network in tqdm(networks, leave=False, desc="Evaluating.")
    ]

def evaluate_efficacy(
    networks: nn.Module | Iterable[nn.Module],
    dataset: Dataset
) -> pd.DataFrame:
    """ Evaluate the efficacy of network(s) on the given dataset. """
    return pd.DataFrame(data={('performance', 'efficacy'): compute_accuracy(networks, dataset)})

def evaluate_generalization(
    networks: nn.Module | Iterable[nn.Module],
    corruption: str,
) -> pd.DataFrame:
    """ Evaluate the generalization of network(s) on the specified MNIST corruption. """
    dataset = Dataset(corruption=corruption, split='test')
    return pd.DataFrame(data={('performance', 'generalization'): compute_accuracy(networks, dataset)})

def evaluate_drawdown(
    networks: nn.Module | Iterable[nn.Module],
) -> pd.DataFrame:
    """ Evaluate the drawdown of network(s). """
    dataset = Dataset(corruption='identity', split='test')
    return pd.DataFrame(data={('performance', 'drawdown'): compute_accuracy(networks, dataset)})

def evaluate(
    networks: nn.Module | Iterable[nn.Module],
    dataset: Dataset
) -> pd.DataFrame:
    return pd.DataFrame().join([
        evaluate_efficacy(networks, dataset),
        evaluate_generalization(networks, dataset.corruption),
        evaluate_drawdown(networks)
    ], how='outer')
