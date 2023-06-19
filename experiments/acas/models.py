from typing import TypeVar, Literal
import warnings
import itertools
import sytorch
from experiments.base import get_models_root

__all__ = [
    "acas",
    "all_model_keys",
    "all_models",
]

T = TypeVar("T", bound=sytorch.nn.Module)
def _exclude_last_relu(network: T) -> T:
    if isinstance(network, sytorch.nn.Sequential) and isinstance(network[-1], sytorch.nn.ReLU):
        return network[:-1]

    warnings.warn(
        "The network doesn't have a last ReLU layer to exclude."
    )


def acas(
    a_prev: Literal[1,2,3,4,5],
    tau: Literal[1,2,3,4,5,6,7,8,9],
    exclude_last_relu=True
) -> sytorch.nn.Sequential:
    """
    a_prev: [Clear-of-Conflict, weak left, weak right, strong left, strong right]
    tau   : [0, 1, 5, 10, 20, 40, 60, 80, 100]
    """
    ...
    network, normalize, denormalize = sytorch.nn.from_nnet(
        (get_models_root() / "acasxu" / f"ACASXU_run2a_{a_prev}_{tau}_batch_2000.nnet").as_posix()
    )
    if exclude_last_relu:
        network = _exclude_last_relu(network)
    return network, normalize, denormalize

def all_model_keys(filter_fn=lambda keys: keys):
    return tuple(filter(filter_fn, itertools.product(range(1,6), range(1, 10))))

def all_models(filter_fn=lambda keys: keys, exclude_last_relu=True):
    return tuple(
        acas(a_prev=a_prev, tau=tau, exclude_last_relu=exclude_last_relu)
        for a_prev, tau in all_model_keys(filter_fn=filter_fn)
    )
