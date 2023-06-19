import sytorch as st
from . import models
from .datasets import *
from .repair import *
from .evaluation import *
from experiments.base import get_workspace_root

def model(name):
    filename = models.model_dict[name]
    filepath = get_workspace_root() / "models" / "mnist" / filename
    
    if name == '3x100_hardswish':
        network = st.nn.Sequential(
            st.nn.Linear(784, 100), st.nn.Hardswish(),
            st.nn.Linear(100, 100), st.nn.Hardswish(),
            st.nn.Linear(100,  10)
        ).load(filepath)

    elif name in [ '3x100_gelu' ]:
        network = st.nn.Sequential(
            st.nn.Linear(784, 100), st.nn.GELU(),
            st.nn.Linear(100, 100), st.nn.GELU(),
            st.nn.Linear(100,  10)
        ).load(filepath)
    else:
        network = st.nn.from_file(filepath.as_posix())

    if isinstance(network[-1], sytorch.nn.ReLU):
        network = network[:-1]
    return network
