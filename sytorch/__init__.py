""" Helpers. """
import importlib
def _from_import(module: str, *attrs):
    if isinstance(module, str):
        module = importlib.import_module(module)
    for attr in attrs:
        globals()[attr] = getattr(module, attr)

def _from_import_everything(module, excludes):
    if isinstance(module, str):
        module = importlib.import_module(module)
    excludes = set(excludes) | {
        '__name__',
        '__doc__',
        '__package__',
        '__loader__',
        '__spec__',
        '__path__',
        '__file__',
        '__cached__',
        '__builtins__',
        '__all__',
    }
    for k, v in module.__dict__.items():
        if k in excludes:
            continue
        globals()[k] = v

""" Setup. """
__all__ = []

from . import pervasives
from .pervasives import *

""" Bypass `torch.__all__` and import everything from torch. """
import torch
_from_import_everything(torch, excludes={'__version__'})
__all__ += torch.__all__
__version__ = f'0.0.1+{torch.__version__}'

""" Import `.solver`. """
from . import solver
from .solver import *
from .solver import lightning
__all__ +=  getattr(solver, '__all__', [])

""" Import `.nn` and override torch's `nn`. """
from . import nn
from .nn import no_symbolic
__all__ += ['no_symbolic']

""" Clean up. """
del torch
del _from_import_everything
del _from_import
del importlib
