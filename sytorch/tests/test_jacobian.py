# pylint: disable=import-error
try:
    from external.bazel_python.pytest_helper import main
    IN_BAZEL = True
except ImportError:
    IN_BAZEL = False

import itertools
import numpy as np
from sytorch import *
import sytorch
import random

def test_dummy():
    pass

# def test_jacobian():
#     net = nn.Linear(4, 3)
#     x = sytorch.ones(net.weight.shape[1])
#     print(sytorch.jacobian.jacobian_with_inputs_inplace(net, x))

if IN_BAZEL:
    main(__name__, __file__)
