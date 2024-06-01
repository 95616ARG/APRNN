# pylint: disable=import-error
try:
    from external.bazel_python.pytest_helper import main
    IN_BAZEL = True
except ImportError:
    IN_BAZEL = False

import numpy as np
import sytorch
from sytorch import *
import sytorch.legacy
import sytorch
import random
