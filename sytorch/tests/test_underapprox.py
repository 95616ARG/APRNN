try:
    from external.bazel_python.pytest_helper import main
    IN_BAZEL = True
except ImportError:
    IN_BAZEL = False
import numpy as np
import sytorch.nn as nn
from sytorch import *

def test_boxify():
    # Test boxify when polytope is a box
    A = np.asarray([[-1., 0.], [1., 0.], [0., -1.], [0., 1]])
    sense = ['<', '<', '<', '<']
    b = [-2., 4., -1., 3.]
    lower, upper = nn.boxify(A, sense, b, lb=-3., ub=3.)
    assert(np.allclose(lower, [2., 1.]))
    assert(np.allclose(upper, [4., 3.]))

    # Test boxify on a small pentagon
    A = np.asarray([[1., 0.], [0., -1.], [0., 1], [-1.5, 1.], [1., 1.]])
    sense = ['<', '<', '<', '<', '>']
    b = [ 4., -1., 3., 0., 3.]
    lower, upper = nn.boxify(A, sense, b, lb=-3., ub=3.)
    assert(np.allclose(lower, [2., 1.]))
    assert(np.allclose(upper, [4., 3.]))

    # Test boxify on a large pentagon
    A = np.asarray([[1., 0.], [0., 1.], [0., 1.], [0.5, 1.], [-1.5, 1.]])
    sense = ['>', '<', '>', '<', '>']
    b = [-3., 3., -1., 4., -4.]
    lower, upper = nn.boxify(A, sense, b, lb=-4., ub=4.)
    assert(np.allclose(lower, [-3., -1.]))
    assert(np.allclose(upper, [2., 3.]))

def test_fix_sense():
    # Test '>' to '<'
    A = np.eye(2, 2)
    sense = ['<', '>']
    b = [1., 2.]
    A, sense, b = nn.fix_sense(A, sense, b)

    assert(np.allclose(A, np.asarray([[1., 0.], [0., -1.]])))
    assert(sense == ['<', '<'])
    assert(b == [1., -2.])

    # Test '=' to '<'
    A = np.eye(2, 2)
    sense = ['=', '<']
    b = [1., 2.]
    A, sense, b = nn.fix_sense(A, sense, b)

    assert(np.allclose(A, np.asarray([[1., 0.], [0., 1.], [-1., 0]])))
    assert(sense == ['<', '<', '<'])
    assert(b == [1., 2., -1.])

if IN_BAZEL:
    main(__name__, __file__)
