# pylint: disable=import-error
try:
    from external.bazel_python.pytest_helper import main
    IN_BAZEL = True
except ImportError:
    IN_BAZEL = False

import numpy as np
import sytorch
from sytorch import nn

def test_ap_default():
    net = nn.Module()
    x = sytorch.randn(10,10) - .5
    try:
        net.activation_pattern(x)
    except NotImplementedError:
        pass
    except:
        assert False

def test_ap_linear():
    net = nn.Linear(3,4)
    x = sytorch.randn(5,3) - .5
    ap = net.activation_pattern(x)
    assert ap == []

def _ap_eq_relu(ap0, ap1):
    return np.array_equal(ap0, ap1)

def test_ap_relu():
    net = nn.ReLU()
    x = sytorch.FloatTensor([[1., -1.9, 0.7], [-1.5, 0., 2.]])
    ap = net.activation_pattern(x)
    assert _ap_eq_relu(ap, np.array([[True, False, True], [False, True, True]]))

def test_ap_sequential():
    net = nn.Sequential(
        nn.Identity(),
        nn.ReLU(),
        nn.Sequential(
            nn.Identity(),
            nn.ReLU(),
        ),
        nn.Identity(),
    )
    x = sytorch.randn(5,2) - .5
    ap = net.activation_pattern(x)
    assert ap[0] == []
    assert _ap_eq_relu(ap[1], nn.ReLU().activation_pattern(x))
    assert ap[2][0] == []
    assert _ap_eq_relu(ap[2][1], nn.ReLU().activation_pattern(sytorch.ones_like(x)))
    assert ap[3] == []

def test_ap_parallel():
    net = nn.Parallel(
        nn.Identity(),
        nn.ReLU(),
        nn.Parallel(
            nn.Identity(),
            nn.ReLU(),
            mode = 'add',
        ),
        mode = 'cat', dim=-1
    )
    x = sytorch.randn(5,2) - .5
    ap = net.activation_pattern(x)
    assert ap[0] == []
    assert _ap_eq_relu(ap[1], nn.ReLU().activation_pattern(x))
    assert ap[2][0] == []
    assert _ap_eq_relu(ap[2][1], nn.ReLU().activation_pattern(x))

if IN_BAZEL:
    main(__name__, __file__)
