# pylint: disable=import-error
try:
    from external.bazel_python.pytest_helper import main
    IN_BAZEL = True
except ImportError:
    IN_BAZEL = False

import numpy as np
import sytorch
from sytorch import nn

def test_parallel_add_mode():
    sytorch.manual_seed(0)
    network = nn.Parallel(
        nn.Linear(4, 3),
        nn.Linear(4, 3),
        mode = 'add'
    ).eval()
    input = sytorch.randn((2, 4))
    output = network(input)
    assert output.shape == (2, 3)
    assert (output == network[0](input) + network[1](input)).all()

def test_decoupled_parallel_add_mode():
    sytorch.manual_seed(0)
    network = nn.Parallel(
        nn.Linear(4, 3),
        nn.Linear(4, 3),
        mode = 'add'
    ).decouple().eval()
    input = sytorch.randn((2, 4))
    output = network(input)
    print(output.shape)
    assert output.shape == (2, 3)
    assert (output == network[0](input) + network[1](input)).all()

def test_parallel_cat():
    sytorch.manual_seed(0)
    network = nn.Parallel(
        nn.Linear(4, 3),
        nn.Linear(4, 5),
        mode = 'cat', dim=-1
    ).eval()
    input = sytorch.randn((2, 4))
    output = network(input)
    assert output.shape == (2, 8)
    assert (output == sytorch.cat([module(input) for module in network], dim=1)).all()

def test_decoupled_parallel_cat_mode():
    sytorch.manual_seed(0)
    network = nn.Parallel(
        nn.Linear(4, 3),
        nn.Linear(4, 5),
        mode = 'cat', dim=-1
    ).decouple().eval()
    input = sytorch.randn((2, 4))
    output = network(input)
    assert output.shape == (2, 8)
    assert (output == sytorch.cat([module(input) for module in network], dim=1)).all()

def test_decoupled_maxpool2d():
    sytorch.manual_seed(0)
    maxpool2d_layer = nn.MaxPool2d(kernel_size=(2,2)).decouple()

    input = sytorch.tensor([[[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]]]).unsqueeze(0).float()
    output = maxpool2d_layer(input)
    assert output.shape == (input.shape[0], input.shape[1], 2, 2)
    assert sytorch.equal(output, sytorch.tensor([[[[6, 8], [14, 16]]]]).float())

def test_decoupled_avgpool2d():
    sytorch.manual_seed(0)
    adptavgpool2d_layer = nn.AdaptiveAvgPool2d(output_size=(1,1)).decouple()

    input = sytorch.tensor([[[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]]]).unsqueeze(0).float()
    output = adptavgpool2d_layer(input)

    assert output.shape == (input.shape[0], input.shape[1], 1, 1)
    assert sytorch.equal(output, sytorch.tensor([[[[8.5000]]]]))

def test_dropout():
    sytorch.manual_seed(0)
    solver = sytorch.GurobiSolver()
    dropout = sytorch.nn.Dropout(p=.5).to(solver)
    dropout.symbolic_(True)
    x = sytorch.randn(1,3,4,4)
    assert (x == dropout(x, x)[0]).all()

def test_linear():
    sytorch.manual_seed(0)
    solver = sytorch.GurobiSolver()
    fc = sytorch.nn.Linear(5, 6)
    fc = fc.to(solver).symbolic_(True).requires_symbolic_()

    sx = solver.reals((3, 5))
    cx = sytorch.randn(sx.shape)
    cw = sytorch.randn(fc.weight.delta.shape)
    cb = sytorch.randn(fc.bias  .delta.shape)

    sy = fc(sx, cx)

    solver.solve(
        sx==cx,
        fc.weight.delta==cw,
        fc.bias.delta==cb
    )

    fc.update_().repair(False)
    assert sytorch.allclose(fc(cx), sy.evaluate().float())

def test_conv2d():
    sytorch.manual_seed(0)
    solver = sytorch.GurobiSolver()
    conv = sytorch.nn.Conv2d(1, 2, kernel_size=(3, 4), padding=(2, 1), stride=(1, 2))
    conv = conv.to(solver).symbolic_(True).requires_symbolic_()
    sx = solver.reals((3, 1, 7, 8))
    sy = conv._conv_symbolic_forward(sx)

    cx = sytorch.randn(sx.shape)
    cw = sytorch.randn(conv.weight.delta.shape)
    cb = sytorch.randn(conv.bias  .delta.shape)

    solver.solve(
        sx==cx,
        conv.weight.delta==cw,
        conv.bias.delta==cb
    )

    conv.update_().repair(False)
    assert sytorch.allclose(conv(cx), sy.evaluate().float())

def test_decouple():
    pass


if IN_BAZEL:
    main(__name__, __file__)
