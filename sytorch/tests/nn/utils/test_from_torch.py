# pylint: disable=import-error

import sytorch as st
from sytorch import nn
from sytorch.pervasives import *
from sytorch.solver.symbolic_array import SymbolicArray

# NOTE(anonymous): We can not test on GPU until we resolve the Bezel setup for CUDA
# libraries, etc.
seed = 0
dtype = torch.float64
device = torch.device('cpu')

def test_from_torch():
    from sytorch.tests.nn.modules.test_linear import linear_specs
    from sytorch.tests.nn.modules.test_conv import conv2d_specs
    from sytorch.tests.nn.modules.test_pooling import adaptive_average_pool2d_specs
    from sytorch.tests.nn.modules.test_pooling import max_pool2d_specs
    from sytorch.tests.nn.modules.test_batchnorm import batch_norm2d_specs

    with torch.no_grad(), st.no_symbolic():
        for TorchModule, specs, input_shape in (
            (torch.nn.Identity, (dict(),), (2, 3)),
            (torch.nn.ReLU, (dict(),), (2, 3)),
            (torch.nn.Linear, linear_specs, (2, 3)),
            (torch.nn.Conv2d, conv2d_specs, (2, 3, 12, 11)),
            (torch.nn.AdaptiveAvgPool2d, adaptive_average_pool2d_specs, ((2, 3, 12, 11))),
            (torch.nn.MaxPool2d, max_pool2d_specs, ((2, 3, 12, 11))),
            (torch.nn.Dropout, (dict(),), (2, 3)),
            (torch.nn.BatchNorm2d, batch_norm2d_specs, (2, 3, 9, 8)),
        ):
            st.set_all_seed(seed)
            input = torch.randn(input_shape).to(device,dtype)
            for spec in specs:
                print(TorchModule, spec)
                st.set_all_seed(seed)
                net_torch = TorchModule(**spec).to(device,dtype).eval()
                net_sytorch = nn.from_torch(net_torch).eval()
                output_torch = net_torch(input)
                output_sytorch = net_sytorch(input)
                assert output_torch.shape == output_sytorch.shape
                assert (output_torch == output_sytorch).all()

try:
    from external.bazel_python.pytest_helper import main
    main(__name__, __file__)
except ImportError:
    pass
