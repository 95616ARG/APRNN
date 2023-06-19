# pylint: disable=import-error

import sytorch as st
from sytorch import nn
from sytorch.pervasives import *
import torchvision

from sytorch.solver.symbolic_array import SymbolicArray

seed = 0
dtype = torch.float64
device = torch.device('cpu')

def test_resnet18():
    return
    st.set_all_seed(0)
    with torch.no_grad():
        solver = st.LightningSolver()
        resnet = nn.from_torch(torchvision.models.resnet18(pretrained=False).to(device,dtype)).symbolic_()
        input = torch.randn((2, 3, 224, 224)).to(device,dtype)
        symbolic_input = solver.reals(input.shape)

        with st.no_symbolic():
            output = resnet(input)

        pattern = resnet.activation_pattern(input)
        symbolic_output = resnet(symbolic_input, pattern=pattern)
        assert isinstance(symbolic_output, SymbolicArray)
        assert solver.solve(symbolic_input == input)
        assert torch.allclose(symbolic_output.evaluate().to(device,dtype), output)

try:
    from external.bazel_python.pytest_helper import main
    main(__name__, __file__)
except ImportError:
    pass
