# pylint: disable=import-error

import sytorch as st
from sytorch import nn
from sytorch.pervasives import *
import torchvision

# NOTE(anonymous): We can not test on GPU until we resolve the Bezel setup for CUDA
# libraries, etc.
seed = 0
dtype = torch.float64
device = torch.device('cpu')

def _get_kwargs(**kwargs):
    return kwargs

# NOTE(anonymous): Loading pretrained parameters requires network, etc., but Bazel env
# file system is read-only, therefore we use random parameters to test.
torchvision_default_specs = (
    _get_kwargs(pretrained=False),
)

def test_from_torchvision():
    with torch.no_grad(), st.no_symbolic():
        for TorchModule, specs, input_shape in (
            (torchvision.models.squeezenet1_0, torchvision_default_specs, (2, 3, 224, 224)),
            (torchvision.models.squeezenet1_1, torchvision_default_specs, (2, 3, 224, 224)),
            (torchvision.models.vgg11        , torchvision_default_specs, (2, 3, 224, 224)),
            (torchvision.models.vgg11_bn     , torchvision_default_specs, (2, 3, 224, 224)),
            (torchvision.models.vgg13        , torchvision_default_specs, (2, 3, 224, 224)),
            (torchvision.models.vgg13_bn     , torchvision_default_specs, (2, 3, 224, 224)),
            (torchvision.models.vgg16        , torchvision_default_specs, (2, 3, 224, 224)),
            (torchvision.models.vgg16_bn     , torchvision_default_specs, (2, 3, 224, 224)),
            (torchvision.models.vgg19        , torchvision_default_specs, (2, 3, 224, 224)),
            (torchvision.models.vgg19_bn     , torchvision_default_specs, (2, 3, 224, 224)),
            (torchvision.models.resnet18     , torchvision_default_specs, (2, 3, 224, 224)),
            (torchvision.models.resnet34     , torchvision_default_specs, (2, 3, 224, 224)),
            (torchvision.models.resnet50     , torchvision_default_specs, (2, 3, 224, 224)),
            (torchvision.models.resnet101    , torchvision_default_specs, (2, 3, 224, 224)),
            (torchvision.models.resnet152    , torchvision_default_specs, (2, 3, 224, 224)),
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
