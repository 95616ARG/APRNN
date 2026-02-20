import pathlib
from .modules import *
import torch
import numpy as np
import ast

__all__ = [
    "from_file",
    "from_eran",
    "from_torch",
    "from_onnx",
    "from_nnet",
    "to_onnx",
    'from_eran_pyt',
    'to_syrenn',
    'syrenn',
]

def to_syrenn(net):
    import pysyrenn

    if isinstance(net, Sequential):
        return pysyrenn.frontend.Network([
            to_syrenn(module) for module in net
        ])

    elif isinstance(net, Linear):
        return pysyrenn.frontend.FullyConnectedLayer(
            weights = net.weight.detach().cpu().numpy().T,
            biases  = net.bias.detach().cpu().numpy()
        )

    elif isinstance(net, ReLU):
        return pysyrenn.frontend.ReluLayer()

    elif isinstance(net, NormalizeInput):
        return pysyrenn.frontend.NormalizeLayer(
            means = net.mean.detach().cpu().numpy(),
            standard_deviations = net.std.detach().cpu().numpy()
        )

    else:
        raise NotImplementedError(
            f"unimplemented .to_syrenn(...) for {type(net)}."
        )

def syrenn(net, polytopes, preimages=True, raw=False):
    import pysyrenn

    if isinstance(polytopes, Tensor):
        polytopes = polytopes.detach().cpu().numpy()

    # if polytopes.shape[-1] != 2:
    #     raise RuntimeError(
    #         f"unimplemented SyReNN on {polytopes.shape[-1]}D polytopes."
    #     )

    if len(polytopes.shape) == 2:
        polytopes = polytopes[None,...]

    if len(polytopes.shape) != 3:
        raise RuntimeError(
            f"expect input polytopes for SyReNN of shape ([n_polytopes, ]n_vertices, n_dims)."
        )

    classifier = pysyrenn.PlanesClassifier(
        network   = to_syrenn(net),
        planes    = polytopes,
        preimages = preimages,
    )

    classifier.partial_compute()

    """ list of (pre, post) """
    if raw:
        return [
            [
                (torch.from_numpy(pre), torch.from_numpy(post))
                for pre, post in polytopes
            ] for polytopes in classifier.transformed_planes
        ]

    else:
        pres = []
        posts = []
        for polytopes in classifier.transformed_planes:
            poly_pre = []
            poly_post = []
            for pre, post in polytopes:
                poly_pre.append(torch.from_numpy(pre))
                poly_post.append(torch.from_numpy(post))
            pres.append(poly_pre)
            posts.append(poly_post)
        return pres, posts

def _parse_np_array_as_tensor(serialized):
    """Given a string, returns a Numpy array of its contents.

    Used when parsing the ERAN model definition files.
    """
    if isinstance(serialized, str):
        return torch.from_numpy(np.array(ast.literal_eval(serialized)))
    # Helper to read directly from a file.
    return _parse_np_array_as_tensor(serialized.readline()[:-1].strip())

def from_file(path: str, file_type=None):
        """Loads a network from an ONNX or ERAN file format.

        Files ending in .onnx will be loaded as ONNX files, ones ending in
        .eran will be loaded as ERAN files. Pass file_tye="{eran, onnx}" to
        override this behavior.
        """
        if file_type is None:
            file_type = path.split(".")[-1]
        file_type = file_type.lower()

        if file_type in ("eran", "tf"):
            return from_eran(path)
        elif file_type == "onnx":
            return from_onnx(path)
        elif file_type == "nnet":
            return from_nnet(path)

        raise NotImplementedError

def from_eran(path):
    """
    Modified from https://github.com/95616ARG/indra/blob/5cfbd139745d720dac31854b87efc
    d221f5e620b/SyReNN/pysyrenn/frontend/network.py

    Helper method to read an ERAN net_file into a Network.

    Currently only supports a subset of those supported by the original
    read_net_file.py. See an example of the type of network file we're
    reading here:

    https://files.sri.inf.ethz.ch/eran/nets/tensorflow/mnist/mnist_relu_3_100.tf

    This code has been adapted (with heavy modifications) from the ERAN
    source code. Each layer has a header line that describes the type of
    layer, which is then followed by the weights (if applicable). Note that
    some layers are rolled together in ERAN but we do separately (eg.
    "ReLU" in the ERAN format corresponds to Affine + ReLU in our
    representation).
    """
    layers = []
    net_file = open(path, "r")
    with torch.no_grad():
        curr_line = None
        while True:
            prev_line = curr_line
            curr_line = net_file.readline()[:-1]
            if curr_line in {"Affine", "ReLU", "HardTanh"}:

                if prev_line == "MaxPooling2D":
                    # Make sure to add a flattening operation, so the dimensions match
                    layer = Flatten(start_dim=1)
                    layers.append(layer)

                weight = _parse_np_array_as_tensor(net_file)
                bias   = _parse_np_array_as_tensor(net_file)

                # Add the fully-connected layer.
                layer = Linear(weight.shape[1], weight.shape[0])
                layer.weight[:] = weight
                layer.bias[:] = bias
                layers.append(layer)

                # Maybe add a non-linearity.
                if curr_line == "ReLU":
                    layers.append(ReLU())
                else:
                    raise NotImplementedError(
                        f"unimplemented {curr_line}"
                    )

            elif curr_line.strip() == "":
                break

            elif curr_line.startswith("Conv2D"):

                info_line = net_file.readline()[:-1].strip()
                activation = info_line.split(",")[0]

                input_shape = info_line.split("input_shape=")[1].split("],")[0]
                input_shape = _parse_np_array_as_tensor(input_shape)

                if "stride=" in info_line:
                    stride = _parse_np_array_as_tensor(
                        info_line.split("stride=")[1].split("],")[0] + "]")
                else:
                    stride = 1 # Default.

                pad = (0, 0)
                if "padding=" in info_line:
                    pad = int(info_line.split("padding=")[1])
                    pad = (pad, pad)

                # (f_h, f_w, i_c, o_c)
                filter_weights = _parse_np_array_as_tensor(net_file)
                filter_weights = filter_weights.permute(3, 2, 0, 1).float() # (o_c, i_c, f_h, f_w) (torch style)
                # (o_c,)
                biases = _parse_np_array_as_tensor(net_file).float()

                in_channels = filter_weights.shape[1]
                out_channels = filter_weights.shape[0]
                kernel_size = filter_weights.shape[2:]

                layer = Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size)
                layer.weight.data = filter_weights
                layer.bias.data = biases
                layers.append(layer)

                if activation == 'ReLU':
                    layers.append(ReLU())
                else:
                    raise NotImplementedError

            elif curr_line.startswith('MaxPooling2D'):
                info_line = net_file.readline()[:-1].strip()

                if "stride=" in info_line:
                    stride = _parse_np_array_as_tensor(info_line.split("stride=")[1].split("],")[0] + "]")
                else:
                    stride = None # default.

                if "padding=" in info_line:
                    pad = int(info_line.split("padding=")[1])
                    pad = (pad, pad)
                else:
                    pad = 0

                # tuple(_parse_np_array_as_tensor(info_line.split("pool_size=")[1].split("],")[0] + "]"))
                kernel_size = tuple(ast.literal_eval(info_line.split("pool_size=")[1].split("],")[0] + "]"))

                layer = MaxPool2d(kernel_size=kernel_size, stride=stride, padding=pad)
                layers.append(layer)

            else:
                raise NotImplementedError(
                    f"unimplemented {curr_line.split(' ')[0]}"
                )

    return Sequential(*layers)

def _copy_from_torch_paramater(param):
    if param is None:
        return None

    if isinstance(param, torch.nn.Parameter):
        return torch.nn.Parameter(param.detach().clone())

    elif isinstance(param, torch.Tensor):
        return param.detach().clone()

    else:
        raise NotImplementedError(
            f"unimplemented copy from {type(param)}."
        )

def from_torch(module: torch.nn.Module):
    """ Helper function to convert a torch.nn.Module to a symbolic sytorch.nn.Module """
    import torchvision

    if isinstance(module, torch.nn.Identity):
        return Identity()

    elif isinstance(module, torch.nn.Linear):
        out = Linear(
            in_features  = module.in_features,
            out_features = module.out_features,
            bias         = module.bias is not None
        )
        out.weight = _copy_from_torch_paramater(module.weight)
        out.bias   = _copy_from_torch_paramater(module.bias)
        return out

    elif isinstance(module, torch.nn.Conv2d):
        out = Conv2d(
            in_channels  = module.in_channels,
            out_channels = module.out_channels,
            kernel_size  = module.kernel_size,
            stride       = module.stride,
            padding      = module.padding,
            dilation     = module.dilation,
            groups       = module.groups,
            bias         = module.bias is not None,
            padding_mode = module.padding_mode)
        out.weight = _copy_from_torch_paramater(module.weight)
        out.bias   = _copy_from_torch_paramater(module.bias)
        return out

    elif isinstance(module, torch.nn.ReLU):
        return ReLU(inplace = module.inplace)

    elif isinstance(module, torch.nn.Dropout):
        return Dropout(
            p       = module.p,
            inplace = module.inplace
        )

    elif isinstance(module, torch.nn.MaxPool2d):
        return MaxPool2d(
            kernel_size    = module.kernel_size,
            stride         = module.stride,
            padding        = module.padding,
            dilation       = module.dilation,
            return_indices = module.return_indices,
            ceil_mode      = module.ceil_mode,
        )

    elif isinstance(module, torch.nn.AdaptiveAvgPool2d):
        return AdaptiveAvgPool2d(
            output_size = module.output_size
        )

    elif isinstance(module, torch.nn.BatchNorm2d):
        # return module
        out = BatchNorm2d(
            num_features=module.num_features,
            eps=module.eps,
            momentum=module.momentum,
            affine=module.affine,
            track_running_stats=module.track_running_stats,
        )
        out.weight = _copy_from_torch_paramater(module.weight)
        out.bias   = _copy_from_torch_paramater(module.bias)
        out.running_mean = _copy_from_torch_paramater(module.running_mean)
        out.running_var  = _copy_from_torch_paramater(module.running_var)
        out.num_batches_tracked = _copy_from_torch_paramater(module.num_batches_tracked)
        return out

    elif isinstance(module, torch.nn.Sequential):
        return Sequential(
            *(from_torch(layer) for layer in module)
        )

    elif isinstance(module, torchvision.models.squeezenet.Fire):
        return Sequential(
            from_torch(module.squeeze),
            from_torch(module.squeeze_activation),
            Parallel(
                Sequential(
                    from_torch(module.expand1x1),
                    from_torch(module.expand1x1_activation)
                ),
                Sequential(
                    from_torch(module.expand3x3),
                    from_torch(module.expand3x3_activation)
                ),
                mode = 'cat',
                dim  = 1,
            )
        )

    elif isinstance(module, torchvision.models.squeezenet.SqueezeNet):
        return Sequential(
            from_torch(module.features),
            from_torch(module.classifier),
            Flatten(start_dim=1)
        )

    # Handle `torchvision.models.vgg.VGG`.
    elif isinstance(module, torchvision.models.vgg.VGG):
        return Sequential(
            from_torch(module.features),
            from_torch(module.avgpool),
            Flatten(start_dim=1),
            from_torch(module.classifier)
        )

    # Handle `torchvision.models.resnet.ResNet`.
    elif isinstance(module, torchvision.models.resnet.BasicBlock):
        return Sequential(
            Parallel(
                Sequential(
                    from_torch(module.conv1),
                    from_torch(module.bn1),
                    from_torch(module.relu),

                    from_torch(module.conv2),
                    from_torch(module.bn2),
                ),
                from_torch(module.downsample) if module.downsample is not None else Identity(),
                mode = 'add'
            ),
            ReLU(),
        )

    elif isinstance(module, torchvision.models.resnet.Bottleneck):
        return Sequential(
            Parallel(
                Sequential(
                    from_torch(module.conv1),
                    from_torch(module.bn1),
                    from_torch(module.relu),

                    from_torch(module.conv2),
                    from_torch(module.bn2),
                    from_torch(module.relu),

                    from_torch(module.conv3),
                    from_torch(module.bn3),
                ),
                from_torch(module.downsample) if module.downsample is not None else Identity(),
                mode = 'add'
            ),
            ReLU(),
        )
    elif isinstance(module, torchvision.models.resnet.ResNet):
        return Sequential(
            from_torch(module.conv1),
            from_torch(module.bn1),
            from_torch(module.relu),
            from_torch(module.maxpool),

            from_torch(module.layer1),
            from_torch(module.layer2),
            from_torch(module.layer3),
            from_torch(module.layer4),

            from_torch(module.avgpool),
            Flatten(start_dim=1),
            from_torch(module.fc)
        )

    else:
        raise NotImplementedError(
            f"unsupported torch module {module}."
        )


def from_onnx(path, skip_until=None):
    """
    Modified from https://github.com/95616ARG/indra/blob/5cfbd139745d720dac31854b87efc
    d221f5e620b/SyReNN/eran/frontend/network.py

    Reads a network stored in the ONNX format.
    Note: Concat, AveragePool layers not supported.
    """

    import onnx

    model = onnx.load(path)
    model = onnx.shape_inference.infer_shapes(model)

    layers = []

    skipping = skip_until is not None
    start_index = 0
    if model.graph.node[0].op_type == 'Constant':
        assert model.graph.node[1].op_type == 'Sub'
        assert model.graph.node[2].op_type == 'Constant'
        assert model.graph.node[3].op_type == 'Div'
        import onnx2torch
        _onet = onnx2torch.convert(model, stop_at='Div_0')
        layers.append(
            NormalizeInput(
                mean = getattr(_onet, 'Constant_0').value,
                std  = getattr(_onet, 'Constant_1').value,
            )
        )
        del _onet
        start_index = 4

    for node in model.graph.node[start_index:]:
        if skipping:
            # print(f"skipped {node.name}, {node.op_type}")
            if node.name == skip_until:
                skipping = False
        # if skipping:
        #     print(f"skipped {node.op_type}")
        # print(f"converting {node.op_type}")
        layer = layer_from_onnx(model.graph, node)
        if not layer:
            continue
        layers.append(layer)

    return Sequential(*layers)

# ONNX helper functions:

def onnx_ints_attribute(node, name):
    """
    Taken from https://github.com/95616ARG/indra/blob/5cfbd139745d720dac31854b87efc
    d221f5e620b/SyReNN/eran/frontend/network.py

    Reads int attributes (eg. weight shape) from an ONNX node.
    """
    return next(attribute.ints
                for attribute in node.attribute
                if attribute.name == name)

def layer_from_onnx(graph, node):
    """
    Modified from https://github.com/95616ARG/indra/blob/5cfbd139745d720dac31854b87efc
    d221f5e620b/SyReNN/eran/frontend/network.py

    Reads a layer from an ONNX node.
    Specs for the ONNX operators are available at:
    https://github.com/onnx/onnx/blob/master/docs/Operators.md
    """

    import onnx
    from onnx import numpy_helper
    from onnx import shape_inference

    # First, we get info about inputs to the layer (including previous
    # layer outputs & things like weight matrices).
    inputs = node.input
    deserialized_inputs = []
    deserialized_input_shapes = []
    for input_name in inputs:
        # We need to find the initializers (which I think are basically
        # weight tensors) for the particular input.
        initializers = [init for init in graph.initializer
                        if str(init.name) == str(input_name)]
        if initializers:
            assert len(initializers) == 1
            # Get the weight tensor as a Numpy array and save it.
            deserialized_inputs.append(numpy_helper.to_array(initializers[0]))
        else:
            # This input is the output of another node, so just store the
            # name of that other node (we'll link them up later). Eg.
            # squeezenet0_conv0_fwd.
            deserialized_inputs.append(str(input_name))
        # Get metadata about the input (eg. its shape).
        infos = [info for info in graph.value_info
                    if info.name == input_name]
        if infos:
            # This is an input with a particular shape.
            assert len(infos) == 1
            input_shape = [d.dim_value
                            for d in infos[0].type.tensor_type.shape.dim]
            deserialized_input_shapes.append(input_shape)
        elif input_name == "data":
            # This is an input to the entire network, its handled
            # separately.
            net_input_shape = graph.input[0].type.tensor_type.shape
            input_shape = [d.dim_value for d in net_input_shape.dim]
            deserialized_input_shapes.append(input_shape)
        else:
            # This doesn't have any inputs.
            deserialized_input_shapes.append(None)

    layer = None

    # Standardize some of the data shared by the strided-window layers.
    if node.op_type in {"Conv", "MaxPool", "AveragePool"}:
        # NCHW -> NHWC
        input_shape = deserialized_input_shapes[0]
        input_shape = [input_shape[2], input_shape[3], input_shape[1]]
        strides = list(onnx_ints_attribute(node, "strides"))
        pads = list(onnx_ints_attribute(node, "pads"))
        # We do not support separate begin/end padding.
        assert pads[0] == pads[2]
        assert pads[1] == pads[3]
        pads = pads[1:3]

    # Now, parse the actual layers.
    if node.op_type == "Conv":
        # We don't support dilations or non-1 groups.
        dilations = list(onnx_ints_attribute(node, "dilations"))
        assert all(dilation == 1 for dilation in dilations)
        group = onnx_ints_attribute(node, "group")
        assert not group or group == 1

        # biases are technically optional, but I don't *think* anyone uses
        # that feature.
        assert len(deserialized_inputs) == 3
        input_data, filters, biases = deserialized_inputs

        # OIHW (we don't need to change this since pytorch stores the weights this way.)
        in_channels = filters.shape[1]
        out_channels = filters.shape[0]
        kernel_size = filters.shape[-2:]
        stride = tuple(strides)
        pads = tuple(pads)
        # dilation is by default 1

        layer = Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=pads)
        layer.weight.data = torch.tensor(filters)
        layer.bias.data = torch.tensor(biases)
    elif node.op_type == "Relu":
        layer = ReLU()
    elif node.op_type == "MaxPool":
        kernel_shape = onnx_ints_attribute(node, "kernel_shape")
        layer = MaxPool2d(kernel_shape, strides, padding=pads)
    elif node.op_type == "AveragePool":
        # Not in the sytorch framework right now; we only have
        # AdaptiveAvgPool2D.
        raise NotImplementedError
    elif node.op_type == "Gemm":
        input_data, weights, biases = deserialized_inputs
        alpha = onnx_ints_attribute(node, "alpha")
        if alpha:
            weights *= alpha
        beta = onnx_ints_attribute(node, "beta")
        if beta:
            biases *= beta

        # I omit trans_A, as I get an attribute error.
        # NOTE: This may need to be revisited in the future.
        try:
            trans_A = onnx_ints_attribute(node, "transA")
            assert True == False
        except:
            pass
        trans_B = onnx_ints_attribute(node, "transB")

        # We compute (X . W) [+ C].
        if trans_B:
            weights = weights.transpose()

        layer = Linear(weights.shape[1], weights.shape[0]) # IO seems to be reversed between how it's stored in the filter and the IO ordering.
        layer.weight.data = torch.tensor(weights)
        layer.bias.data = torch.tensor(biases)

    elif node.op_type == "BatchNormalization":
        raise NotImplementedError
    elif node.op_type == "Concat":
        raise NotImplementedError
    elif node.op_type in {"Dropout", "Reshape"}:
        # These are (more-or-less) handled implicitly since we pass around
        # flattened activation vectors and only work with testing.
        layer = False
    elif node.op_type == "Flatten":
        layer = Flatten(start_dim=1)
    else:
        raise NotImplementedError(f"{node}")
    assert len(node.output) == 1
    return layer


""" From Kyle Julian's NNet code (The MIT License):
https://github.com/sisl/NNet/blob/master/utils/readNNet.py
"""

def _readNNet(nnetFile, withNorm=False):
    '''
    Read a .nnet file and return list of weight matrices and bias vectors

    Inputs:
        nnetFile: (string) .nnet file to read
        withNorm: (bool) If true, return normalization parameters

    Returns:
        weights: List of weight matrices for fully connected network
        biases: List of bias vectors for fully connected network
    '''

    # Open NNet file
    f = open(nnetFile,'r')

    # Skip header lines
    line = f.readline()
    while line[:2]=="//":
        line = f.readline()

    # Extract information about network architecture
    record = line.split(',')
    numLayers   = int(record[0])
    inputSize   = int(record[1])

    line = f.readline()
    record = line.split(',')
    layerSizes = np.zeros(numLayers+1,'int')
    for i in range(numLayers+1):
        layerSizes[i]=int(record[i])

    # Skip extra obsolete parameter line
    f.readline()

    # Read the normalization information
    line = f.readline()
    inputMins = [float(x) for x in line.strip().split(",") if x]

    line = f.readline()
    inputMaxes = [float(x) for x in line.strip().split(",") if x]

    line = f.readline()
    means = [float(x) for x in line.strip().split(",") if x]

    line = f.readline()
    ranges = [float(x) for x in line.strip().split(",") if x]

    # Read weights and biases
    weights= []
    biases = []
    for layernum in range(numLayers):

        previousLayerSize = layerSizes[layernum]
        currentLayerSize = layerSizes[layernum+1]
        weights.append([])
        biases.append([])
        weights[layernum] = np.zeros((currentLayerSize,previousLayerSize))
        for i in range(currentLayerSize):
            line=f.readline()
            aux = [float(x) for x in line.strip().split(",")[:-1]]
            for j in range(previousLayerSize):
                weights[layernum][i,j] = aux[j]
        #biases
        biases[layernum] = np.zeros(currentLayerSize)
        for i in range(currentLayerSize):
            line=f.readline()
            x = float(line.strip().split(",")[0])
            biases[layernum][i] = x

    f.close()

    if withNorm:
        return weights, biases, inputMins, inputMaxes, means, ranges
    return weights, biases

def from_nnet(path: str):
    weights, biases, mins, maxes, means, stds = _readNNet(path, withNorm=True)
    mins, maxes, means, stds = tuple(np.array(a) for a in (mins, maxes, means, stds))
    means, stds = means[:-1], stds[:-1]

    def normalize_input(i):
        return ((np.clip(i, mins, maxes) - means) / stds)

    def denormalize_input(i):
        return ((i * stds) + means)

    layers = []
    for weight, bias in zip(weights, biases):
        with torch.no_grad():
            fc = Linear(*weight.shape[::-1])
            fc.weight[:] = torch.from_numpy(weight)
            fc.bias  [:] = torch.from_numpy(bias)
        layers += [fc, ReLU()]

    return Sequential(*layers), normalize_input, denormalize_input

def _create_sample_input(net, sample_input=None):
    if isinstance(sample_input, tuple):
        return torch.zeros(sample_input)

    elif isinstance(sample_input, Tensor):
        return sample_input

    elif hasattr(sample_input, 'shape'):
        return torch.zeros(sample_input.shape)

    assert sample_input is None

    if isinstance(net, Sequential):
        return _create_sample_input(net[0])

    elif isinstance(net, Linear):
        ref = next(net.parameters())
        return torch.randn(
            (1, net.weight.shape[1]),
            dtype=ref.dtype,
            device=ref.device,
            requires_grad=True
        )

    elif isinstance(net, Conv2d):
        ref = next(net.parameters())
        warnings.warn(f"assuming MNIST conv network and using onnx input shape (1,1,28,28).")
        return torch.randn(
            # This shape is not always correct.
            (1, net.weight.shape[1], 28, 28),
            dtype=ref.dtype,
            device=ref.device,
            requires_grad=True
        )

    else:
        raise NotImplementedError(
            f"unimplemented sample input create for {type(net)}"
        )

def to_onnx(net, sample_input=None, file=None):
    # https://learn.microsoft.com/en-us/windows/ai/windows-ml/tutorials/pytorch-convert-model
    # https://stackoverflow.com/questions/68016514/onnx-object-from-pytorch-model-without-exporting
    import onnx

    # for module in net.modules():
    #     assert isinstance(module, (Linear, Conv2d, ReLU, Sequential))

    net = net.to_onnx_compatible()

    import io
    if file is None:
        file = io.BytesIO()

    sample_input = _create_sample_input(net, sample_input=sample_input)

    # print(sample_input.shape)
    training = net.training
    try:
        net.eval()
        torch.onnx.export(
            net,
            sample_input,
            file,
            export_params=True,  # store the trained parameter weights inside the model file
            opset_version=10,    # the ONNX version to export the model to
            do_constant_folding=True,  # whether to execute constant folding for optimization
            input_names = ['modelInput'],   # the model's input names
            output_names = ['modelOutput'], # the model's output names
            dynamic_axes={'modelInput' : {0 : 'batch_size'},    # variable length axes
                        'modelOutput' : {0 : 'batch_size'}}
        )
    finally:
        net.train(training)

    if isinstance(file, io.BytesIO):
        model = onnx.load_model_from_string(file.getvalue())
    else:
        model = onnx.load(file)

    return model

class _with_paths():
    def __init__(self, *paths):
        self.paths = paths

    def __enter__(self):
        import sys
        for path in self.paths:
            sys.path.insert(0, path)

    def __exit__(self, exc_type, exc_value, traceback):
        import sys
        for path in self.paths:
            try:
                sys.path.remove(path)
            except ValueError:
                pass

def from_eran_pyt(path):
    import re

    def runRepl(arg, repl):
        for a in repl:
            arg = arg.replace(a+"=", "'"+a+"':")
        return eval("{"+arg+"}")

    def parseVec(net):
        return np.array(eval(net.readline()[:-1]))

    def myConst(array):
        return torch.from_numpy(array)

    def extract_mean(text):
        mean = ''
        m = re.search('mean=\[(.+?)\]', text)

        if m:
            means = m.group(1)
        mean_str = means.split(',')
        num_means = len(mean_str)
        mean_array = np.zeros(num_means)
        for i in range(num_means):
            mean_array[i] = np.float64(mean_str[i])
        return mean_array

    def extract_std(text):
        std = ''
        m = re.search('std=\[(.+?)\]', text)
        if m:
            stds = m.group(1)
        std_str =stds.split(',')
        num_std = len(std_str)
        std_array = np.zeros(num_std)
        for i in range(num_std):
            std_array[i] = np.float64(std_str[i])
        return std_array

    net = open(path,'r')
    last_layer = None

    layers = []
    while True:
        curr_line = net.readline()[:-1]

        if 'Normalize' in curr_line:
            mean = myConst(extract_mean(curr_line))
            std  = myConst(extract_std(curr_line))
            layers.append(
                NormalizeInput(
                    mean = mean.reshape(1, 3, 1, 1),
                    std  = std.reshape(1, 3, 1, 1),
                )
            )

        elif curr_line in ["ReLU", "Affine"]:
            W = None

            if (last_layer in ["Conv2D"]):
                layers.append(Flatten(start_dim=1))
                W = myConst(parseVec(net).transpose())

            else:
                W = myConst(parseVec(net).transpose())

            I, O = W.shape
            b = myConst(parseVec(net))

            layer = Linear(
                in_features = I,
                out_features= O,
            )
            with torch.no_grad():
                layer.weight[:] = W.T
                layer.bias[:] = b

            layers.append(layer)

            if(curr_line=="Affine"):
                pass

            elif(curr_line=="ReLU"):
                layers.append(ReLU())
                pass

            else:
                raise NotImplementedError

        elif curr_line == "Conv2D":
            line = net.readline()

            start = 0
            if("ReLU" in line):
                start = 5
            elif("Affine" in line):
                start = 7

            if 'padding' in line:
                args = runRepl(line[start:-1], ["filters", "input_shape", "kernel_size", "stride", "padding"])
            else:
                args = runRepl(line[start:-1], ["filters", "input_shape", "kernel_size"])

            _, _, I = args['input_shape']
            O = args['filters']
            W, H = args['kernel_size']
            padding = args.get('padding', 0)

            layer = Conv2d(
                in_channels  = I,
                out_channels = O,
                kernel_size  = (W, H),
                stride       = tuple(args['stride']),
                padding      = padding,
                padding_mode = 'zeros' if padding >= 1 else 'zeros',
            )

            with torch.no_grad():
                W = myConst(parseVec(net)).permute(3,2,0,1)
                b = myConst(parseVec(net))
                assert W.shape == layer.weight.shape
                assert b.shape == layer.bias.shape
                layer.weight[:] = W
                layer.bias  [:] = b

            layers.append(layer)

            if("ReLU" in line):
                layers.append(ReLU())
            else:
                raise Exception("Unsupported activation: ", curr_line)

        elif curr_line == "":
            break

        else:
            raise Exception("Unsupported Operation: ", curr_line)

        last_layer = curr_line

    return Sequential(*layers)
