import sytorch as st
import torchvision

def _deflat_sequeezenet(net):
    layers = []
    for layer in net[0]:
        if isinstance(layer, st.nn.Sequential):
            for m in layer:
                layers.append(m)
        else:
            layers.append(layer)

    for layer in net[1]:
            layers.append(layer)
    layers.append(net[2])
    return st.nn.Sequential(*layers)

def squeezenet1_0(pretrained=True, eval=True, flatten=False):
    network = torchvision.models.squeezenet1_1(pretrained=pretrained).train(mode=not eval)
    network = st.nn.from_torch(network)
    # network.propagate_prefix()
    if flatten:
        network = _deflat_sequeezenet(network)
    return network

def squeezenet1_1(pretrained=True, eval=True, flatten=False):
    network = torchvision.models.squeezenet1_1(pretrained=pretrained).train(mode=not eval)
    network = st.nn.from_torch(network)
    if flatten:
        network = _deflat_sequeezenet(network)
    return network

def vgg11(pretrained=True, eval=True, flatten=False):
    network = torchvision.models.vgg11(pretrained=pretrained).train(mode=not eval)
    network = st.nn.from_torch(network)
    if flatten:
        network = st.nn.Sequential(
            *network[0],
             network[1],
             network[2],
            *network[3],
        )
    return network

def vgg11_bn(pretrained=True, eval=True):
    network = torchvision.models.vgg11_bn(pretrained=pretrained).train(mode=not eval)
    network = st.nn.from_torch(network)
    return network

def vgg13(pretrained=True, eval=True):
    network = torchvision.models.vgg13(pretrained=pretrained).train(mode=not eval)
    network = st.nn.from_torch(network)
    return network

def vgg13_bn(pretrained=True, eval=True):
    network = torchvision.models.vgg13_bn(pretrained=pretrained).train(mode=not eval)
    network = st.nn.from_torch(network)
    return network

def vgg16(pretrained=True, eval=True):
    network = torchvision.models.vgg16(pretrained=pretrained).train(mode=not eval)
    network = st.nn.from_torch(network)
    return network

def vgg16_bn(pretrained=True, eval=True):
    network = torchvision.models.vgg16_bn(pretrained=pretrained).train(mode=not eval)
    network = st.nn.from_torch(network)
    return network

def vgg19(pretrained=True, eval=True):
    network = torchvision.models.vgg19(pretrained=pretrained).train(mode=not eval)
    network = st.nn.from_torch(network)
    return network

def vgg19_bn(pretrained=True, eval=True):
    network = torchvision.models.vgg19_bn(pretrained=pretrained).train(mode=not eval)
    network = st.nn.from_torch(network)
    return network

def resnet18(pretrained=True, eval=True):
    network = torchvision.models.resnet18(pretrained=pretrained).train(mode=not eval)
    network = st.nn.from_torch(network)
    return network

def resnet34(pretrained=True, eval=True):
    network = torchvision.models.resnet34(pretrained=pretrained).train(mode=not eval)
    network = st.nn.from_torch(network)
    return network

def resnet50(pretrained=True, eval=True):
    network = torchvision.models.resnet50(pretrained=pretrained).train(mode=not eval)
    network = st.nn.from_torch(network)
    return network

def resnet101(pretrained=True, eval=True):
    network = torchvision.models.resnet101(pretrained=pretrained).train(mode=not eval)
    network = st.nn.from_torch(network)
    return network

def resnet152(pretrained=True, eval=True):
    network = torchvision.models.resnet152(pretrained=pretrained).train(mode=not eval)
    network = st.nn.from_torch(network)
    return network
