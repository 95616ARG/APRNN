import torch, torchvision, numpy as np
import torch.nn as nn
import torch.nn.functional as F


class MLPNet(nn.Module):
    def __init__(self, size):
        super(MLPNet, self).__init__()
        self.size = size
        self.layers = nn.ModuleList()
        for i in range(len(size)-1):
            self.layers.append(nn.Linear(size[i], size[i+1]))

    def forward(self, x):
        x = x.view(-1, self.size[0])
        for layer in self.layers:
            if layer is self.layers[-1]:
                x = layer(x)
            else:
                x = F.relu(layer(x))
        return x

    def all_hidden_neurons(self, x):
        hidden_neurons = []
        x = x.view(self.size[0])
        for layer in self.layers[:-1]:
            if layer is self.layers[0]:
                x = layer(x)
            else:
                x = layer(F.relu(x))
            hidden_neurons.append(x)
        return torch.cat(hidden_neurons, dim=-1)

    def activation_pattern(self, x):
        x_activation_pattern = self.all_hidden_neurons(x) > 0
        return [entry.item() for entry in x_activation_pattern]


class AlexClassifier(MLPNet):
    def __init__(self, train_labels):
        model = torchvision.models.alexnet(pretrained=True).classifier
        super(AlexClassifier, self).__init__([4096, 4096, 1000, len(set(train_labels))])

        sorted_labels = sorted(set(train_labels))
        final_weights = np.zeros((1000, len(sorted_labels)))
        final_biases = np.zeros(len(sorted_labels))

        for new_label, old_label in enumerate(sorted_labels):
            final_weights[old_label, new_label] = 1.
        self.layers[0] = model[4]
        self.layers[1] = model[6]
        self.layers[2].weight, self.layers[2].bias = torch.nn.Parameter(torch.from_numpy(final_weights).permute([1, 0]).float()),\
                                                     torch.nn.Parameter(torch.from_numpy(final_biases).float())
        del model





