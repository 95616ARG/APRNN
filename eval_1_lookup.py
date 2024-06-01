import warnings; warnings.filterwarnings("ignore")

from experiments import mnist
from experiments.base import *
import sytorch as st
import argparse
from timeit import default_timer as timer

parser = argparse.ArgumentParser(description='')
parser.add_argument('--net', type=str, dest='net', action='store', required=True,
                    help='3x100, 9x100 or 9x200')
parser.add_argument('--device', type=str, dest='device', action='store', default='cpu',
                    help='device to use, e.g., cuda, cuda:0, cpu. (default=cpu).')
parser.add_argument('--use_artifact', dest='use_artifact', action='store_true',
                    help='use authors\' repaired DNN.')
args = parser.parse_args()

device = get_device(args.device)
dtype = st.float64

k = {
    '3x100': 2,
    '9x100': 10,
    '9x200': 12,
}[args.net]

n_points = 100
network = mnist.model(args.net).to(dtype=dtype, device=device)

corruption = 'fog'
GeneralizationSet = mnist.GeneralizationSet(corruption)[n_points:].reshape(784)
DrawdownSet = mnist.DrawdownSet().reshape(784)
RepairSet = mnist.RepairSet(corruption).reshape(784)

class LookupTable(st.nn.Module):
    def __init__(self, dnn, points, labels):
        super().__init__()
        self.dnn = dnn
        self.table = {
            tuple(point.tolist()): int(label)
            for point, label in zip(points, labels.flatten())
        }

    def lookup(self, point):
        return self.table.get(tuple(point.tolist()), None)

    def forward(self, inputs):
        outputs = self.dnn(inputs)
        for i, point in enumerate(inputs):
            label = self.lookup(point)
            if label is not None:
                outputs[i] = label
        return outputs

if not args.use_artifact:

    images, labels = RepairSet.load(n_points)
    images = images.to(dtype=dtype, device=device)

    start = timer()
    N = LookupTable(network, images, labels)
    time = timer() - start

    result_path = (get_results_root() / 'eval_1' / f'lookup_{args.net}').as_posix()

else:
    N = network.deepcopy()
    N.load((get_artifact_root() / 'eval_1' / f'lookup_{args.net}.pth'))
    time = None

    result_path = (get_results_root() / 'eval_1' / f'artifact_lookup_{args.net}').as_posix()

d0 = DrawdownSet.accuracy(network)

start = timer()
d1 = DrawdownSet.accuracy(N)
time_d = timer() - start

g0 = GeneralizationSet.accuracy(network)
g1 = GeneralizationSet.accuracy(N)

result = {
    args.net: {
        ('LT', 'D'): d0 - d1,
        ('LT', 'G'): g1 - g0,
        ('LT', 'T'): 'N/A' if time is None else f'{int(time)}s',
    }
}

np.save(result_path+".npy", result, allow_pickle=True)

print_msg_box(
    f"Experiment 1 for MNIST {args.net} using Lookup SUCCEED.\n"
    f"Saved result to {result_path}.npy"
)