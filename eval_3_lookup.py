import warnings; warnings.filterwarnings("ignore")

from experiments import mnist
from experiments.base import *
import sytorch as st
import numpy as np
import sys, argparse
from timeit import default_timer as timer

parser = argparse.ArgumentParser(description='')
parser.add_argument('--net', '-n', type=str, dest='net', action='store',
                    default='9x100',
                    choices=['9x100'],
                    help='Networks to repair in experiments (if applicable).')
parser.add_argument('--device', type=str, dest='device', action='store', default='cpu',
                    help='device to use, e.g., cuda, cuda:0, cpu. (default=cpu).')

args = parser.parse_args()

device = get_device(args.device)
dtype = st.float64

npolys = 5

dnn = mnist.model('9x100').to(device,dtype)

testset = mnist.datasets.Dataset('identity', 'test').reshape(784).to(device,dtype)
mnist_c = mnist.datasets.MNIST_C(corruption='fog', split='test').reshape(784).to(device,dtype)

_, rotset0 = mnist_c.filter_label(8).misclassified(dnn)
rotset = rotset0.rotate(degs=np.linspace(-30., 30., 7), scale=1.)
generalization_set = rotset[npolys:]
g2 = rotset0.rotate(degs=np.arange(-30., 30.1, .1), scale=1.)[:npolys]

imgs, labels = rotset.load(npolys)
imgs = imgs.to(device,dtype)

time = 0
result_path = (get_results_root() / 'eval_3' / f'lookup_{args.net}').as_posix()

# import multiprocessing as mp
import os
class Lookup(st.nn.Module):
    def __init__(self, dnn, vpolytopes, labels):
        super().__init__()
        assert tuple(labels.shape) == (vpolytopes.shape[0], 1)
        self.dnn = dnn
        self.vpolytopes = vpolytopes.detach().cpu().numpy().copy()
        self.labels = labels.detach().cpu().numpy().copy()

    def forward(self, inputs):
        outputs = self.dnn(inputs)

        return self.solve_in_parallel(inputs.detach().cpu().numpy().copy(), outputs)

    def solve_in_parallel(self, points, outputs):
        executor = st.ProcessPoolExecutor(os.cpu_count())
        def foo_closure(i):
            for vpoly, label in zip(self.vpolytopes, self.labels.flat):
                if st.isin_vpoly(points[i], vpoly):
                    return int(label)
            return -1
        greg = st.GlobalRegister(globals(), foo_closure)
        greg.register()
        future = executor.map(
            foo_closure, range(points.shape[0])
        )
        from tqdm.auto import tqdm
        for i, label in tqdm(
            enumerate(future),
            desc='lookup points in parallel',
            total=points.shape[0],
            leave=False
        ):
            if label >= 0:
                outputs[i] = 0.
                outputs[i, label] = 1.
        executor.shutdown(wait=True)
        greg.unregister()
        return outputs


N = Lookup(dnn.deepcopy(), imgs, labels[:,0])

acc0 = testset.accuracy(dnn)

start = timer()
acc1 = testset.accuracy(N, batch_size=1000)
time_d = timer() - start

D = acc0 - acc1
print(f"lookup Drawdown: {D:.2%} ({acc0:.2%} -> {acc1:.2%}) ({int(time_d)}s).")
Dstr = f"{D:.2%} ({acc0:.2%} -> {acc1:.2%})"


acc0 = generalization_set.accuracy(dnn)

start = timer()
acc1 = generalization_set.accuracy(N)
time_g1 = timer() - start

G1 = acc1 - acc0
print(f"lookup Generalization on S1: {G1:.2%} ({acc0:.2%} -> {acc1:.2%}) ({int(time_g1)}s).")
G1str = f"{G1:.2%} ({acc0:.2%} -> {acc1:.2%})"

acc0 = g2.accuracy(dnn)

start = timer()
acc1 = g2.accuracy(N)
time_g2 = timer() - start

G2 = acc1 - acc0
print(f"lookup Generalization on S2: {G2:.2%} ({acc0:.2%} -> {acc1:.2%}) ({int(time_g2)}s).")
G2str = f"{G2:.2%} ({acc0:.2%} -> {acc1:.2%})"

result = {
    'LOOKUP': {
        'D' : Dstr,
        'G1': G1str,
        'G2': G2str,
        'T': 'N/A' if time is None else f'{int(time)}',
    }
}

np.save(result_path+".npy", result, allow_pickle=True)

print_msg_box(
    f"Experiment 3 using lookup SUCCEED.\n"
    f"Saved result to {result_path}.npy"
)
