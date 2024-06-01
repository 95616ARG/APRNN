import warnings; warnings.filterwarnings("ignore")

from experiments import mnist
from experiments.base import *
import sytorch as st
import numpy as np
import sys, argparse
from timeit import default_timer as timer

parser = argparse.ArgumentParser(description='')
parser.add_argument('--net', '-n', type=str, dest='net', action='store',
                    default='3x100',
                    choices=['3x100', '9x100', '9x200'],
                    help='Networks to repair in experiments (if applicable).')
parser.add_argument('--ndims', type=int, dest='ndims', action='store', required=True,
                    help='Number of pixels from the center 4x4.')
parser.add_argument('--num', type=int, dest='num', action='store', required=True,
                    help='num points.')
parser.add_argument('--k', type=int, dest='k', action='store',
                    default=0,
                    help='k.')
parser.add_argument('--seed', type=int, dest='seed', action='store',
                    default=-1,
                    help='seed.')
parser.add_argument('--pick', type=str, dest='pick', action='store',
                    default='grouped_block',
                    choices=['leading', 'center', 'nonzero', 'random', 'grouped', 'grouped_block', 'grouped_row'],
                    help='pick.')
parser.add_argument('--eps', type=float, dest='eps', action='store', required=True,
                    help='epsilon for L^\infty norm.')
parser.add_argument('--device', type=str, dest='device', action='store', default='cpu',
                    help='device to use, e.g., cuda, cuda:0, cpu. (default=cpu).')
parser.add_argument('--use_artifact', dest='use_artifact', action='store_true',
                    help='use authors\' repaired DNN.')

args = parser.parse_args()

device = get_device(args.device)
dtype = st.float64

dnn = mnist.model(args.net).to(device,dtype)

def crop_center(img,cropx,cropy):
    y,x = img.shape
    startx = x//2-(cropx//2)
    starty = y//2-(cropy//2)
    return img[starty:starty+cropy,startx:startx+cropx]

testset = mnist.datasets.Dataset('identity', 'test').reshape(784).to(device,dtype)
correctset, buggyset = testset.filter_misclassified(dnn)
images, labels = buggyset.load(args.num)
print(labels)

if args.pick == 'grouped':
    pixel_indices = np.arange(784, dtype=int)
    # np.random.default_rng(args.seed).shuffle(pixel_indices)
    if args.seed >= 0:
        print(f"shuffle with seed {args.seed}")
        np.random.default_rng(args.seed).shuffle(pixel_indices)
    dim_groups = np.array_split(pixel_indices, args.ndims)
    print("dim groups:", [
        np.sort(arr)
        for arr in dim_groups
    ])
    vpolytopes = st.points_to_vboxes(images.reshape(-1, 784), size=args.eps, groups=dim_groups, flatten=False)

elif args.pick == 'grouped_row':
    a = np.arange(28*28,dtype=int).reshape(28,28)
    b = list(np.ndindex(4,4))

    pixel_indices = np.arange(28, dtype=int)
    if args.seed >= 0:
        print(f"shuffle with seed {args.seed}")
        np.random.default_rng(args.seed).shuffle(pixel_indices)

    dim_groups = np.array_split(pixel_indices, args.ndims)

    dim_groups = [
        np.concatenate(
            [
                # a[b[idx][0]*7:b[idx][0]*7+7, b[idx][1]*7:b[idx][1]*7+7]
                a[idx]
                for idx in arr
            ]
        ).reshape(-1)
        for arr in dim_groups
    ]

    print("dim groups:", [
        np.sort(arr)
        for arr in dim_groups
    ])
    assert np.unique(np.concatenate(dim_groups)).size == 784
    vpolytopes = st.points_to_vboxes(images.reshape(-1, 784), size=args.eps, groups=dim_groups, flatten=False)

elif args.pick == 'grouped_block':
    a = np.arange(28*28,dtype=int).reshape(28,28)
    b = list(np.ndindex(4,4))

    pixel_indices = np.arange(4*4, dtype=int)
    if args.seed >= 0:
        print(f"shuffle with seed {args.seed}")
        np.random.default_rng(args.seed).shuffle(pixel_indices)

    dim_groups = np.array_split(pixel_indices, args.ndims)

    dim_groups = [
        np.concatenate(
            [
                a[b[idx][0]*7:b[idx][0]*7+7, b[idx][1]*7:b[idx][1]*7+7]
                for idx in arr
            ]
        ).reshape(-1)
        for arr in dim_groups
    ]

    print("dim groups:", [
        np.sort(arr)
        for arr in dim_groups
    ])
    vpolytopes = st.points_to_vboxes(images.reshape(-1, 784), size=args.eps, groups=dim_groups, flatten=False)

else:
    if args.pick == 'leading':
        pixel_indices = list(range(args.ndims))

    elif args.pick == 'center':
        pixel_indices = crop_center(np.arange(784, dtype=int).reshape(28,28), 4, 4).reshape(-1)

    elif args.pick == 'nonzero':
        pixel_indices = st.where(images[0] != 0.)[0]

    elif args.pick == 'random':
        pixel_indices = np.random.default_rng(args.seed if args.seed != -1 else 0).choice(784, args.ndims, replace=False)

    else:
        raise NotImplementedError(f"{args.pick}")

    dims = pixel_indices[:args.ndims]

    vpolytopes = st.points_to_vboxes(images.reshape(-1, 784), size=args.eps, dims=dims, flatten=False)


images = images.to(device=device,dtype=dtype)

vpolytopes = vpolytopes.to(device,dtype)

time = 0
result_path = (get_results_root() / 'eval_8' / f'lookup_{args.net}').as_posix()

import os
class Lookup(st.nn.Module):
    def __init__(self, dnn, vpolytopes, labels):
        super().__init__()
        print(vpolytopes.shape, labels.shape)
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
            for vpoly, label in zip(self.vpolytopes, self.labels.squeeze(-1)):
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



N = Lookup(dnn.deepcopy(), vpolytopes, labels)

acc0 = testset.accuracy(dnn)

start = timer()
acc1 = testset.accuracy(N, batch_size=10000)
time_d = timer() - start

D = acc0 - acc1
print(f"lookup repair time: {timd_d} s.")
print(f"lookup Drawdown: {D:.2%} ({acc0:.2%} -> {acc1:.2%})")
