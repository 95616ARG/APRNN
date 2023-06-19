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
                    choices=['3x100'],
                    help='Networks to repair in experiments (if applicable).')
parser.add_argument('--ndims', type=int, dest='ndims', action='store', required=True,
                    help='Number of pixels or groups.')
parser.add_argument('--k', type=int, dest='k', action='store',
                    default=0,
                    help='k.')
parser.add_argument('--seed', type=int, dest='seed', action='store',
                    default=-1,
                    help='seed.')
parser.add_argument('--img_seed', type=int, dest='img_seed', action='store',
                    default=-1,
                    help='img_seed.')
parser.add_argument('--pick', type=str, dest='pick', action='store',
                    default='grouped_block',
                    choices=['leading', 'center', 'nonzero', 'random', 'grouped', 'grouped_block', 'grouped_row'],
                    help='pick.')
parser.add_argument('--eps', type=float, dest='eps', action='store', required=True,
                    help='epsilon for L^\infty norm.')
parser.add_argument('--device', type=str, dest='device', action='store', default='cpu',
                    help='device to use, e.g., cuda, cuda:0, cpu. (default=cpu).')

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
images, labels = buggyset.shuffle(args.img_seed).load(1)
print(images, labels)

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
    vpolytopes = st.points_to_vboxes(images.reshape(-1, 784), size=args.eps, groups=dim_groups)[None]
    inner_vpolytopes = st.points_to_vboxes(images.reshape(-1, 784), size=args.eps/2., groups=dim_groups)[None]

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
    vpolytopes = st.points_to_vboxes(images.reshape(-1, 784), size=args.eps, groups=dim_groups)[None]
    inner_vpolytopes = st.points_to_vboxes(images.reshape(-1, 784), size=args.eps/2., groups=dim_groups)[None]

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
    vpolytopes = st.points_to_vboxes(images.reshape(-1, 784), size=args.eps, groups=dim_groups)[None]
    inner_vpolytopes = st.points_to_vboxes(images.reshape(-1, 784), size=args.eps/2., groups=dim_groups)[None]

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
    print("dims:", dims)

    vpolytopes = st.points_to_vboxes(images.reshape(-1, 784), size=args.eps, dims=dims)[None]
    inner_vpolytopes = st.points_to_vboxes(images.reshape(-1, 784), size=args.eps/2., dims=dims)[None]

print(args)

images = images.to(device=device,dtype=dtype)

# vpolytopes = vpolytopes.reshape(*vpolytopes.shape[:2], 1, 28, 28)
vpolytopes = vpolytopes.to(device,dtype)
print(vpolytopes.shape)

vpolytope_labels = torch.broadcast_to(labels, (*vpolytopes.shape[:-1], 1))
print(vpolytope_labels.shape)

def facets(ndim):
    # assert vbox.shape[0] == 2**ndim
    vind = np.arange(2**ndim, dtype=int).reshape((2,)*ndim)
    for dim in range(ndim):
        facet_idx = [st.as_slice[:],]*ndim
        facet_idx[dim] = st.as_slice[0]
        yield vind[tuple(facet_idx)].flatten()
        facet_idx[dim] = st.as_slice[1]
        yield vind[tuple(facet_idx)].flatten()

def partition_by_facets(dnn, ndim, vboxes, centers, flatten=True):
    assert vboxes.shape[0] == 1
    # centers = ...
    facet_indices = facets(ndim)
    vboxes_aps = np.empty(vboxes.shape[:2], dtype=object)
    # vboxes_og_aps = dnn.activation_pattern(vboxes.flatten(0,1)).reshape(vboxes.shape[:2])
    for vbox, center, aps in zip(vboxes, centers, vboxes_aps):
        for facet_idx in facet_indices:
            facet = vbox[facet_idx]
            # facet_og_aps = og_aps[facet_idx]
            # facet_aps = aps[facet_idx]

            facet_ref = (facet.mean(0) + center) / 2.
            facet_ref_ap = dnn.activation_pattern(facet_ref[None])

            for idx in facet_idx:
                if aps[idx] is None:
                    aps[idx] = facet_ref_ap
                else:
                    aps[idx] = st.meet_patterns(aps[idx], facet_ref_ap)


    out_ap = []
    example_ap = vboxes_aps.item(0)
    for i in range(len(example_ap)):
        if example_ap[i] == []:
            out_ap.append([])
        else:
            out_ap.append(
                np.concatenate([
                    ap[i]
                    for ap in vboxes_aps.reshape(-1, *vboxes_aps.shape[2:])
                ], 0)
            )
            if not flatten:
                out_ap[-1] = out_ap[-1].reshape(
                    *vboxes_aps.shape[:2], *out_ap[-1].shape[1:]
                )

    return out_ap

def foo(N, vpolytopes, lb=-3., ub=3., centers=None):
    N.requires_symbolic_(False).to(None).repair(False)
    solver = st.GurobiSolver()
    solver.solver.Params.BarConvTol = 1e-2
    N.to(solver).repair().requires_symbolic_weight_and_bias(lb=lb, ub=ub)
    if centers is None:
        sy_vpolytopes = N.v(vpolytopes)
    else:
        print('partition facets')
        vertices = vpolytopes.flatten(0, 1)
        vertices_ap = partition_by_facets(N, args.ndims, vpolytopes, centers)
        # import pdb; pdb.set_trace()
        sy_vertices = N(vertices, pattern=vertices_ap)
        sy_vpolytopes = sy_vertices.reshape(*vpolytopes.shape[:2], *sy_vertices.shape[1:])
    return sy_vpolytopes, solver

def vpoly_repair(dnn, vpolytopes, labels, s, k, lb=-3., ub=3., Method=-1, centers=None):
    N = dnn.deepcopy()
    for l0, l1 in s:
        if l0 == l1:
            continue
        N.requires_symbolic_(False).to(None).repair(False)
        if centers is not None:
            centers_upto_here = N[:l0](centers)
        else:
            centers_upto_here = None
        sy_vpolytopes, solver = foo(N[l0:l1], N[:l0].v(vpolytopes), lb=lb, ub=ub, centers=centers_upto_here)
        param_delta = N[l0:l1].parameter_deltas(concat=True)
        output_delta = (sy_vpolytopes - dnn[:l1].v(vpolytopes)).reshape(-1)
        delta = param_delta.norm_ub('linf+l1_normalized') + output_delta.norm_ub('linf+l1_normalized')
        assert solver.solve(minimize=delta, Method=Method)
        N[l0:l1].update_().requires_symbolic_(False).to(None).repair(False)

    N.requires_symbolic_(False).to(None).repair(False)
    if centers is not None:
        centers_upto_here = N[:k](centers)
    else:
        centers_upto_here = None
    sy_vpolytopes, solver = foo(N[k:], N[:k].v(vpolytopes), lb=lb, ub=ub, centers=centers_upto_here)
    param_delta = N[k:].parameter_deltas(concat=True)
    output_delta = (sy_vpolytopes - dnn.v(vpolytopes)).reshape(-1)
    delta = param_delta.norm_ub('linf+l1_normalized') + output_delta.norm_ub('linf+l1_normalized')
    assert solver.solve(sy_vpolytopes.argmax(-1) == labels, minimize=delta, Method=Method)
    N[k:].update_().requires_symbolic_(False).to(None).repair(False)

    return N

s = [(0, args.k)]
k = args.k

start = timer()

N = vpoly_repair(dnn, vpolytopes, vpolytope_labels, s=s, k=k, lb=-10., ub=10., Method=2,
                # centers=images
    )

time = timer() - start

result_path = (get_results_root() / 'eval_7' / f'aprnn_{args.net}_ndims={args.ndims}_eps={args.eps}_pick={args.pick}_k={k}_seed={args.seed}').as_posix()
N.save(
    (get_results_root() / 'eval_7' / f'aprnn_{args.net}_ndims={args.ndims}_eps={args.eps}_pick={args.pick}_k={k}_seed={args.seed}.pth').as_posix()
)
print(f"APRNN Time: {int(time)}")


acc0 = testset.accuracy(dnn)
acc1 = testset.accuracy(N)
D = acc0 - acc1
print(f"APRNN Drawdown: {D:.2%} ({acc0:.2%} -> {acc1:.2%}).")
Dstr = f"{D:.2%} ({acc0:.2%} -> {acc1:.2%})"

result = {
    'APRNN': {
        'args': vars(args),
        'D' : Dstr,
        'T': 'N/A' if time is None else f'{int(time)}',
    }
}

# print(result)

np.save(result_path+".npy", result, allow_pickle=True)

# print_msg_box(
#     f"Experiment 7 using APRNN SUCCEED.\n"
#     f"Saved result to {result_path}.npy"
# )
