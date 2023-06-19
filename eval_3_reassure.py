import sys, os
import torch, torchvision, time, numpy as np
from REASSURE.tools.models import *
from REASSURE.tools.build_PNN import *
from REASSURE.exp_tools import *
import warnings
warnings.filterwarnings("ignore")

from experiments import mnist
import sytorch
import torch, time
import sytorch as st
from timeit import default_timer as timer
import sys, argparse, pathlib

parser = argparse.ArgumentParser(description='')
parser.add_argument('--net', '-n', type=str, dest='net', action='store',
                    default='3x100',
                    choices=['9x100'],
                    help='Networks to repair in experiments (if applicable).')
parser.add_argument('--num', type=int, dest='num', action='store', default=100,
                    help='device to use, e.g., cuda, cuda:0, cpu. (default=cpu).')
parser.add_argument('--device', type=str, dest='device', action='store', default='cuda:1',
                    help='device to use, e.g., cuda, cuda:0, cpu. (default=cpu).')
parser.add_argument('--use_artifact', dest='use_artifact', action='store_true',
                    help='use authors\' repaired DNN.')

args = parser.parse_args()

netname = args.net

def PNN_MNIST(repair_num, n, num_core, remove_redundant_constraint=False):
    torch.manual_seed(0)

    device = args.device
    dtype = st.float64

    npolys = 5

    network = mnist.model(netname).to(device,dtype)
    print("params: ", sum([p.numel() for p in network.parameters()]))

    testset = mnist.datasets.Dataset('identity', 'test').reshape(784).to(device,dtype)
    mnist_c = mnist.datasets.MNIST_C(corruption='fog', split='test').reshape(784).to(device,dtype)

    _, rotset0 = mnist_c.filter_label(8).misclassified(network)
    rotset = rotset0.rotate(degs=np.linspace(-30., 30., 7), scale=1.)
    g1 = rotset[npolys:]
    g2 = rotset0.rotate(degs=np.arange(-30., 30.1, .1), scale=1.)[:npolys]

    device = st.device('cpu')
    dtype = st.float32

    nlayers, width = tuple(map(int, netname.split('x')))
    target_model = MLPNet([28*28] + [width] * (nlayers-1) + [10])
    network = mnist.model(netname).to(device=device,dtype=dtype)
    N = network.deepcopy()
    print(network)
    print(target_model.layers)
    for l in range(len(target_model.layers)):
        target_model.layers[l] = N[l*2]
    print(target_model.layers)

    bounds=[torch.zeros(28*28), torch.ones(28*28)]

    # target_model.load_state_dict(torch.load('Experiments/MNIST/target_model.pt'))

    test_dataloader = torch.utils.data.DataLoader(torchvision.datasets.MNIST(root='../../data', train=False,
                                            download=True, transform=torchvision.transforms.ToTensor()), batch_size=64)
    train_dataloader = torch.utils.data.DataLoader(torchvision.datasets.MNIST(root='../../data', train=True,
                                            download=True, transform=torchvision.transforms.ToTensor()), batch_size=64,
                                            shuffle=True)

    images = st.from_numpy(np.load('data/eval_3_reassure_100_uniq_ap_buggy_points.npy')).flatten(0,1)
    labels = st.from_numpy(np.load('data/eval_3_reassure_100_uniq_ap_buggy_labels.npy')).flatten(0,1)
    images = images.to(dtype=dtype, device=device)

    buggy_inputs = images
    right_label = labels

    acc = test_acc(test_dataloader, target_model)
    print(f'Test Acc after before: {acc*100}%')
    assert (buggy_inputs.shape[0] >= repair_num)
    buggy_inputs, right_label = buggy_inputs[:repair_num], right_label[:repair_num]

    print(buggy_inputs.shape)

    print(
        "buggy acc: ",
        (target_model(buggy_inputs).argmax(-1).flatten() == right_label.flatten()).float().mean()
    )

    savename = f'PNNed_model_new_eval_3_{netname}_{repair_num}p.pt'
    print(savename)
    if args.use_artifact and pathlib.Path(savename).exists():
        repaired_model = torch.load(savename)

    else:

        P, ql, qu = specification_matrix_from_labels(right_label)

        start = timer()
        PNN = MultiPointsPNN(target_model, n, bounds=bounds)
        PNN.point_wise_repair(buggy_inputs, P, ql, qu, remove_redundant_constraint=remove_redundant_constraint)
        repaired_model = PNN.compute(num_core, is_gurobi=True)
        cost_time = timer()-start
        print('cost time:', cost_time)

        torch.save(repaired_model, savename)

    print("params: ", sum([p.numel() for p in repaired_model.parameters()]), flush=True)

    # otherwise runs out of memory
    # device = st.device('cuda:1')
    device = g1.device
    dtype = st.float32
    network = mnist.model(netname).to(device,dtype)
    N = repaired_model.to(device,dtype)
    g1.to(device,dtype)
    g2.to(device,dtype)
    testset.to(device,dtype)
    batch_size = 1

    #
    acc0 = g1.accuracy(network)

    start = timer()
    acc1 = g1.accuracy(N, batch_size=batch_size)
    time_g1 = timer() - start

    G1 = acc1 - acc0
    print(f"REASSURE Generalization on S1: {G1:.2%} ({acc0:.2%} -> {acc1:.2%}) ({time_g1}s).", flush=True)
    G1str = f"{G1:.2%} ({acc0:.2%} -> {acc1:.2%})"

    #
    acc0 = g2.accuracy(network)

    start = timer()
    acc1 = g2.accuracy(N, batch_size=batch_size)
    time_g2 = timer() - start

    G2 = acc1 - acc0
    print(f"REASSURE Generalization on S2: {G2:.2%} ({acc0:.2%} -> {acc1:.2%}) ({time_g2}s).", flush=True)
    G2str = f"{G2:.2%} ({acc0:.2%} -> {acc1:.2%})"

    #
    print(
        "buggy acc: ",
        (repaired_model(buggy_inputs.to(device,dtype)).argmax(-1).flatten().cpu() == right_label.flatten()).float().mean(), flush=True
    )

    #
    acc0 = testset.accuracy(network)

    start = timer()
    acc1 = testset.accuracy(N, batch_size=batch_size)
    time_d = timer() - start

    D = acc0 - acc1
    print(f"REASSURE Drawdown: {D:.2%} ({acc0:.2%} -> {acc1:.2%}) ({time_d}s).", flush=True)
    Dstr = f"{D:.2%} ({acc0:.2%} -> {acc1:.2%})"


if __name__ == '__main__':
    PNN_MNIST(args.num, 0.5, num_core=os.cpu_count(), remove_redundant_constraint=False)
