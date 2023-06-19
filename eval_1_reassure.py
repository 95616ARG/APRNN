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
                    choices=['3x100', '9x100', '9x200'],
                    help='Networks to repair in experiments (if applicable).')
parser.add_argument('--num', type=int, dest='num', action='store', required=True,
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
    dtype = st.float32

    target_model = MLPNet([28*28] + [100] * 2 + [10])
    network = mnist.model(netname).to(device=device,dtype=dtype)

    print(network)
    print("params: ", sum([p.numel() for p in network.parameters()]))

    N = network.deepcopy()
    print(network)
    print(target_model.layers)
    for l in range(len(target_model.layers)):
        target_model.layers[l] = N[l*2]
    print(target_model.layers)

    bounds=[torch.zeros(28*28), torch.ones(28*28)]

    test_dataloader = torch.utils.data.DataLoader(torchvision.datasets.MNIST(root='../../data', train=False,
                                            download=True, transform=torchvision.transforms.ToTensor()), batch_size=64)
    train_dataloader = torch.utils.data.DataLoader(torchvision.datasets.MNIST(root='../../data', train=True,
                                            download=True, transform=torchvision.transforms.ToTensor()), batch_size=64,
                                            shuffle=True)

    corruption = 'fog'

    GeneralizationSet = mnist.Dataset(corruption, 'test')[repair_num:].reshape(784)
    DrawdownSet = mnist.Dataset('identity', 'test').reshape(784)
    RepairSet = mnist.Dataset(corruption, 'test')[:repair_num].reshape(784)

    d0 = DrawdownSet.accuracy(target_model)
    g0 = GeneralizationSet.accuracy(target_model)
    r0 = RepairSet.accuracy(target_model)
    print(f"repair set acc: {r0:.2%}")

    images, labels = RepairSet.load(repair_num)
    images = images.to(dtype=dtype, device=device)
    labels = labels.to(device=device).flatten()

    buggy_inputs = images
    right_label = labels

    acc = test_acc(test_dataloader, target_model)
    print(f'Test Acc after before: {acc*100}%')
    buggy_inputs, right_label = buggy_inputs[:repair_num], right_label[:repair_num]

    print(buggy_inputs.shape)

    savename = f'PNNed_model_{netname}_{repair_num}p.pt'
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

    print("params: ", sum([p.numel() for p in repaired_model.parameters()]))

    # otherwise runs out of memory
    device = st.device(args.device)
    dtype = st.float64
    print(device, dtype)
    N = repaired_model.to(device,dtype)

    start = timer()
    d1 = DrawdownSet.accuracy(repaired_model, dtype=dtype)
    time_d = timer() - start
    print(f"drawdown      : {d0:.2%} -> {d1:.2%} ({d0 - d1:.2%}) (D Time: {time_d} s).")

    start = timer()
    g1 = GeneralizationSet.accuracy(repaired_model, dtype=dtype)
    time_g1 = timer() - start
    print(f"generalization: {g0:.2%} -> {g1:.2%} ({g1 - g0:.2%}) (G Time: {time_g1} s).")

    start = timer()
    r1 = RepairSet.accuracy(repaired_model, dtype=dtype)
    time_r = timer() - start
    print(f"repair set acc: {r1:.2%} (R Time: {time_r} s).")


if __name__ == '__main__':
    for num in [args.num]:
        print('-'*50, ';repair num =', num, '-'*50)
        PNN_MNIST(num, 0.5, num_core=64, remove_redundant_constraint=False)
