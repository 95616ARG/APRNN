import warnings; warnings.filterwarnings("ignore")
from experiments import imagenet
from experiments.base import *
import sytorch as st
import argparse
from timeit import default_timer as timer

""" Extend beyond this experiment.
    ==============================

    1. Different buggy network.
       - Change `net` to load a different buggy network.
         Specifically, the following command loads a torch DNN `torch_dnn_object`:

         ```
         network = st.nn.from_torch(torch_dnn_object)
         ```

         And the following command loads an ONNX DNN `onnx_dnn_path` from file:

         ```
         network = st.nn.from_file(onnx_dnn_path)
         ```

    2. Different dataset.
       - Change the `images` and `labels` tensors to load expected buggy inputs
         and the correct labels, in the same way as training a PyTorch DNN.

    3. Different repair parameters:
       - Change the `k` and `rows` parameters.
       - Change `lb` and `ub`.
"""

parser = argparse.ArgumentParser(description='')
parser.add_argument('--net', type=str, dest='net', action='store', required=True,
                    help='resnet152, vgg19')
parser.add_argument('--device', type=str, dest='device', action='store', default='cpu',
                    help='device to use, e.g., cuda, cuda:0, cpu. (default=cpu).')
parser.add_argument('--use_artifact', dest='use_artifact', action='store_true',
                    help='use authors\' repaired DNN.')
args = parser.parse_args()

device = get_device(args.device)
dtype = st.float64

st.set_all_seed(0)

if args.net == 'resnet152':
    npoints = 50
    k = st.as_slice[-4,-1,0,0,-2]
    rows = st.as_slice[0:200]
    net = imagenet.models.resnet152(pretrained=True)\
        .to(dtype=dtype, device=device)\
        .eval()

elif args.net == 'vgg19':
    npoints = 50
    k = st.as_slice[-1, 3]
    rows = st.as_slice[400:800]
    net = imagenet.models.vgg19(pretrained=True)\
        .to(dtype=dtype, device=device)\
        .eval()

if not args.use_artifact:

    repair_dataset = imagenet.datasets.ImageNet_A()\
            .to(dtype=dtype, device=device)\
            .misclassified(net, num=npoints, seed=None)

    images, labels = repair_dataset.load(npoints)
    with st.no_grad():
        reference_output = net(images).cpu()

    """ Repair Phase. """

    start_time = timer()
    """ Create a new solver. """
    solver = st.LightningSolver()\
            .verbose_()

    """ Attach `net` to the solver and turn the repair (symbolic) mode on. """
    net = net.to(solver)\
            .repair()\
            .requires_symbolic_(False)

    net[k].requires_symbolic_(
        lb  = -10.,
        ub  =  10.,
        rows= rows,
        cols= None,
        bias= False,
        seed= 0
    )

    """ Compute the symbolic outputs of shape `(npoints, 10)`. Note that activation
        constraints are added implicitly to the attached solver.
    """
    symbolic_outputs = net(images)

    """ Add the classification constraints. """
    solver.add_constraints(symbolic_outputs.argmax(axis=-1) == labels)

    """ Collect the (symbolic) deltas of all symbolic parameters, concatenated as an
        1d-array.
    """
    param_deltas = net.parameter_deltas(concat=True)

    solver2 = solver.gurobi()
    solver2.solver.Params.Method = 2
    solver.solver = solver2

    param_deltas = param_deltas.to(solver2)
    print(param_deltas.shape)
    output_deltas = (symbolic_outputs.to(solver2) - reference_output).alias()
    print(output_deltas.shape)

    assert solver2.verbose_().solve(
        minimize = (
            param_deltas.norm_ub(order="linf") +
            param_deltas.norm_ub(order="l1_normalized") +
            output_deltas.reshape(-1).norm_ub(order='linf') +
            output_deltas.reshape(-1).norm_ub(order='l1_normalized')
        ),
    )
    time = timer() - start_time

    net = net.update_().repair(False)

    result_path = (get_results_root() / 'eval_2' / f'aprnn_{args.net}').as_posix()

else:

    net[k].load(get_artifact_root() / 'eval_2' / f'aprnn_{args.net}_diff.pth')
    time = None
    result_path = (get_results_root() / 'eval_2' / f'artifact_aprnn_{args.net}').as_posix()

# See: https://pytorch.org/vision/stable/models.html#table-of-all-available-classification-weights
og_acc1, og_acc5 = {
    'resnet152': (0.78312, 0.94046),
    'vgg19'    : (0.72376, 0.90876)
}[args.net]

with st.no_symbolic():
    net = net.to(dtype=st.float32)
    testset = imagenet.datasets.ImageNet(split='val').to(dtype=st.float32, device=device)
    acc1, acc5 = testset.accuracy(net, topk=(1, 5))

result = {
    'APRNN': {
        (args.net, 'D@top-1'): float(og_acc1 - acc1),
        (args.net, 'D@top-5'): float(og_acc5 - acc5),
        (args.net, 'T'): "N/A" if time is None else f'{int(time)}s',
    }
}

np.save(result_path+".npy", result, allow_pickle=True)

print_msg_box(
    f"Experiment 2 using APRNN for {args.net} SUCCEED.\n"
    f"Saved result to {result_path}.npy"
)
