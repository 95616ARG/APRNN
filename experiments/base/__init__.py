from typing import overload
import copy
import numpy as np
import torch
import sytorch as st
import argparse

def print_msg_box(msg, indent=1, width=None, title=None):
    """Print message-box with optional title."""
    lines = msg.split('\n')
    space = " " * indent
    if not width:
        width = max(map(len, lines))
    box = f'╔{"═" * (width + indent * 2)}╗\n'  # upper_border
    if title:
        box += f'║{space}{title:<{width}}{space}║\n'  # title
        box += f'║{space}{"-" * len(title):<{width}}{space}║\n'  # underscore
    box += ''.join([f'║{space}{line:<{width}}{space}║\n' for line in lines])
    box += f'╚{"═" * (width + indent * 2)}╝'  # lower_border
    print(box)

def get_device(device='cuda'):

    if device == 'cpu':
        return st.device('cpu')

    if not st.cuda.is_available():
        print_msg_box(
            "Warning:\n"
            "\n"
            f"GPU/CUDA ('{device}') is not available, therefore this experiment will \n"
            "take more time to finish. If you want to use GPU, please make sure \n"
            "compatible NVIDIA driver and CUDA toolkit are installed. \n"
            "You could check installed NVIDIA driver and CUDA toolkit via `nvidia-smi`.\n"
            "You could also specify a torch (and torchvision) version that is \n"
            "compatible with your setup `requirements.txt` and rerun `make venv`. \n"
            "\n"
            "If you are running inside a Docker container, please append `--gpus all`\n"
            "to `docker run`. You could also specify a compatible torch version in \n"
            "Dockerfile accordingly and rebuild the image. If that does not work, \n"
            "please check the official NVIDIA Container Toolkit to start container \n"
            "with GPU support; see \n"
            "https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html\n"
            "\n"
            "If you are running inside a virtual machine, please check official NVIDIA \n"
            "Virtual GPU Software User Guide to enable GPU pass-through (if applicable);\n"
            "see: https://docs.nvidia.com/grid/5.0/grid-vgpu-user-guide/index.html"
        )

    try:
        if ':' in device:
            cuda_id = int(device.split(":")[-1])
        else:
            cuda_id = None
        avail, total = torch.cuda.mem_get_info(cuda_id)
        avail = avail / 1024 / 1024 / 1024
        total = total / 1024 / 1024 / 1024
        print(f"{avail:.1f}G / {total:.1f}G memory available on torch.device('{device}').")

    except:
        pass

    return st.device(device)

def get_workspace_root():
    import os, sys, pathlib
    return pathlib.Path(
        os.environ.get(
            'BUILD_WORKSPACE_DIRECTORY',
            default = sys.path[0]
        )
    )

def get_models_root():
    path = get_workspace_root() / "models"
    path.mkdir(parents=True, exist_ok=True)
    return path

def get_datasets_root():
    path = get_workspace_root() / "data"
    path.mkdir(parents=True, exist_ok=True)
    return path

def get_results_root():
    path = get_workspace_root() / "results"
    path.mkdir(parents=True, exist_ok=True)
    for i in (1,2,3,4,5):
        (path / f'eval_{i}').mkdir(parents=True, exist_ok=True)
    return path

def get_artifact_root():
    path = get_workspace_root() / "artifact"
    path.mkdir(parents=True, exist_ok=True)
    return path

class Dataset(torch.utils.data.Dataset):

    def copy(self):
        return copy.copy(self)

    def to(self, *args, **kwargs):
        for arg in args:
            if arg == None:
                pass

            elif isinstance(arg, torch.device):
                assert 'device' not in kwargs
                kwargs['device'] = arg

            elif isinstance(arg, torch.dtype):
                assert 'dtype' not in kwargs
                kwargs['dtype'] = arg

            else:
                raise RuntimeError(f"unsupported {arg}.")

        self.device = kwargs.get('device', self.device)
        self.dtype = kwargs.get('dtype', self.dtype)

        return self

    def misclassified(self, network, num, seed, device=None, dtype=None):
        device = device or self.device or next(iter(network.parameters())).device
        dtype  = dtype  or self.dtype  or next(iter(network.parameters())).dtype
        network = network.to(device=device).to(dtype=dtype)
        network.eval()
        indices = []
        shuffled = np.arange(len(self)).astype(int)
        if seed is not None:
            np.random.default_rng(seed).shuffle(shuffled)

        from tqdm.auto import tqdm
        progress = tqdm(desc='misclassified', total=num)
        with st.no_grad(), st.no_symbolic():
            batch_size = 100
            for n_batch, (images, labels) in enumerate(self.dataloader(batch_size=batch_size, shuffle=False)):
                results = network(images).argmax(-1).cpu() == labels.squeeze(1).cpu()
                for i in range(len(results)):
                    if not results[i]:
                        indices.append(n_batch * batch_size + i)
                        progress.update(1)
                    if len(indices) >= num: break
                if len(indices) >= num: break

        progress.close()
            # for i in tqdm(range(len(self)), total=num, desc='misclassified'):
            #     i = int(shuffled[i])
            #     images, labels = self[i]
            #     images = images[None, ...].to(device, dtype)
            #     if network(images).argmax(-1)[0].cpu() != labels:
            #         indices.append(i)
            #         if len(indices) >= num:
            #             break

        # print(len(indices), indices)
        subset = self.subset(indices)
        # print("double checking repair set accuracy ... ", end='')
        # acc = subset.accuracy(network)
        # print(acc)
        # assert acc == 0.
        return subset

    def subset(self, indices):
        obj = copy.copy(self)
        obj.indices = indices
        return obj

    @overload
    def load(self, size=None, shuffle=False, sampler=None,
           batch_sampler=None, num_workers=0, collate_fn=None,
           pin_memory=False, drop_last=False, timeout=0,
           worker_init_fn=None, *, prefetch_factor=2,
           persistent_workers=False): ...

    def load(self, size=None, **kwargs):
        if size == None:
            size = len(self)
        kwargs['batch_size'] = size
        return next(iter(self.dataloader(**kwargs)))

    @overload
    def dataloader(self, batch_size=1, shuffle=False, sampler=None,
           batch_sampler=None, num_workers=0, collate_fn=None,
           pin_memory=False, drop_last=False, timeout=0,
           worker_init_fn=None, *, prefetch_factor=2,
           persistent_workers=False): ...

    def dataloader(self, **kwargs) -> torch.utils.data.DataLoader:
        return torch.utils.data.DataLoader(self, **kwargs)

    @overload
    def topk(self, network, k=1, largest=True,
        device = None, dtype  = None, batch_size=100, num_workers=8,
        prefetch_factor=2, pin_memory=True, **kwargs): ...
    def topk(self, network, k=1, largest=True, **kwargs):
        return self.accuracy(network=network, topk=k, largest=largest, **kwargs)

    def accuracy(self, network, topk=1, largest=True,
        device = None, dtype  = None, batch_size=100, num_workers=8,
        prefetch_factor=2, pin_memory=True, **kwargs):
        device = device or self.device or next(iter(network.parameters())).device
        dtype  = dtype  or self.dtype  or next(iter(network.parameters())).dtype
        if device != next(iter(network.parameters())).device \
        or dtype  != next(iter(network.parameters())).dtype:
            network = network.deepcopy().to(device=device).to(dtype=dtype)

        _device = self.device
        self.to(torch.device('cpu'))

        topks = (topk,) if isinstance(topk, int) else topk

        try:
            from tqdm import tqdm
            network.eval()
            ncorrects = [0] * len(topks)
            for images, labels in tqdm(
                self.dataloader(
                    batch_size      = batch_size,
                    num_workers     = num_workers,
                    prefetch_factor = prefetch_factor,
                    pin_memory      = pin_memory,
                    **kwargs
                ),
                desc  = "Evaluating accuracy...",
                leave = False,
            ):
                output = network(images.to(device,dtype))
                labels = labels.cpu()
                for i in range(len(topks)):
                    topk_labels = output.topk(k=topks[i], dim=-1, largest=largest).indices.cpu()
                    ncorrects[i] += int((topk_labels == labels).any(dim=-1).sum())

            accs = tuple(ncorrect / len(self) for ncorrect in ncorrects)
            if isinstance(topk, int):
                accs = accs[0]

            return accs

        finally:
            self.to(_device)

class ParseMaskAction(argparse.Action):
    def __init__(self, option_strings, dest, nargs=None, **kwargs):
        if nargs is not None:
            raise ValueError("nargs not allowed")
        super().__init__(option_strings, dest, **kwargs)

    def __call__(self, parser, namespace, values, option_string=None):
        setattr(namespace, self.dest, self.parse(values))

    @staticmethod
    def parse(s):
        if s == '-1' or s == None:
            return None
        elif ':' in s:
            start, end = tuple(map(int, s.split(':')))
            return as_slice[start:end]
        elif '.' in s:
            return float(s)
        elif '+' in s:
            rows, step = tuple(map(int, s.split('+')))
            rows = rows
            return as_slice[rows:rows+step]
        else:
            return int(s)

class ParseIndex(argparse.Action):
    def __init__(self, option_strings, dest, nargs=None, **kwargs):
        if nargs is not None:
            raise ValueError("nargs not allowed")
        super().__init__(option_strings, dest, **kwargs)

    def __call__(self, parser, namespace, values, option_string=None):
        setattr(namespace, self.dest, self.parse(values))

    @staticmethod
    def parse(s):
        return as_slice[tuple(map(int, s.split(",")))]
