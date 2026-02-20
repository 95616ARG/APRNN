from __future__ import annotations
import pathlib
import sytorch as torch
from sytorch import nn
import numpy as np
from typing import Literal, TypeVar, Callable, Tuple, overload
import copy
from tqdm.auto import tqdm

import sytorch
from experiments.base import get_workspace_root

class _IdentityIndices:
    def __getitem__(self, idx):
        if isinstance(idx, int):
            return idx
        elif isinstance(idx, slice):
            return tuple(range(
                idx.start or 0,
                idx.stop or 0,
                idx.step or 1,
            ))

T = TypeVar('T', bound='Dataset')
class Dataset(torch.utils.data.Dataset):
    def __init__(self: T, corruption: str, split: Literal['test', 'train'], root = (get_workspace_root() / 'data' / 'mnist_c').as_posix()):
        if not pathlib.Path(root).exists():
            raise RuntimeError(
                f"\n\nMNIST-C dataset ({root}) doesn't exist.\nPlease run `make datasets-mnist`.\n\n"
            )
        self.corruption = corruption
        self.split = split
        self.images = torch.from_numpy(
            np.load(f"{root}/{corruption}/{split}_images.npy")\
                .reshape((-1, 28 * 28)) / 255.)

        self.labels = torch.from_numpy(
            np.load(f"{root}/{corruption}/{split}_labels.npy"))\
                .int()[:,None]

        self.shape = (1,28,28)

        self.device = None
        self.dtype = None
        # self.indices = _IdentityIndices()
        self.indices = np.arange(len(self.labels)).astype(int)

    def shuffle(self, seed):
        obj = self.copy()
        if seed == -1:
            # print(f"seed is {seed}, not shuffling.")
            return obj
        rng = np.random.default_rng(seed)
        rng.shuffle(obj.indices)
        return obj

    def to(self, *args, **kwargs):
        for arg in args:
            if isinstance(arg, torch.device):
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

    def sample_incorrect_label(self, network, label, num=1):
        device, dtype = network.device, network.dtype
        remaining = num
        out = []
        out_labels = []
        for images, labels in self.filter_label(label).dataloader(batch_size=100):
            images = images.to(device,dtype)
            mask = network(images).argmax(-1).reshape(-1).cpu() != labels.reshape(-1)
            if mask.any():
                out.append(images[mask])
                out_labels.append(labels[mask])
                remaining -= mask.sum()
            if remaining <= 0:
                break
        out = torch.cat(out, 0)[:num]
        assert len(out) == num
        out_labels = torch.cat(out_labels, 0)[:num]
        return out, out_labels

    def sample_correct_label(self, network, label, num=1):
        device, dtype = network.device, network.dtype
        remaining = num
        out = []
        out_labels = []
        for images, labels in self.filter_label(label).dataloader(batch_size=100):
            images = images.to(device,dtype)
            mask = network(images).argmax(-1).reshape(-1).cpu() == labels.reshape(-1)
            if mask.any():
                out.append(images[mask])
                out_labels.append(labels[mask])
                remaining -= mask.sum()
            if remaining <= 0:
                break
        out = torch.cat(out, 0)[:num]
        assert len(out) == num
        out_labels = torch.cat(out_labels, 0)[:num]
        return out, out_labels

    def sample_correct_labels(self, network, labels, num=1, keepdim=True):
        out = []
        out_labels = []
        for label in labels:
            label_out, label_labels = self.sample_correct_label(network, label, num=num)
            out.append(label_out)
            out_labels.append(label_labels)
        if keepdim:
            return torch.stack(out, 0), torch.stack(out_labels, 0)
        else:
            return torch.cat(out, 0), torch.cat(out_labels, 0)

    def sample_incorrect_labels(self, network, labels, num=1, keepdim=True):
        out = []
        out_labels = []
        for label in labels:
            label_out, label_labels = self.sample_incorrect_label(network, label, num=num)
            out.append(label_out)
            out_labels.append(label_labels)
        if keepdim:
            return torch.stack(out, 0), torch.stack(out_labels, 0)
        else:
            return torch.cat(out, 0), torch.cat(out_labels, 0)

    # def sample_all_labels(self, network, target_labels=tuple(range(10)), num=1):
    #     device, dtype = network.device, network.dtype
    #     out = []
    #     out_labels = []
    #     for label in target_labels:
    #         for images, labels in self.filter_label(label).dataloader(batch_size=100):
    #             assert (labels == label).all()
    #             mask = network(images.to(device, dtype)).argmax(-1).reshape(-1).cpu() == labels.reshape(-1)
    #             if mask.any():
    #                 out.append(images[mask])
    #                 out_labels.append(labels[mask])
    #                 break
    #     assert len(out) == 10
    #     return torch.stack(out), torch.tensor(out_labels)[...,None]

    def filter_label(self, label):
        if label is None:
            return self
        mask = (self.labels[self.indices].squeeze(-1) == label)
        obj = copy.copy(self)
        # obj.images = self.images[mask]
        # obj.labels = self.labels[mask]
        obj.indices = self.indices[mask]
        return obj

    # def filter(self,
    #     filter_fn: Literal['misclassified'] | Callable[[Tuple[torch.Tensor, torch.Tensor]], bool],
    #     *args, **kwargs
    # ) -> T:

    #     if filter_fn == 'misclassified':
    #         return self.filter_misclassified(*args, **kwargs)

    #     obj = copy.copy(self)
    #     obj.images, obj.labels = tuple(map(torch.stack, zip(*filter(filter_fn, (self[i] for i in range(len(self)))))))

    #     return obj

    def reshape(self, *shape):
        self.shape = shape
        return self

    def filter_misclassified(self, network) -> T:
        out = network(self.images[self.indices].reshape(-1, *self.shape).clone().to(network.device, network.dtype)).argmax(dim=-1).cpu()
        correct, incorrect = copy.copy(self), copy.copy(self)
        correct.indices = self.indices[out == self.labels[self.indices][:,0]]
        incorrect.indices = self.indices[out != self.labels[self.indices][:,0]]
        return correct, incorrect

    misclassified = filter_misclassified

    def __len__(self) -> int:
        if isinstance(self.indices, _IdentityIndices):
            return self.images.shape[0]
        else:
            return len(self.indices)

    def __getitem__(self, idx) -> torch.Tensor:
        if isinstance(idx, int):
            idx = self.indices[idx]
            return self.images[idx].reshape(self.shape).clone(), self.labels[idx].clone()

        elif isinstance(idx, slice):
            # assert isinstance(self.indices, _IdentityIndices)
            obj = copy.copy(self)
            # obj.images = self.images[idx]
            # obj.labels = self.labels[idx]
            obj.indices = self.indices[idx]
            # obj.shape = self.shape
            return obj

    def copy(self):
        return copy.copy(self)

    @overload
    def load(self, size=1, shuffle=False, sampler=None,
           batch_sampler=None, num_workers=0, collate_fn=None,
           pin_memory=False, drop_last=False, timeout=0,
           worker_init_fn=None, *, prefetch_factor=2,
           persistent_workers=False): ...

    def load(self, size=1, **kwargs):
        if size == 'all':
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
        if kwargs.get('kwargs', 0) == 0:
            kwargs['prefetch_factor'] = None
        return torch.utils.data.DataLoader(self, **kwargs)

    @overload
    def topk(self, network: sytorch.nn.Module, k=1, largest=True,
        device = None, dtype  = None, batch_size=100, num_workers=8,
        prefetch_factor=2, pin_memory=True, **kwargs): ...
    def topk(self, network: sytorch.nn.Module, k=1, largest=True, **kwargs):
        return self.accuracy(network=network, topk=k, largest=largest, **kwargs)

    def accuracy(self, network: sytorch.nn.Module, topk=1, largest=True,
        device = None, dtype  = None, batch_size=100, num_workers=0,
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
            network.eval()
            ncorrects = [0] * len(topks)
            from tqdm.auto import tqdm
            for images, labels in tqdm(
                self.dataloader(
                    batch_size      = batch_size,
                    num_workers     = num_workers,
                    prefetch_factor = prefetch_factor,
                    pin_memory      = pin_memory,
                    **kwargs
                ),
                desc  = "Evaluating ...",
                leave = False,
            ):
            # for images, labels in self.dataloader(
            #     batch_size      = batch_size,
            #     num_workers     = num_workers,
            #     prefetch_factor = prefetch_factor,
            #     pin_memory      = pin_memory,
            #     **kwargs
            # ):
                output = network(images.to(device,dtype))
                labels = labels.cpu()
                for i in range(len(topks)):
                    topk_labels = output.topk(k=topks[i], dim=-1, largest=largest).indices.cpu()
                    assert topk_labels.ndim == labels.ndim
                    ncorrects[i] += int((topk_labels == labels).any(dim=-1).sum())

            accs = tuple(ncorrect / len(self) for ncorrect in ncorrects)
            if isinstance(topk, int):
                accs = accs[0]

            return accs

        finally:
            self.to(device=_device)

    def rotate(self, degs, scale=1., **kwargs):
        obj = self.copy()
        obj.scale= scale
        obj.degs = degs
        obj.__class__ = _Rot
        return obj

MNIST_C = Dataset

num_repair_set = 1000

def RepairSet(corruption):
    return Dataset(corruption=corruption, split='test')[:num_repair_set]

def GeneralizationSet(corruption):
    return Dataset(corruption=corruption, split='test')[num_repair_set:]

def DrawdownSet():
    return Dataset(corruption='identity', split='test')

import torchvision
from PIL import Image as PILImage
import functools

class _Rot(Dataset):
    # def __init__(self, degs, scale=1., root='data/ILSVRC2012', split='val', **kwargs):
    #     super().__init__(root=root, split=split, **kwargs)
    #     self.scale = scale
    #     self.degs = degs

    # @functools.lru_cache(maxsize=10000)
    def __getitem__(self, idx):
        if isinstance(idx, int):
            # idx = self.indices[idx]
            image, label = super().__getitem__(idx)
            _shape = image.shape
            polytope = torch.stack(tuple(
                torchvision.transforms.functional.affine(
                    image.reshape(-1, 1, 28, 28),
                    angle=deg,
                    translate=(0, 0),
                    scale=self.scale,
                    shear=0,
                    resample=PILImage.BILINEAR
                    # interpolation=torchvision.transforms.InterpolationMode.BILINEAR
                ).reshape(_shape) for deg in self.degs
            ),0)
            return polytope, torch.broadcast_to(label, (len(self.degs),1))

        return super().__getitem__(idx)

    def accuracy(self, network, topk=1, largest=True,
        device = None, dtype  = None, batch_size=100, num_workers=0,
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
            from tqdm.auto import tqdm
            network.eval()
            ncorrects = [0] * len(topks)
            total = 0
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
                images = images.flatten(0,1)
                labels = labels.flatten(0,1)
                total += labels.numel()
                output = network(images.to(device,dtype))
                labels = labels.cpu()
                for i in range(len(topks)):
                    topk_labels = output.topk(k=topks[i], dim=-1, largest=largest).indices.cpu()
                    assert topk_labels.ndim == labels.ndim
                    ncorrects[i] += int((topk_labels == labels).any(dim=-1).sum())

            accs = tuple(ncorrect / total for ncorrect in ncorrects)
            if isinstance(topk, int):
                accs = accs[0]

            return accs

        finally:
            self.to(device=_device)


# class RotPolys(MNIST_C):
#     def __init__(self, corruption, split, degs, npoints):
#         super().__init__(corruption=corruption, split=split)
#         self.degs = np.array(degs)
#         self.images = torch.from_numpy(np.stack([
#             rotate(
#                 self.images[:npoints].numpy()\
#                     .reshape(-1, 1, 28, 28)\
#                     .transpose(0, 2, 3, 1),
#                 deg).transpose(0, 3, 1, 2)
#             for deg in degs
#         ], 1))
#         self.labels = sytorch.broadcast_at(self.labels[:npoints].numpy(), 1, (len(degs),))

#     def filter_label(self, label):
#         if label is None:
#             return self
#         mask = (self.labels.squeeze(-1)[:,0] == label)
#         obj = copy.copy(self)
#         obj.images = self.images[mask]
#         obj.labels = self.labels[mask]
#         return obj

#     def __getitem__(self, idx):
#         if isinstance(idx, int):
#             idx = self.indices[idx]
#             return self.images[idx].reshape(-1, *self.shape), self.labels[idx]

#         return super().__getitem__(idx)

#     def accuracy(self, network, topk=1, largest=True,
#         device = None, dtype  = None, batch_size=100, num_workers=0,
#         prefetch_factor=2, pin_memory=True, **kwargs):
#         device = device or self.device or next(iter(network.parameters())).device
#         dtype  = dtype  or self.dtype  or next(iter(network.parameters())).dtype
#         if device != next(iter(network.parameters())).device \
#         or dtype  != next(iter(network.parameters())).dtype:
#             network = network.deepcopy().to(device=device).to(dtype=dtype)

#         _device = self.device
#         self.to(torch.device('cpu'))

#         topks = (topk,) if isinstance(topk, int) else topk

#         try:
#             from tqdm.auto import tqdm
#             network.eval()
#             ncorrects = [0] * len(topks)
#             total = 0
#             for images, labels in tqdm(
#                 self.dataloader(
#                     batch_size      = batch_size,
#                     num_workers     = num_workers,
#                     prefetch_factor = prefetch_factor,
#                     pin_memory      = pin_memory,
#                     **kwargs
#                 ),
#                 desc  = "Evaluating accuracy...",
#                 leave = False,
#             ):
#                 images = images.flatten(0,1)
#                 labels = labels.flatten(0,1)
#                 total += labels.numel()
#                 output = network(images.to(device,dtype))
#                 labels = labels.cpu()
#                 for i in range(len(topks)):
#                     topk_labels = output.topk(k=topks[i], dim=-1, largest=largest).indices.cpu()
#                     assert topk_labels.ndim == labels.ndim
#                     ncorrects[i] += int((topk_labels == labels).any(dim=-1).sum())

#             accs = tuple(ncorrect / total for ncorrect in ncorrects)
#             if isinstance(topk, int):
#                 accs = accs[0]

#             return accs

#         finally:
#             self.to(device=_device)

def rotate(images, deg):
    import skimage as sk
    from skimage.filters import gaussian
    from skimage import transform, feature
    rad = deg * np.pi / 180.

    aff = transform.AffineTransform(rotation=rad)

    a1, a2 = aff.params[0,:2]
    b1, b2 = aff.params[1,:2]
    a3 = 13.5 * (1 - a1 - a2)
    b3 = 13.5 * (1 - b1 - b2)
    aff = transform.AffineTransform(rotation=rad, translation=[a3, b3])

    rotated = []
    for img in images:
        img = img #/ 255.
        img = transform.warp(img, inverse_map=aff)
        img = np.clip(img, 0, 1) #* 255
        rotated.append(img)

    return np.stack(rotated, 0)
