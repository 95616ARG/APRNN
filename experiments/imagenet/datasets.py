from __future__ import annotations
import sytorch as st
import sytorch as torch
import torchvision
from torchvision import transforms as trn
import numpy as np
from typing import Literal, TypeVar, Callable, Tuple, Iterable, overload
import copy
import pathlib
from tqdm.auto import tqdm

import experiments.base
from experiments.base import get_workspace_root

imagenet_transform = trn.Compose([
    trn.Resize(256, interpolation=trn.InterpolationMode.BILINEAR),
    trn.CenterCrop(224),
    trn.ToTensor(),
    trn.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

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

class Dataset(experiments.base.Dataset):

    def __init__(self, device=None, dtype=None):
        self.indices = _IdentityIndices()
        self.dtype = dtype
        self.device = device

    def __len__(self):
        if isinstance(self.indices, _IdentityIndices):
            return len(self.dataset)
        else:
            return len(self.indices)

    def __getitem__(self, idx) -> torch.Tensor:

        if isinstance(idx, int):
            idx = self.indices[idx]
            image, label = self.dataset[idx]
            label = torch.IntTensor([label])

            if self.dtype is not None:
                image = image.to(dtype=self.dtype)

            if self.device is not None:
                image = image.to(device=self.device)
                label = label.to(device=self.device)

            return image, label

        elif isinstance(idx, slice):
            obj = self.copy()
            obj.indices = self.indices[idx]
            return obj

        else:
            raise NotImplementedError(
                f"unsupproted idx {idx} {type(idx)}"
            )

class ImageNet(Dataset):
    def __init__(self, root=(get_workspace_root() / 'data' / 'ILSVRC2012').as_posix(), split='val', **kwargs):
        super().__init__(**kwargs)
        self.dataset = torchvision.datasets.ImageNet(root, split=split, transform=imagenet_transform)

class ImageNet_C(Dataset):
    def __init__(self, root=(get_workspace_root() / 'data' / 'imagenet-c').as_posix(), corruption='fog', severity=3, **kwargs):
        super().__init__(**kwargs)
        self.dataset = torchvision.datasets.ImageFolder(
            (pathlib.Path(root) / corruption / str(severity)).as_posix()
        , transform=imagenet_transform)

class ImageNet_A(Dataset):
    def __init__(self, root=(get_workspace_root() / 'data' / 'imagenet-a').as_posix(), **kwargs):
        super().__init__(**kwargs)
        # https://github.com/hendrycks/natural-adv-examples/blob/master/eval.py
        self.dataset = torchvision.datasets.ImageFolder(root, transform=imagenet_transform)

    def __getitem__(self, idx) -> torch.Tensor:
        img, lbl = super().__getitem__(idx)
        return img, lbl