from __future__ import annotations
from typing import Iterable, Literal, TypeVar, Callable, Tuple, overload, Final
import warnings
import torch
import sytorch
import sytorch as st
import numpy as np
import os, copy
from tqdm.auto import tqdm
from . import properties
from . import models

__all__ = [
    "Dataset",
]

def read_inputs(filename):
    scriptpath = os.path.dirname(__file__)
    inputs = np.load(f'{scriptpath}/datasets/{filename}')
    return inputs

def valid_properties(aprev, tau):
    """Returns a list of ints representing the properties applicable to the network N{aprev}{tau}."""

    assert aprev == 2 and tau == 9

    path = os.path.dirname(__file__)
    path += f'/datasets/N{aprev}{tau}_property'
    counterexamples = []
    for i in range(1, 11):
        if os.path.exists(f'{path}{i}_false.npy'):
            counterexamples.append(i)

    properties = [1]
    if aprev >= 2:
        properties.append(2)
    if aprev != 1 or tau not in [7, 8, 9]:
        properties.extend([3, 4])
    if aprev == 1 and tau == 1:
        properties.extend([5, 6])
    if aprev == 1 and tau == 9:
        properties.append(7)
    if aprev == 2 and tau == 9:
        properties.append(8)
    if aprev == 3 and tau == 3:
        properties.append(9)
    if aprev == 4 and tau == 5:
        properties.append(10)

    if (aprev, tau) == (2, 9):
        assert properties == [1, 2, 3, 4, 8]

    return properties, counterexamples

COC, WL, WR, SL, SR = list(range(5))

""" Pointwise dataset. """
T = TypeVar('T', bound='Dataset')

# def get_datasets(
#     aprev, tau,
#     repair_properties: Iterable[int]=None,
#     valid_properties: Iterable[int]=None,
#     h = 0.05,
#     gap = 0.0,
#     h_sample = 0.005,
#     num_repair = 100,
#     seed = 0,
#     device = torch.device('cpu'),
#     dtype = torch.float64
# ):
#     network, _norm, _denorm = models.acas(
#         aprev, tau, exclude_last_relu=True)
#     network = network.to(device, dtype)

#     """
#     repair, drawdown and generalization are defined on the repair property?
#     - repair set: points (boxes) that violates the repair property
#     - generalization set: points (boxes) that violates the repair property
#     - drawdown set: points (boxes) was satisfying both repair and valid
#         properties? Within the valid input region of repair property or not?
#     """

#     property_data = {
#         prop: property(prop).partition_and_classify(
#             _norm, network, h=h, gap=gap, h_sample=h_sample
#         ) for prop in valid_properties
#     }

#     """ repair and generalization set """
#     for prop in repair_properties:
#         hboxes, hboxes_points, indices_satisfy, indices_violates = property_data[prop]
#         repair_indices = indices_violates[:num_repair]
#         gen_indices = indices_violates[num_repair:]

#         repair_hboxes = hboxes[repair_indices]
#         repair_hboxes_points = hboxes_points[repair_indices]

#         gen_hboxes = hboxes[gen_indices]
#         gen_hboxes_points = hboxes_points[gen_indices]

#     """ drawdown set """
#     for prop in valid_properties:
#         hboxes, hboxes_points, indices_satisfy, indices_violates = property_data[prop]
#         drawdown_hboxes = hboxes[indices_satisfy]
#         drawdown_hboxes_points = hboxes_points[indices_satisfy]

# # class _Dataset(torch.utils.data.Dataset):
# #     def __init__(self, hboxes, hboxes_points):
# #         self.hboxes = hboxes
# #         self.hboxes_points = hboxes_points

# #     def __len__(self):
# #         return len(self.hboxes)




# class Dataset(torch.utils.data.Dataset):
#     def __init__(self: T,
#         aprev, tau,
#         repair_properties: Iterable[int]=None,
#         valid_properties: Iterable[int]=None,
#         h = 0.05,
#         gap = 0.0,
#         h_sample = 0.005,
#         num_repair = 100,
#         seed = 0,
#         device = torch.device('cpu'),
#         dtype = torch.float64,
#     ):
#         self.network, self._norm, self._denorm = models.acas(
#                 aprev, tau, exclude_last_relu=True)
#         self.network = self.network.to(device, dtype)
#         self.h, self.gap, self.h_sample = h, gap, h_sample
#         self.seed = seed
#         self.repair_properties = repair_properties
#         self.valid_properties = valid_properties

#         """
#         repair, drawdown and generalization are defined on the repair property?
#         - repair set: points (boxes) that violates the repair property
#         - generalization set: points (boxes) that violates the repair property
#         - drawdown set: points (boxes) was satisfying both repair and valid
#           properties? Within the valid input region of repair property or not?
#         """

#         self.property_data = {
#             prop: property(prop).partition_and_classify(
#                 self._norm, self.network,
#                 h=self.h, gap=self.gap, h_sample=self.h_sample
#             ) for prop in self.valid_properties
#         }

#         """ repair set """
#         for prop in self.repair_properties:
#             hboxes, hboxes_points, indices_satisfy, indices_violates = self.property_data[prop]
#             hboxes[indices_violates][:num_repair]
#             hboxes_points[indices_violates][:num_repair]

#         """ generalization set """

#         """ drawdown set """






class Dataset(torch.utils.data.Dataset):
    def __init__(self: T,
        network,
        repair_properties,
        valid_properties,
        split: Literal['drawdown', 'repair', 'generalization'] = 'repair',
        repair_num=100,
        seed=42
    ):

        self.network = network
        self.repair_properties = [properties.property(p) for p in repair_properties]
        self.valid_properties = [properties.property(p) for p in valid_properties]
        self.split = split

        if split == 'repair' or split == 'generalization':

            self.points = np.unique(np.concatenate([
                read_inputs(f'N{network[0]}{network[1]}_property{p}_false.npy')
                for p in repair_properties
            ]), axis=0)

            # print(f"Randomly splitting repair and gen set with seed {seed}.")
            shuffle_indices = np.random.default_rng(seed=seed).permutation(self.points.shape[0])
            self.points = self.points[shuffle_indices]

            if split == 'repair':
                # TODO: Guarantee equal representation of repair properties in the repair spec.
                self.points = self.points[:repair_num]
            elif split == 'generalization':
                self.points = self.points[repair_num:]
            else:
                raise NotImplementedError

        elif split == 'drawdown':

            warnings.warn("revise drawdown set")
            net, normalize, _ = models.acas(network[0], network[1])
            drawdown_set = []
            for p in self.valid_properties:
                for ranges in p.input_polytopes:
                    sample_points = np.empty([1000, 5])
                    for i in range(5):
                        sample_points[..., i] = np.random.uniform(ranges[i][0], ranges[i][1], size=1000)
                    sample_points = np.asarray([normalize(point) for point in sample_points])
                    results = p(net(torch.from_numpy(sample_points).float()))
                    sample_points = [sample_points[i] for i in range(len(sample_points)) if results[i]]
                    drawdown_set.extend(sample_points)

            self.points = []
            for point in np.unique(np.asarray(drawdown_set), axis=0):
                valid = True
                for p in self.valid_properties:
                    if p.applicable([point]) and not p(net(torch.from_numpy(point).float())):
                        valid = False
                if valid:
                    self.points.append(point)
            self.points = np.asarray(self.points)
            # print("Drawdown points: ", self.points.shape)

        else:
            raise NotImplementedError

    def __len__(self) -> int:
        return self.points.shape[0]

    def __getitem__(self, idx) -> sytorch.Tensor:
        if isinstance(idx, int):
            return sytorch.from_numpy(self.points[idx].copy()).float()
        elif isinstance(idx, slice):
            obj = copy.copy(self)
            obj.points = self.points[idx]
            return obj

    @overload
    def dataloader(batch_size=1, shuffle=False, sampler=None,
           batch_sampler=None, num_workers=0, collate_fn=None,
           pin_memory=False, drop_last=False, timeout=0,
           worker_init_fn=None, *, prefetch_factor=2,
           persistent_workers=False): ...

    def dataloader(self, **kwargs) -> sytorch.utils.data.DataLoader:
        return sytorch.utils.data.DataLoader(self, **kwargs)

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

    def accuracy(self,
        network: sytorch.nn.Module,
        device = None,
        dtype = None,
        batch_size=100,
        num_workers=1,
        **kwargs
    ):
        """Returns a list of accuracies for all properties in self.valid_properties."""
        if network == None:
            return np.nan

        device = device or next(iter(network.parameters())).device
        dtype  = dtype  or next(iter(network.parameters())).dtype
        network = network.to(device=device).to(dtype=dtype)

        with torch.no_grad(), sytorch.no_symbolic():
            network.eval()
            accuracies = []
            for p in self.valid_properties:
                correct = 0.
                applicable = 0.
                for points in tqdm(self.dataloader(batch_size=batch_size, num_workers=num_workers, **kwargs), desc="Evaluating accuracy...", leave=False):
                    applicable_batch = p.applicable(points)
                    applicable += np.count_nonzero(applicable_batch)
                    correct += np.count_nonzero(p(network(points[applicable_batch].to(device).to(dtype).cpu())))
                if applicable == 0:
                    accuracies.append(None)
                else:
                    accuracies.append(correct / applicable)

        return accuracies




