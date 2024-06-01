from logging import warning
import pathlib
from typing import overload
import warnings
from .models import *
import torch
import numpy as np
from tqdm.auto import tqdm
import sytorch as st
from sytorch import SymbolicArray
from sytorch.solver.symbolic_array import SymbolicGurobiConstrArray
from experiments.base import *

class Property:
    def __call__(self, *args, **kwargs):
        return self.output_constraints(*args, **kwargs)

    def save(self, path, **kwargs):
        return torch.save(
            {
                key: getattr(self, key)
                for key in (
                    'hboxes',
                    'hboxes_points',
                    'hboxes_satisfy_indices',
                    'hboxes_violate_indices',
                    'hboxes_violate_points_indices',
                )
            }, path, **kwargs
        )

    def load(self, path):
        for k, v in torch.load(path).items():
            setattr(self, k, v)
        return self

    def satisfy_accuracy(self, network, other_prop=None):
        network.eval()
        device, dtype = network.device, network.dtype

        total = 0
        satisfied = 0
        from tqdm.auto import tqdm
        for idx in tqdm(self.hboxes_satisfy_indices, desc='evaluating', leave=False):
            points = self.hboxes_points[idx]
            total += len(points)
            if other_prop is not None:
                out = network(points.to(device,dtype))
                satisfied += (self(out) * other_prop(out)).sum()
            else:
                satisfied += self(network(points.to(device,dtype))).sum()

        return satisfied / total, satisfied, total

    def efficacy_accuracy(self, network):
        network.eval()
        device, dtype = network.device, network.dtype

        total = 0
        satisfied = 0
        for points in self.hboxes_points:
            total += len(points)
            satisfied += self(network(points.to(device,dtype))).sum()

        return satisfied / total

    # @property
    # def satisfy_points(self):
    #     return torch.cat(tuple(
    #         self.hboxes_points[idx][points_idx]
    #         for idx, points_idx in zip(
    #             self.hboxes_violate_indices,
    #             self.hboxes_violate_points_indices
    #         )
    #     ), dim=0)

    @property
    def violate_hboxes(self):
        return self.hboxes[self.hboxes_violate_indices].clone()

    @property
    def violate_points(self):
        return torch.cat(tuple(
            self.hboxes_points[idx][points_idx]
            for idx, points_idx in zip(
                self.hboxes_violate_indices,
                self.hboxes_violate_points_indices
            )
        ), dim=0)

    @property
    def violate_vertices(self):
        return st.hboxes_to_vboxes(
            self.hboxes[self.hboxes_violate_indices],
            flatten=True
        )

    def violate_accuracy(self, network):
        network.eval()
        device, dtype = network.device, network.dtype

        total = 0
        satisfied = 0
        for idx, points_idx in zip(self.hboxes_violate_indices, self.hboxes_violate_points_indices):
            points = self.hboxes_points[idx][points_idx]
            this_satisfied = self(network(points.to(device,dtype))).sum()
            total += len(points)
            satisfied += this_satisfied
            # print(idx, this_satisfied / len(points), this_satisfied, len(points))

        return satisfied / total, satisfied, total

    def partition_and_classify_(self,
        normalize_input, network, *, h, gap, h_sample,
        label=None, update=False,
    ):
        """ 1. partition into hboxes
            2. classify into hboxes that violates or satisfies the property.
            3. points that violates and satisfy the property.

        Args:
            normalize_input (_type_): _description_
            network (_type_): _description_
            h (float, optional): _description_. Defaults to .1.
            gap (float, optional): _description_. Defaults to .0.

        Returns:
            _type_: _description_
        """
        device = network.device
        dtype = network.dtype

        if label is not None:
            dir = get_datasets_root() / 'acasxu'
            dir.mkdir(parents=True, exist_ok=True)
            filename = dir / f"acas_cache_{type(self).__name__}_{label}_h{h}_gap{gap}_hsample{h_sample}.pth"
            if not update and filename.exists():
                # print(f"Loading cache {filename}.")
                return self.load(filename.as_posix())

        # print(f"Updating cache {filename}.")
        hboxes = self.partition(normalize_input=normalize_input, h=h, gap=gap, device=device)
        hboxes_points = tuple(points.to(dtype) for points in st.sample_hbox(hboxes, h=h_sample))

        hboxes_satisfy_indices = []
        hboxes_violate_indices = []
        hboxes_violate_points_indices = []

        for i, (hbox, points) in tqdm(
                enumerate(zip(hboxes, hboxes_points)),
                total=len(hboxes), desc='classifying hboxes', leave=False
        ):
            output = network(points.to(device))
            violated_points_indices = torch.where(~self(output))[0].cpu()

            if len(violated_points_indices) > 0:
                hboxes_violate_indices.append(i)
                hboxes_violate_points_indices.append(violated_points_indices)
            else:
                hboxes_satisfy_indices.append(i)

        self.hboxes = hboxes
        self.hboxes_points = hboxes_points
        self.hboxes_satisfy_indices = hboxes_satisfy_indices
        self.hboxes_violate_indices = hboxes_violate_indices
        self.hboxes_violate_points_indices = hboxes_violate_points_indices

        if label is not None:
            self.save(filename)

        return self
        # return hboxes, hboxes_points, hboxes_satisfy_indices, hboxes_violate_indices

    def split(self, num_repair, shuffle=True, seed=0):
        hboxes_violate_indices = self.hboxes_violate_indices
        hboxes_violate_points_indices = self.hboxes_violate_points_indices

        if shuffle:
            indices = np.random.default_rng(seed).permutation(len(hboxes_violate_indices))
            hboxes_violate_indices = [hboxes_violate_indices[idx] for idx in indices]
            hboxes_violate_points_indices = [hboxes_violate_points_indices[idx] for idx in indices]

        repair_set = type(self)()
        repair_set.hboxes = self.hboxes
        repair_set.hboxes_points = self.hboxes_points
        repair_set.hboxes_satisfy_indices = self.hboxes_satisfy_indices
        repair_set.hboxes_violate_indices = hboxes_violate_indices[:num_repair]
        repair_set.hboxes_violate_points_indices = hboxes_violate_points_indices[:num_repair]

        gen_set = type(self)()
        gen_set.hboxes = self.hboxes
        gen_set.hboxes_points = self.hboxes_points
        gen_set.hboxes_satisfy_indices = self.hboxes_satisfy_indices
        gen_set.hboxes_violate_indices = hboxes_violate_indices[num_repair:]
        gen_set.hboxes_violate_points_indices = hboxes_violate_points_indices[num_repair:]

        return repair_set, gen_set

    def partition(self, normalize_input, *, h=.1, gap=.0, device=torch.device('cpu')):
        """Partition the valid input polytopes of this property into hboxes
        given the size of box and gap between boxes.

        Args:
            normalize_input (_type_): _description_ h (float, optional):
            _description_. Defaults to .1. gap (float, optional): _description_.
            Defaults to .0.

        Returns:
            _type_: _description_
        """
        boxes = []
        for valid_box in self.input_polytopes:
            valid_box = torch.from_numpy(normalize_input(valid_box.T).T).double().to(device)
            boxes.append(st.partition_hbox(valid_box, h=h, gap=gap))
        return torch.cat(boxes, 0)

    def intersect(self, boxes, normalize_input):
        """_summary_

        Args:
            boxes (Tensor[N, 5, 2]): _description_
            normalize_input (_type_): _description_

        Returns:
            _type_: _description_
        """
        assert self.input_polytopes.ndim == 3
        assert isinstance(boxes, torch.Tensor)
        assert boxes.ndim == 3 and boxes.shape[1] == 5 and boxes.shape[2] == 2

        intersections = []
        for valid_box in self.input_polytopes:
            valid_box = torch.from_numpy(normalize_input(valid_box.T).T)

            for box in boxes:
                intersection = st.intersect_hbox(box, valid_box)
                if intersection is not None:
                    intersections.append(intersection)

        return intersections

    def filter_applicable(self, model_key, points, denormalize_input=None):
        if model_key not in self.model_keys:
            raise NotImplementedError

        if denormalize_input is None:
            _, _, denormalize_input = acas(*model_key)

        denormalized_points = denormalize_input(points)
        in_polytope_mask = torch.stack(
            tuple(
                torch.stack(
                    (polytope[:,0] <= denormalized_points,
                     denormalized_points <= polytope[:,1])
                ).all(dim=0).all(dim=-1)
                for polytope in self.input_polytopes
            )
        ).any(dim=0)
        return points[in_polytope_mask]

    def applicable(self, points):
        applicable = []
        for point in points:
            applies = True
            for i in range(len(point)):
                bounds = self.input_polytopes[0][i]
                lb = normalize_input(bounds[0])[i]
                ub = normalize_input(bounds[1])[i]
                if (point[i] < lb and not np.allclose(point[i], lb)) or (point[i] > ub and not np.allclose(point[i], ub)):
                    applies = False
            applicable.append(applies)
        return np.asarray(applicable)

""" Helper functions to preprocess (normalize) and reset (denormalize) ACAS Xu input.
(i) ρ: Distance from ownship to intruder; (ii) θ: Angle to intruder relative to
ownship heading direction; (iii) ψ: Heading angle of intruder relative to ownship
heading direction; (iv) vown: Speed of ownship; (v) vint: Speed of intruder; (vi) τ :
Time until loss of vertical separation; and (vii) aprev: Previous advisory.
"""
# def _acas_preprocess_closure_constructor():
#     mins = np.array([0.0, -3.141593, -3.141593, 100.0, 0.0])
#     maxes = np.array([60760.0, 3.141593, 3.141593, 1200.0, 1200.0])
#     means = np.array([1.9791091e+04, 0.0, 0.0, 650.0, 600.0])
#     std_deviations = np.array([60261.0, 6.28318530718, 6.28318530718, 1100.0, 1200.0])

#     def normalize_input(i):
#         return ((np.clip(i, mins, maxes) - means) / std_deviations)

#     def denormalize_input(i):
#         return ((i * std_deviations) + means)

#     return normalize_input, denormalize_input

# normalize_input, denormalize_input = _acas_preprocess_closure_constructor()
# del _acas_preprocess_closure_constructor

RHO, THETA, PSI, V_OWN, V_INT = list(range(5))

LB = [0.0, -3.141593, -3.141593, 100.0, 0.0]
UB = [60760.0, 3.141593, 3.141593, 1200.0, 1200.0]

COC, WL, WR, SL, SR = list(range(5))

class Property_1(Property):
    def __init__(self):
        self.model_keys = all_model_keys()

        """ Input constraints: ρ ≥ 55947.691, vown ≥ 1145, vint ≤ 60. """
        self.input_polytopes = np.asarray([[
            [55947.691,  UB[  RHO]], # ρ ≥ 55947.691
            [LB[THETA],  UB[THETA]], # theta
            [LB[  PSI],  UB[  PSI]], # psi
            [    1145.,  UB[V_OWN]], # vown ≥ 1145
            [LB[V_INT],        60.], # vint ≤ 60
        ]])

    def output_constraints(self, output: SymbolicArray):
        """ Desired output property: the score for COC is at most 1500. """
        return output[..., COC] <= 1500.

class Property_2(Property):
    def __init__(self):
        self.model_keys = all_model_keys(lambda keys: keys[0] >= 2)

        """ Input constraints: ρ ≥ 55947.691, vown ≥ 1145, vint ≤ 60. """
        self.input_polytopes = np.asarray([[
            [55947.691,  UB[  RHO]], # ρ ≥ 55947.691
            [LB[THETA],  UB[THETA]], # theta
            [LB[  PSI],  UB[  PSI]], # psi
            [    1145.,  UB[V_OWN]], # vown ≥ 1145
            [LB[V_INT],        60.], # vint ≤ 60
        ]])

    def output_constraints(self, output: SymbolicArray):
        """ Desired output property: the score for COC is not the maximal score. """
        if isinstance(output, SymbolicArray):
            constr = SymbolicGurobiConstrArray(
                    np.asarray([output[..., other] > output[..., COC] for other in [WL, WR, SL, SR]]),
            output.solver).any(axis=0)
            assert constr.shape[0] == output.shape[0]
            return constr
        else:
            return output.argmax(-1) != COC

class Property_3(Property):
    def __init__(self):
        self.models = all_model_keys(lambda keys: keys not in [(1,7), (1,8), (1,9)])

        self.input_polytopes = np.asarray([[
            [    1500.,      1800.], # 1500 ≤ ρ ≤ 1800
            [    -0.06,       0.06], # −0.06 ≤ θ ≤ 0.06
            [     3.10,  UB[  PSI]], # ψ ≥ 3.10
            [     980.,  UB[V_OWN]], # vown ≥ 980
            [     960.,  UB[V_INT]], # vint ≥ 960
        ]])

    def output_constraints(self, output: SymbolicArray):
        """ Desired output property: the score for COC is not the minimal score. """
        if isinstance(output, SymbolicArray):
            constr = SymbolicGurobiConstrArray(
                    np.asarray([output[..., other] < output[..., COC] for other in [WL, WR, SL, SR]])
            , output.solver).any(axis=0)
            assert constr.shape[0] == output.shape[0]
            return constr
        else:
            return output.argmin(-1) != COC

class Property_4(Property):
    def __init__(self):
        self.models = all_model_keys(lambda keys: keys not in [(1,7), (1,8), (1,9)])

        self.input_polytopes = np.asarray([[
            [    1500.,      1800.], # 1500 ≤ ρ ≤ 1800
            [    -0.06,       0.06], # −0.06 ≤ θ ≤ 0.06
            [       0.,         0.], # ψ = 0
            [    1000.,  UB[V_OWN]], # vown ≥ 1000
            [     700.,       800.], # 700 ≤ vint ≤ 800
        ]])

    def output_constraints(self, output: SymbolicArray):
        """ Desired output property: the score for COC is not the minimal score. """
        if isinstance(output, SymbolicArray):
            constr = SymbolicGurobiConstrArray(
                    np.asarray([output[..., other] < output[..., COC] for other in [WL, WR, SL, SR]])
            , output.solver).any(axis=0)
            assert constr.shape[0] == output.shape[0]
            return constr
        else:
            return output.argmin(-1) != COC

class Property_5(Property):
    def __init__(self):
        self.models = all_model_keys(lambda keys: keys == (1,1))

        self.input_polytopes = np.asarray([[
            [     250.,      400.], # 250 ≤ ρ ≤ 400
            [      0.2,       0.4], # 0.2 ≤ θ ≤ 0.4
            [-3.141592, -3.141592 + 0.005], # −3.141592 ≤ ψ ≤ −3.141592 + 0.005
            [     100.,      400.], # 100 ≤ vown ≤ 400
            [       0.,      400.], # 0 ≤ vint ≤ 400
        ]])

    def output_constraints(self, output: SymbolicArray):
        """ Desired output property: the score for “strong right” is the minimal score. """
        raise NotImplementedError
        label = output.argmin(-1)
        return np.asarray(label == SR)

class Property_6(Property):
    def __init__(self):
        self.models = all_model_keys(lambda keys: keys == (1,1))

        # 12000 ≤ ρ ≤ 62000, (0.7 ≤ θ ≤ 3.141592) ∨ (−3.141592 ≤
        # θ ≤ −0.7), −3.141592 ≤ ψ ≤ −3.141592 + 0.005, 100 ≤ vown ≤ 1200,
        # 0 ≤ vint ≤ 1200.
        self.input_polytopes = np.asarray([
            [
                [    12000.,   62000.], # 12000 ≤ ρ ≤ 62000
                [       0.7, 3.141592], # 0.7 ≤ θ ≤ 3.141592
                [ -3.141592, -3.141592 + 0.005], # −3.141592 ≤ ψ ≤ −3.141592 + 0.005
                [      100.,    1200.], # 100 ≤ vown ≤ 1200
                [        0.,    1200.], # 0 ≤ vint ≤ 1200
            ],
            [
                [    12000.,   62000.], # 12000 ≤ ρ ≤ 62000
                [ -3.141592,     -0.7], # −3.141592 ≤ θ ≤ −0.7
                [ -3.141592, -3.141592 + 0.005], # −3.141592 ≤ ψ ≤ −3.141592 + 0.005
                [      100.,    1200.], # 100 ≤ vown ≤ 1200
                [        0.,    1200.], # 0 ≤ vint ≤ 1200

            ]
        ])

    def output_constraints(self, output: SymbolicArray):
        """ Desired output property: the score for COC is the minimal score."""
        raise NotImplementedError
        label = output.argmin(axis=-1)
        return np.asarray(label == COC)

    @overload
    def applicable(self, points): ...

    def applicable(self, points):
        applicable = []
        bounds1 = self.input_polytopes[0]
        bounds2 = self.input_polytopes[1]
        for point in points:
            applies = True
            for i in range(len(point)):
                lb1 = normalize_input(bounds1[i][0])[i]
                ub1 = normalize_input(bounds1[i][1])[i]
                lb2 = normalize_input(bounds2[i][0])[i]
                ub2 = normalize_input(bounds2[i][1])[i]
                if (point[i] < lb1 and not np.allclose(point[i], lb1)) or (point[i] > ub1 and not np.allclose(point[i], ub1)) and \
                   (point[i] < lb2 and not np.allclose(point[i], lb2)) or (point[i] > ub2 and not np.allclose(point[i], ub2)) :
                    applies = False
            applicable.append(applies)
        return np.asarray(applicable)

class Property_7(Property):
    def __init__(self):
        self.models = all_model_keys(lambda keys: keys == (1,9))

        self.input_polytopes = np.asarray([[
            [       0.,    60760.], # 0 ≤ ρ ≤ 60760
            [-3.141592,  3.141592], # -3.141592 ≤ θ ≤ 3.141592
            [-3.141592,  3.141592], # −3.141592 ≤ ψ ≤ 3.141592
            [     100.,     1200.], # 100 ≤ vown ≤ 1200
            [       0.,     1200.], # 0 ≤ vint ≤ 1200
        ]])

    def output_constraints(self, output):
        raise NotImplementedError
        if isinstance(output, SymbolicArray):
            not_SR = SymbolicGurobiConstrArray(np.asarray([output[..., other] < output[..., SR] for other in [COC, WL, WR]]), output.solver).any(axis=-1)
            not_SL = SymbolicGurobiConstrArray(np.asarray([output[..., other] < output[..., SL] for other in [COC, WL, WR]]), output.solver).any(axis=-1)
            return np.concatenate((not_SR, not_SL), axis=0)
        else:
            label = output.argmin(axis=-1)
            return np.stack((label != SL, label != SR)).all(axis=0)

class Property_8(Property):
    def __init__(self):
        self.models = all_model_keys(lambda keys: keys == (2, 9))

        self.input_polytopes = np.asarray([[
            [       0.,     60760.], # 0 ≤ ρ ≤ 60760
            [-3.141592, -0.75*3.141592], # −3.141592 ≤ θ ≤ -0.75 * 3.141592
            [     -0.1,        0.1], # -0.1 ≤ ψ ≤ 0.1
            [     600.,  UB[V_OWN]], # 600 ≤ vown ≤ 1200
            [     600.,  UB[V_INT]], # 600 ≤ vint ≤ 1200
        ]])

    def output_constraints(self, output):
        """Desired output property: the score for “weak left” is minimal or the score for COC is minimal."""
        label = output.argmin(-1)
        if isinstance(output, SymbolicArray):
            # return SymbolicArray.stack((label == COC, label == WL)).any(axis=0)
            return (
                output[...,[COC, WR, SL, SR]].argmin(axis=-1) == 0,
                output[...,[ WL, WR, SL, SR]].argmin(axis=-1) == 0,
            )

        elif isinstance(output, torch.Tensor):
            return torch.isin(label, torch.tensor([COC, WL]).to(label.device))

        else:
            return np.stack((label == COC, label == WL)).any(axis=0)

class Property_9(Property):
    def __init__(self):
        self.models = all_model_keys(lambda keys: keys == (3, 3))

        self.input_polytopes = np.asarray([[
            [    2000.,      7000.], # 2000 ≤ ρ ≤ 7000
            [     -0.4,      -0.14], # −0.4 ≤ θ ≤ -0.14
            [-3.141592, -3.141592+0.01], # -3.141592 ≤ ψ ≤ -3.141592 + 0.01
            [     100.,       150.], # 100 ≤ vown ≤ 150
            [       0.,       150.], # 0 ≤ vint ≤ 150
        ]])
    def output_constraints(self, output):
        """Desired output property: the score for “strong left” is minimal."""
        raise NotImplementedError
        return output.argmin(axis=-1) == SL

class Property_10(Property):
    def __init__(self):
        self.models = all_model_keys(lambda keys: keys == (4, 5))

        self.input_polytopes = np.asarray([[
            [   36000.,     60760.], # 36000 ≤ ρ ≤ 60760
            [      0.7,   3.141592], # 0.7 ≤ θ ≤ 3.141592
            [-3.141592, -3.141592+0.01], # -3.141592 ≤ ψ ≤ -3.141592 + 0.01
            [     900.,      1200.], # 900 ≤ vown ≤ 1200
            [     600.,      1200.], # 600 ≤ vint ≤ 1200
        ]])
    def output_constraints(self, output):
        """Desired output property: the score for COC is minimal."""
        raise NotImplementedError
        return output.argmin(axis=-1) == COC


DICT = {
    1: Property_1,
    2: Property_2,
    3: Property_3,
    4: Property_4,
    5: Property_5,
    6: Property_6,
    7: Property_7,
    8: Property_8,
    9: Property_9,
    10: Property_10
}

def property(no) -> Property:
    return DICT[no]()
