from __future__ import annotations
from typing import Any, Callable, Iterable, Iterator, Tuple, Union, overload, Literal
from collections import OrderedDict
from itertools import islice
import operator
import warnings

import torch
import torch.nn as nn
from torch._jit_internal import _copy_to_script_wrapper

from sytorch.pervasives import *
from sytorch.nn.symbolic_mode import no_symbolic
from sytorch.solver import lightning
from sytorch.solver.lightning import SymbolicLightningArray
from sytorch.solver.symbolic_array import SymbolicGurobiArray
from .module import *
from .module import T
from .linear import *
from .reshape import Flatten

__all__ = [
# """ symbolic """

    'Sequential',
    'Parallel',
    'Residual',

# """ decoupled """
    'DecoupledSequential',
    'DecoupledParallel',
    'DecoupledResidual',

# """ symbolic decoupled """
    'SymbolicDecoupledSequential',
    'SymbolicDecoupledParallel',
    'SymbolicDecoupledResidual'
]

class Sequential(Container, nn.Sequential):

    def v(self, vpolytopes, pattern=None):
        if pattern is None:
            pattern = self.activation_pattern(centroid_of_vpolytopes(vpolytopes))

        for module, module_pattern in vtqdm2(zip(self, pattern), total=len(self), desc="VSequential", leave=False):
            vpolytopes = module.v(vpolytopes, pattern=module_pattern)

        return vpolytopes

    def requires_symbolic_weight_and_bias(self, *args, **kwargs):
        is_sym = False
        for l in self:
            assert isinstance(l, Layer), "for now this method only support flat sequence."
            if isinstance(l, NonLinearLayer):
                continue
            is_sym = is_sym or l.has_symbolic_parameter
            if hasattr(l, 'weight') and not is_sym:
                l.weight.requires_symbolic_(*args, **kwargs)
                is_sym=True
            if hasattr(l, 'bias') and is_sym:
                l.bias.requires_symbolic_(*args, **kwargs)
                is_sym=True
        return self

    def to_onnx_compatible(self):
        return type(self)(
            *(module.to_onnx_compatible() for module in self)
        )

    def activation_pattern(self: T, input):
        A = []
        with no_symbolic(), torch.no_grad():
            for module in self:
                A.append(module.activation_pattern(input))
                input = module(input)
        return A

    def forward_symbolic(self:T, input, pattern):
        for module, module_pattern in vtqdm2(zip(self, pattern), total=len(self), desc="Sequential", leave=False):
            input = module.forward_symbolic(input, pattern=module_pattern)
        return input

    def decouple(self: T) -> SymbolicDecoupledSequential:
        if isinstance(self, Decoupled):
            assert isinstance(self, SymbolicDecoupledSequential)
            return self
        return SymbolicDecoupledSequential(*self)

    @_copy_to_script_wrapper
    def __getitem__(self, idx) -> Union['Sequential', T]:
        if not isinstance(idx, tuple):
            idx = (idx,)

        item = super().__getitem__(idx[0])
        if isinstance(idx[0], slice):
            item.to(executor=self.executor).symbolic_(mode=self.symbolic_mode)

        if len(idx) > 1:
            return item[idx[1:]]
        else:
            return item

    def h_symbolic(self, boxes, ref=None, points=None):
        if points is not None:
            print("h with ref points")
            if isinstance(points, tuple):
                print("h with ref points from another network")
                _n, points = points
                for module, _m in zip(self, _n):
                    boxes = module.h(boxes, points=points)
                    with no_symbolic():
                        points = _m(points)
            else:
                for module in self:
                    boxes = module.h(boxes, points=points)
                    with no_symbolic():
                        points = module(points)
        elif ref is not None:
            ref_idx = 0
            for module in tqdm(self):
                if not isinstance(module, (Flatten, NormalizeInput)):
                    boxes = module.h(boxes, ref=ref[ref_idx])
                    ref_idx += 1
                else:
                    boxes = module.h(boxes)
        else:
            assert points is None and ref is None
            for module in self:
                boxes = module.h(boxes)
        return boxes

    # h2 = h_symbolic

    def h(self, boxes, domain='deeppoly', points=None, **kwargs):
        if self.symbolic_mode is True:
            end = 0
            if isinstance(boxes, Tensor):
                end = 0
                # for il, _l in enumerate(self):
                #     if _l.has_symbolic_parameter:
                #         break
                #     elif isinstance(_l, LinearLayer):
                #         end = il+1
                while end < len(self) and not self[end].has_symbolic_parameter:
                    end += 1

                layer_boxes = self[:end].h_concrete(boxes, domain=domain)
                boxes = layer_boxes[-1]
                if points is not None:
                    with no_symbolic():
                        if isinstance(points, tuple):
                            _n, points = points
                            points = _n[end:], _n[:end](points)
                        else:
                            points = self[:end](points)
            return self[end:].h_symbolic(boxes, points=points)

        else:
            return self.h_concrete(boxes, domain=domain)
            raise NotImplementedError

    # def get_abstract0_point(net, point):
    #     lbs, ubs = []
    #     for layer in net:
    #         point = net(point)
    #         lbs.append(point.detach().clone())
    #         ubs.append(point.detach().clone())

    def h_concrete(self, boxes, domain='deeppoly', **kwargs):
        device, dtype = self.device, self.dtype
        seq = self

        if len(seq) == 0: return [boxes]
        boxes = boxes.clone()
        if isinstance(seq[0], NormalizeInput):
            boxes[...,0] = seq[0](boxes[...,0])
            boxes[...,1] = seq[0](boxes[...,1])
            seq = seq[1:]

        if len(seq) == 0: return [boxes]
        if isinstance(seq[0], Flatten):
            boxes = boxes.flatten(1,-2)
            seq = seq[1:]

        if self.symbolic_mode is True:
            for end, module in enumerate(seq):
                assert not module.has_symbolic_parameter
                # assert not isinstance(seq[0], NormalizeInput)
                # if module.has_symbolic_parameter:
                #     seq = self[:end]
                #     break

        print(f"will deeppoly on {seq}")
        if len(seq) == 0: return [boxes]

        assert not isinstance(seq[0], NormalizeInput)
        layer_outputs = []
        layer_out_transpose = []
        layer_out_shapes = []
        layer_boxes = []
        with no_symbolic(), torch.no_grad():
            sample_output = center_of_hboxes(boxes)
            for m in seq:
                assert isinstance(m, Layer)
                sample_output = m(sample_output)
                if isinstance(m, Flatten):
                    continue
                layer_outputs.append(sample_output)
                layer_boxes.append([])
                if sample_output.ndim == 4:
                    layer_out_shapes.append(tuple(torch.moveaxis(sample_output, -3, -1).shape[1:]))
                    layer_out_transpose.append((2,0,1))
                elif sample_output.ndim == 2:
                    layer_out_shapes.append(tuple(sample_output.shape[1:]))
                    layer_out_transpose.append((0,))
                else:
                    raise NotImplementedError

        seq_eran = seq.to_eran(sample_input=boxes[...,0])
        boxes0 = boxes
        boxes = seq[0].to_eran_bounds(boxes)
        # output_boxes = []

        t_deeppoly = 0

        for i, box in tqdm(enumerate(boxes), total=boxes.shape[0], desc='deeppoly boxes'):
            assert box.ndim == 2
            if (box[...,0] == box[...,1]).all():
                # print("forwarding a point")
                point = boxes0[i][None,...,0]
                j = 0
                for layer in seq:
                    point = layer(point)
                    if not isinstance(layer, Flatten):
                        layer_boxes[j].append(torch.stack((point[0], point[0]), -1))
                        j += 1

            else:
                t0 = timer()
                analyzer, (element, lbs, ubs) = seq_eran.get_abstract0(
                                                    box[:,0].detach().cpu(), box[:,1].detach().cpu(), domain=domain, **kwargs)
                t_deeppoly += timer() - t0

                for lb, ub, shape, transpose, layer_box, layer_output in zip(lbs, ubs, layer_out_shapes, layer_out_transpose, layer_boxes, layer_outputs):
                    # print(shape, transpose)
                    layer_box.append(torch.stack((
                        torch.tensor(lb, dtype=dtype, device=device).reshape(shape).permute(transpose),
                        torch.tensor(ub, dtype=dtype, device=device).reshape(shape).permute(transpose),
                    ), -1))
                    assert (layer_box[-1][...,0] <= layer_output[i]).all() and (layer_output[i] <= layer_box[-1][...,1]).all()

        print(f'deeppoly time {t_deeppoly:.3f}s')

        layer_boxes = [
            torch.stack(layer_box, dim=0)
            for layer_box in layer_boxes
        ]

        if isinstance(seq[-1], Flatten):
            layer_boxes[-1] = layer_boxes[-1].reshape(layer_boxes[-1].shape[0], -1, 2)

        return layer_boxes

class _Parallel(nn.Module, ONNXCompatibleModule):
    _modules: Dict[str, Module]  # type: ignore[assignment]

    @overload
    def __init__(self, *args: Module, mode: Literal['cat', 'add']='cat') -> None:
        ...

    @overload
    def __init__(self, arg: 'OrderedDict[str, Module]', mode: Literal['cat', 'add']='cat') -> None:
        ...

    def __init__(self, *args, mode: Literal['cat', 'add'], dim=None):
        super(_Parallel, self).__init__()
        if len(args) == 1 and isinstance(args[0], OrderedDict):
            for key, module in args[0].items():
                self.add_module(key, module)
        else:
            for idx, module in enumerate(args):
                self.add_module(str(idx), module)
        assert mode in ['cat', 'add']
        if mode == 'cat':
            assert dim is not None
        self.mode = mode
        self.dim = dim

    def _get_item_by_idx(self, iterator, idx) -> T:
        """Get the idx-th item of the iterator"""
        size = len(self)
        idx = operator.index(idx)
        if not -size <= idx < size:
            raise IndexError('index {} is out of range'.format(idx))
        idx %= size
        return next(islice(iterator, idx, None))

    @_copy_to_script_wrapper
    def __getitem__(self, idx) -> Union['Sequential', T]:
        if not isinstance(idx, tuple):
            idx = (idx,)

        item = self._getitem(idx[0])
        if isinstance(idx[0], slice):
            item.to(executor=self.executor).symbolic_(mode=self.symbolic_mode)

        if len(idx) > 1:
            return item[idx[1:]]
        else:
            return item

    def _getitem(self, idx):
        if isinstance(idx, slice):
            return self.__class__(
                OrderedDict(list(self._modules.items())[idx]),
                mode = self.mode,
                dim  = self.dim
            )
        else:
            return self._get_item_by_idx(self._modules.values(), idx)

    def __setitem__(self, idx: int, module: Module) -> None:
        key: str = self._get_item_by_idx(self._modules.keys(), idx)
        return setattr(self, key, module)

    def __delitem__(self, idx: Union[slice, int]) -> None:
        if isinstance(idx, slice):
            for key in list(self._modules.keys())[idx]:
                delattr(self, key)
        else:
            key = self._get_item_by_idx(self._modules.keys(), idx)
            delattr(self, key)

    @_copy_to_script_wrapper
    def __len__(self) -> int:
        return len(self._modules)

    @_copy_to_script_wrapper
    def __dir__(self):
        keys = super(_Parallel, self).__dir__()
        keys = [key for key in keys if not key.isdigit()]
        return keys

    @_copy_to_script_wrapper
    def __iter__(self) -> Iterator[Module]:
        return iter(self._modules.values())

    # NB: We can't really type check this function as the type of input
    # may change dynamically (as is tested in
    # TestScript.test_sequential_intermediary_types).  Cannot annotate
    # with Any as TorchScript expects a more precise type
    def forward(self, input):
        output = self[0](input)

        if self.mode == 'add':
            for module in self[1:]:
                output = output + module(input)

        elif self.mode == 'cat':
            for module in self[1:]:
                output = torch.cat([output, module(input)], dim=self.dim)

        else:
            raise NotImplementedError(
                f"unsupported parallel forward mode {self.mode}."
            )

        return output

class Parallel(Container, _Parallel):

    def activation_pattern(self: T, input):
        with no_symbolic(), torch.no_grad():
            return [
                module.activation_pattern(input)
                for module in self
            ]

    def h(self, boxes):
        assert self.mode in ['cat', 'add']

        outputs = []
        output_array_type = Tensor
        for module in vtqdm2(self, total=len(self), desc="Parallel", leave=False):
            module_output = module.h(boxes)
            if not isinstance(module_output, Tensor):
                output_array_type = type(module_output)
            outputs.append(module_output)

        if self.mode == 'cat':
            assert self.dim == 1
            if issubclass(output_array_type, Tensor):
                return torch.cat(outputs, dim=self.dim)
            else:
                assert output_array_type is SymbolicLightningArray
                return lightning.concatenate(outputs, axis=self.dim)

        elif self.mode == 'add':
            if issubclass(output_array_type, Tensor):
                outputs = torch.stack(outputs, dim=-1)
                return torch.sum(outputs, dim=-1)
            else:
                assert output_array_type is SymbolicLightningArray, f"{output_array_type}"
                outputs = lightning.stack(outputs, axis=-1)
                return lightning.sum(outputs, axis=-1)

        else:
            raise NotImplementedError(
                f"unsupported parallel forward mode {self.mode}."
            )

    def forward_symbolic(self, input, pattern):
        assert self.mode in ['cat', 'add']

        outputs = []
        output_array_type = Tensor
        for module, module_pattern in vtqdm2(zip(self, pattern), total=len(self), desc="Parallel", leave=False):
            module_output = module(input, pattern=module_pattern)
            if not isinstance(module_output, Tensor):
                output_array_type = type(module_output)
            outputs.append(module_output)

        if self.mode == 'cat':
            if issubclass(output_array_type, Tensor):
                return torch.cat(outputs, dim=self.dim)
            else:
                if issubclass(output_array_type, SymbolicGurobiArray):
                    return output_array_type.concatenate(outputs, axis=self.dim)
                elif output_array_type is SymbolicLightningArray:
                    return lightning.concatenate(outputs, axis=self.dim)
                else:
                    raise RuntimeError(f"{output_array_type}")

        elif self.mode == 'add':
            if issubclass(output_array_type, Tensor):
                outputs = torch.stack(outputs, dim=-1)
                return torch.sum(outputs, dim=-1)
            else:
                if issubclass(output_array_type, SymbolicGurobiArray):
                    outputs = output_array_type.stack(outputs, axis=-1)
                    return outputs.sum(axis=-1)
                elif output_array_type is SymbolicLightningArray:
                    outputs = lightning.stack(outputs, axis=-1)
                    return lightning.sum(outputs, axis=-1)
                else:
                    raise RuntimeError(f"{output_array_type}")

        else:
            raise NotImplementedError(
                f"unsupported parallel forward mode {self.mode}."
            )

    def decouple(self: T) -> SymbolicDecoupledParallel:
        if isinstance(self, Decoupled):
            assert isinstance(self, SymbolicDecoupledParallel)
            return self

        decoupled = SymbolicDecoupledParallel(*self, mode=self.mode, dim=self.dim)
        return decoupled

def _Residual(*modules, skip, mode: Literal['cat', 'add']='add') -> _Parallel:
    return _Parallel(*modules, skip, mode=mode)


def Residual(*modules, skip, mode: Literal['cat', 'add']='add') -> Parallel:
    return Parallel(*modules, skip, mode=mode)

class DecoupledSequential(DecoupledContainer, nn.Sequential):

    def forward_decoupled(self, input_val: Tensor, input_act: Tensor) -> Tuple[Any, Any]:
        output_val, output_act = input_val, input_act
        for module in self:
            output_val, output_act = module.forward_decoupled(output_val, output_act)
        return output_val, output_act

class DecoupledParallel(DecoupledContainer, _Parallel):
    def forward_decoupled(self, input_val: Tensor, input_act: Tensor) -> Tuple[Any, Any]:
        output_val, output_act = self[0].forward_decoupled(input_val, input_act)

        if self.mode == 'add':
            for module in self[1:]:
                _output_val, _output_act = module.forward_decoupled(input_val, input_act)
                output_val = output_val + _output_val
                output_act = output_act + _output_act

        elif self.mode == 'cat':
            for module in self[1:]:
                _output_val, _output_act = module.forward_decoupled(input_val, input_act)
                output_val = torch.cat((output_val, _output_val), dim=self.dim)
                output_act = torch.cat((output_act, _output_act), dim=self.dim)

        else:
            raise NotImplementedError(
                f"unsupported parallel decoupled forward mode {self.mode}."
            )

        return output_val, output_act

# class DecoupledResidual(DecoupledParallel, _Residual): ...
def DecoupledResidual(*modules, skip, mode: Literal['cat', 'add']='add') -> DecoupledParallel:
    return DecoupledParallel(*modules, skip, mode=mode)

class SymbolicDecoupledSequential(DecoupledSequential, Sequential): ...
class SymbolicDecoupledParallel(DecoupledParallel, Parallel): ...
def SymbolicDecoupledResidual(*modules, skip, mode: Literal['cat', 'add']) -> SymbolicDecoupledParallel:
    return SymbolicDecoupledParallel(*modules, skip, mode=mode)
