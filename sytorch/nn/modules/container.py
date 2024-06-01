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
