from __future__ import annotations
from concurrent.futures import ProcessPoolExecutor
import copy
from typing import Any, Generator, Iterable, Optional, Tuple, TypeVar, Union, overload, Literal
from collections import OrderedDict
import torch
import torch.nn as nn

from torch import Tensor
from ...solver import *
from ..symbolic_mode import *
from ..parameter import Parameter
from ..jacobian import jacobian_wrt_params
from sytorch.pervasives import *
# from sytorch.nn.utils import to_eran

__all__ = [
# """ symbolic """
    'Module',

    'Layer',
    'LinearLayer',
    'Layer2D',

    'NonLinearLayer',

    'Container',

# """ decoupled """
    'Decoupled',
    'DecoupledLayer',
    'DecoupledLinearLayer',
    'DecoupledNonLinearLayer',

    'DecoupledContainer',

# """ symbolic decoupled """
    'SymbolicDecoupled',

    'SymbolicDecoupledLayer',
    'SymbolicDecoupledLinearLayer',
    'SymbolicDecoupledNonLinearLayer',

    'SymbolicDecoupledContainer',

    'ONNXCompatibleModule',
    'ONNXContainer',
]

""" symbolic """

T = TypeVar('T', bound='Module')
class Module(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.symbolic_mode = False
        self.executor = None
        self.row_mask = None

    """ symbolic parameter management """
    def register_symbolic_parameter(self, name, value) -> None:
        if '_parameters' not in self.__dict__:
            raise AttributeError(
                "cannot assign parameter before Module.__init__() call")

        self._parameters[name] = Parameter(value)

    def __setattr__(self, name, value) -> None:
        super().__setattr__(name, value)

        if isinstance(value, nn.Parameter):
            self.register_symbolic_parameter(name, value)

    """ conversions """
    def z3(self, solver: Optional['Z3Solver']=None) -> T:
        """ like `nn.cuda()`, `nn.cpu()`, etc. """
        raise NotImplementedError(
            "unsupported backend solver switching for ddnn.Module."
        )

    def gurobi(self, solver: Optional['GurobiSolver']=None) -> T:
        """ like `nn.cuda()`, `nn.cpu()`, etc. """
        raise NotImplementedError(
            "unsupported backend solver switching for ddnn.Module."
        )

    @overload
    def to(self: T, device: Optional[Union[int, 'device']] = ..., dtype: Optional[Union['dtype', str]] = ...,
           non_blocking: bool = ...) -> T: ...

    @overload
    def to(self: T, dtype: Union['dtype', str], non_blocking: bool = ...) -> T: ...

    @overload
    def to(self: T, tensor: 'Tensor', non_blocking: bool = ...) -> T: ...

    @overload
    def to(self: T, solver: Solver) -> T: ...

    @overload
    def to(self: T, stype: stype) -> T: ...

    def to(self, *args, **kwargs):
        if (len(args) > 0 and isinstance(args[0], stype)) or 'stype' in kwargs:
            raise NotImplementedError

        if len(args) > 0 and isinstance(args[0], Solver):
            assert 'solver' not in kwargs
            kwargs['solver'] = args[0]
            args = args[1:]

        if 'solver' in kwargs:
            solver = kwargs.pop('solver')
            for param in self.parameters():
                param.to(solver=solver)

        if len(args) > 0 and isinstance(args[0], ProcessPoolExecutor):
            assert 'executor' not in kwargs
            kwargs['executor'] = args[0]
            args = args[1:]

        if 'executor' in kwargs:
            self.executor = kwargs.pop('executor')
            for m in self.children():
                m.to(self.executor)

        return super().to(*args, **kwargs)

    """ modes """
    def symbolic_(self: T,
        mode: Literal['jacobian', 'forward'] | bool=True
    ) -> T:
        self.symbolic_mode = mode
        if mode is not False:
            self.train(mode=False)
        for module in self.children():
            if isinstance(module, Module):
                module.symbolic_(mode=mode)
        return self

    def no_symbolic_(self: T) -> T:
        return self.copy().symbolic_(mode=False)

    def repair(self: T, mode: bool=True):
        return self.symbolic_(mode=mode)

    """ Note(anonymous): for now the training mode and symbolic mode are
    controlled separately.
    """
    # def train(self: T, mode: bool=True) -> T:
    #     return super().train(mode=mode)

    # def eval(self:T) -> T:
    #     return super().eval()

    @overload
    def requires_symbolic_(self: T, mode: bool=True, lb: float=None, ub: float=None) -> T: ...

    def requires_symbolic_(self: T, *args, **kwargs) -> T:
        for param in self.parameters():
            param.requires_symbolic_(*args, **kwargs)
        return self

    """ properties """
    def symbolic_parameters(self) -> Generator[Parameter]:
        """ Returns a generator yielding all parameters whose requires_symbolic
        is True.
        """
        return filter(lambda param: param.requires_symbolic, self.parameters())

    def parameter_deltas(self, concat: bool=False) -> Generator[SymbolicArray] | SymbolicArray:
        """ If `concat=False`, returns a generator yielding the `.delta` of all
        `.symbolic_parameters()`. Otherwise returns the concatenation of
        flattened deltas.
        """
        if concat:
            return SymbolicArray.concatenate(
                tuple(param.delta.flatten() for param in self.symbolic_parameters())
            )
        return map(lambda param: param.delta, self.symbolic_parameters())

    """ symbolic computations """
    def jacobian(self, input) -> Tensor | Iterable[Tensor]:
        """ Returns the Jacobian w.r.t. the input. """
        candidates = tuple(self.symbolic_parameters())
        with no_symbolic():
            t0 = timer()
            Jacobians = jacobian_wrt_params(self, input, candidates)
            # print(f"Jacobian time: {timer() - t0:.3f}s.")
        # print(input.shape)
        # for j in Jacobians:
        #     print(j.shape)
        J = torch.cat([
            J.flatten(start_dim=-candidate.ndim) if candidate.mask is None else
            J[(as_slice[:],) * (J.ndim - candidate.ndim) + (candidate.mask,)]
            for candidate, J in zip(candidates, Jacobians)
        ], dim=-1).cpu().detach().numpy()
        return J

    def forward(self, *args, **kwargs) -> Any:
        if self.symbolic_mode and is_symbolic_enabled():
            # NOTE(anonymous): assuming there is always just one input array.
            if len(args) < 2 and 'pattern' not in kwargs:
                kwargs['pattern'] = self.activation_pattern(*args)
            return self.forward_symbolic(*args, **kwargs)
        return super().forward(*args, **kwargs)

    def forward_symbolic(self, *args, **kwargs) -> SymbolicArray:
        raise NotImplementedError(
            f"unimplemented .forward_symbolic(...) for {type(self)}"
        )

    def update_(self: T, src: T=None) -> T:
        """ Inplace update symbolic parameters with deltas.

        Parameters
        ==========
        src:    The source of deltas, where None means take deltas from self. By
                default it's None.
        """
        if src is None:
            src = self
        for param, param_ref in zip(self.symbolic_parameters(), src.symbolic_parameters()):
            param.update_(src=param_ref)
        return self

    def decouple(self: T) -> SymbolicDecoupled:
        if isinstance(self, Decoupled):
            return self

        raise NotImplementedError(
            f"unimplemented .decouple for {self}({type(self)})"
        )

    def deepcopy(self: T) -> T:
        return copy.deepcopy(self)

    def copy(self: T) -> T:
        return copy.copy(self)

    def activation_pattern(self: T, input):
        raise NotImplementedError(
            f"unimplemented activation_pattern for {type(self)}."
        )

    def save(self, path):
        return torch.save(self.state_dict(), path)

    def load(self, path):
        self.load_state_dict(torch.load(path, map_location=self.device))
        return self

    def to_eran(self, *args, **kwargs):
        import sytorch.nn.utils
        return sytorch.nn.utils.to_eran(self, *args, **kwargs)

    def to_onnx(self, sample_input=None, **kwargs):
        import sytorch.nn.utils
        return sytorch.nn.utils.to_onnx(self, sample_input=sample_input, **kwargs)

    @property
    def has_symbolic_parameter(self):
        for _ in self.symbolic_parameters():
            return True
        return False

    @property
    def dtype(self):
        for p in self.parameters():
            return p.dtype
        return None

    @property
    def device(self):
        for p in self.parameters():
            return p.device
        return None

class Layer(Module):
    def to_onnx_compatible(self):
        return self

class LinearLayer(Layer):

    def requires_symbolic_(
        self: T, mode=True,
        rows=None, row_mask=None,
        cols=None, col_mask=None,
        weight=True, bias=True,
        seed = None,
        *args, **kwargs
    ) -> T:
        if seed is not None: seed = _get_rng(seed)

        """ Handle row mask. """
        if rows is not None:
            assert row_mask is None
            if isinstance(rows, slice):
                # to bool mask
                row_mask = rows
                _row_mask = np.zeros(self.weight.shape[0], dtype=bool)
                _row_mask[row_mask] = True
                row_mask = _row_mask
            else:
                row_mask = random_mask_1d(self.weight.shape[0], rows, seed)

        elif row_mask is not None:
            # to bool mask
            if isinstance(row_mask, int):
                row_mask = as_slice[:row_mask]
            _row_mask = np.zeros(self.weight.shape[0], dtype=bool)
            _row_mask[row_mask] = True
            row_mask = _row_mask

        if row_mask is not None:
            # print(row_mask)
            self.row_mask = row_mask
            self.mask = (as_slice[...], row_mask, as_slice[:], as_slice[:])

            if weight:
                weight_mask = np.zeros(self.weight.shape, dtype=bool)

                if col_mask is not None:
                    assert cols is None
                    weight_mask[row_mask,...] = col_mask

                elif cols is not None:
                    weight_mask[row_mask,...] = np.stack(tuple(
                        random_mask_1d(self.weight.shape[1], cols, seed)
                        for _ in range(weight_mask[row_mask,...].shape[0])
                    ))[(as_slice[...],) + tuple(self.weight.shape[2:])]

                else:
                    weight_mask[row_mask,...] = True
                self.weight.requires_symbolic_(*args, mode=mode, mask=weight_mask, **kwargs)

            if bias and self.bias is not None:
                bias_mask = np.zeros(self.bias.shape, dtype=bool)
                bias_mask[row_mask, ...] = True
                self.bias.requires_symbolic_(*args, mode=mode, mask=bias_mask, **kwargs)

        else:
            for param in self.parameters():
                param.requires_symbolic_(*args, mode=mode, **kwargs)

        return self

    def activation_pattern(self: T, input):
        return []

    def decouple(self: T) -> SymbolicDecoupledLinearLayer:
        if isinstance(self, Decoupled):
            assert isinstance(self, SymbolicDecoupledLinearLayer)
            return self
        return SymbolicDecoupledLinearLayer(
            val = self.copy(),
            act = self.deepcopy().requires_symbolic_(False)
        )

    def v(self, vpolytopes, pattern=None):
        N, V, *S = vpolytopes.shape
        out_vpolytopes = self.forward_symbolic(
            vpolytopes.reshape(N * V, *S),
            pattern = pattern
        )
        out_vpolytopes = out_vpolytopes.reshape(N, V, *out_vpolytopes.shape[1:])
        return out_vpolytopes

class Layer2D(Layer): ...

class NonLinearLayer(Layer):
    def v(self:T, vpolytopes, pattern):
        N, V, *S = vpolytopes.shape
        N, *SP = pattern.shape
        out_vpolytopes = self.forward_symbolic(
            vpolytopes.reshape(N * V, *S),
            pattern = broadcast_at(pattern, 1, (V,)).reshape(N * V, *SP),
        )
        out_vpolytopes = out_vpolytopes.reshape(N, V, *out_vpolytopes.shape[1:])
        return out_vpolytopes


class Container(Module):

    """ Symboloc Module + .decouple for containers. """
    def decouple(self: T) -> SymbolicDecoupledContainer:
        raise NotImplementedError(
            f"unimplemented .decouple for {self}({type(self)})"
        )


""" Derive decoupled module `Decoupled` from `torch.nn.Module`. """

from . import functional

class Decoupled(nn.Module):
    def forward(self, input_val):
        if self.symbolic_mode and is_symbolic_enabled():
            return self.forward_symbolic(input_val)
        output_val, _ = self.forward_decoupled(input_val, input_val)
        return output_val

    def output_delta(self, *args, **kwargs) -> SymbolicArray:
        """ Returns the symbolic output delta w.r.t. the input. """
        deltas = self.parameter_deltas(concat=True) #SymbolicArray.concatenate([d.flatten() for d in self.parameter_deltas()])
        return self.jacobian(*args, **kwargs) @ deltas

    def forward_symbolic(self, input_val):
        with no_symbolic():
            _symbolic_param = next(self.symbolic_parameters()).array_if_symbolic()
            if isinstance(_symbolic_param, SymbolicGurobiArray):
                deltas = self.parameter_deltas(concat=True)
                J = torch.from_numpy(self.jacobian(input_val))
                N, O, D = J.shape
                return functional.linear(
                    deltas.reshape(1, D),
                    J.reshape(N*O, D),
                    self(input_val).reshape(N*O,) # (N*O)
                ).reshape(N, O)
            elif isinstance(_symbolic_param, SymbolicLightningArray):
                with no_symbolic(), torch.no_grad():
                    y0 = self(input_val)
                return y0 + self.output_delta(input_val)
            else:
                raise RuntimeError()


    def forward_decoupled(self, input_val: Tuple, input_act: Tuple) -> Tuple[Any, Any]:
        raise NotImplementedError(
            f"unimplemented .forward_decoupled for {self}({type(self)})"
        )

    def decouple(self: T) -> T:
        return self


""" decoupled non-container layers """

class DecoupledLayer(Decoupled):
    def __init__(self: T, val: Module, act: Module) -> None:
        super().__init__()
        self.val = val
        self.act = act

class DecoupledLinearLayer(DecoupledLayer):
    def forward_decoupled(self, input_val: Tensor, input_act: Tensor) -> Tuple[Tensor, Tensor]:
        # TODO(anonymous): Support inputs/outputs other than a single Tensor.
        # https://github.com/95616ARG/indra/issues/799
        return (
            self.val(input_val),
            self.act(input_act)
        )

class DecoupledNonLinearLayer(DecoupledLayer): ...

""" decoupled containers """

class DecoupledContainer(Decoupled):
    def __init__(self: T, *args: Module, **kwargs):
        if len(args) == 1 and isinstance(args[0], OrderedDict):
            super().__init__(OrderedDict([(k, v.decouple()) for k, v in args[0].items()]), **kwargs)
        else:
            super().__init__(*tuple(module.decouple() for module in args), **kwargs)

""" symbolic decoupled -> symbolic.Module -> Decoupled """
class SymbolicDecoupled(Decoupled, Module): ...
class SymbolicDecoupledLayer(DecoupledLayer, Layer): ...
class SymbolicDecoupledLinearLayer(DecoupledLinearLayer, LinearLayer): ...
class SymbolicDecoupledNonLinearLayer(DecoupledNonLinearLayer, NonLinearLayer): ...
class SymbolicDecoupledContainer(DecoupledContainer, Container): ...

class ONNXCompatibleModule:
    def to_onnx_compatible(self):
        return self

class ONNXContainer(Module, ONNXCompatibleModule):
    def __init__(self, net, eran=None):
        super().__init__()
        self.net = net

    def forward(self, input):
        return self.net(input)

    def to_onnx_compatible(self):
        raise NotImplementedError("onnx2torch->onnx is not tested.")
        return self.net
