from __future__ import annotations
from typing import Any, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch._jit_internal import _copy_to_script_wrapper

from sytorch.pervasives import *

from sytorch.nn.modules.module import *
from sytorch.solver import lightning
from sytorch.solver import *
from sytorch.nn.symbolic_mode import *
from .module import T

__all__ = [
    "ReLU",
    "Hardswish",
    "GELU",
    "DecoupledReLU",
    "SymbolicDecoupledReLU",
    "ArgMax",
]

class ArgMax(NonLinearLayer):
    def __init__(self, *args, **kwargs):
        raise NotImplementedError(
            f"unimplemented class {self.__class__}."
        )

def relu_symbolic_boxes(symbolic_boxes):
    """_summary_

    Args:
        symbolic_boxes (SymbolicArray[..., 2]): _description_

    Returns:
        SymbolicArray[..., 2]: _description_
    """

    solver = symbolic_boxes.solver
    out_approx_boxes = solver.reals(symbolic_boxes.shape)
    solver.add_constraints(
        # Asserts the convex linear programming space for the output upper-bound.
        out_approx_boxes[...,1] >= symbolic_boxes[...,1], # output upper-bound >= input upper-bound
        out_approx_boxes[...,1] >= 0., # output upper-bound >= 0.

        # One under-approx lower-bound -- 0.
        out_approx_boxes[...,0] == 0.,

        # Another under-approx lower-bound -- input lower-bound
        out_approx_boxes[...,0] == symbolic_boxes[...,0]

        # One MILP
        # (out_approx_boxes[...,0] == 0., out_approx_boxes[...,0] == symbolic_boxes[...,0]).any()

        # Exact ReLU MILP: check input

        # Approx lower-bound
    )

    return 1

class ReLU(NonLinearLayer, nn.ReLU, ONNXCompatibleModule):
    @property
    def mask(self):
        return None

    def h(self, boxes, ref=None, points=None):
        if isinstance(boxes, Tensor):
            return F.relu(boxes)

        elif isinstance(boxes, SymbolicLightningArray):
            LB, UB = as_slice[...,0], as_slice[...,1]
            solver = boxes.solver
            out = boxes.copy()
            out[UB] = solver.reals(out[UB].shape)
            solver.add_constraints(
                out[UB] >= boxes[UB],
                out[UB] >= 0.
            )

            if ref is None:
                if points is None:
                    print("ReLU(OUT[LB]) = IN[LB]")
                    # out[LB] = 0.
                    pass

                else:
                    assert tuple(points.shape) == tuple(boxes[LB].shape)

                    # mask_on  = (points => 0.).detach().cpu()
                    # if mask_on.any():
                    #     solver.add_constraints(boxes[mask_on,0] >= 0.)

                    threshold = 0.
                    print(f"ReLU Threshold: {threshold}")
                    mask_off = (points < threshold).detach().cpu()
                    print("ReLU Off: ", mask_off.sum(), "/", mask_off.numel())
                    # for a, b in zip(out[mask_off,0].flat, out[mask_off.numpy(),0].flat):
                    #     assert a is b
                    out[mask_off.numpy(),0] = 0.


            elif isinstance(ref, Tensor):
                raise NotImplementedError
                if ref.shape == boxes.shape:
                    # mask_on  = (ref[LB] > 0.).detach().cpu()
                    mask_off = (ref[LB] <= 0.).detach().cpu()

                    # if mask_on.any():
                    #     solver.add_constraints(boxes[mask_on,0] >= 0.)

                    out[mask_off,0] = 0.

                elif ref.shape == boxes[LB].shape:
                    # mask_on  = (ref > 0.).detach().cpu()
                    mask_off = (ref <= 0.).detach().cpu()

                    # if mask_on.any():
                    #     solver.add_constraints(boxes[mask_on,0] >= 0.)

                    out[mask_off,0] = 0.

                else:
                    raise NotImplementedError

            else:
                raise NotImplementedError

            return out

        else:
            raise NotImplementedError

    def activation_pattern(self: T, input) -> np.ndarray:
        """ Boolean array. True for >= 0, False for < 0.
        """
        with no_symbolic(), torch.no_grad():
            return (input >= 0.).cpu().detach().numpy()

    def v(self:T, vpolytopes, pattern):
        if isinstance(vpolytopes, Tensor):
            with no_symbolic():
                return F.relu(vpolytopes)

        elif isinstance(vpolytopes, (SymbolicLightningArray, SymbolicGurobiArray)):
            N, V, *S = vpolytopes.shape
            # vpolytopes = vpolytopes.reshape(N * V, -1)
            # pattern = broadcast_at(pattern, 1, (V,))
            # print(vpolytopes.reshape(N * V, -1).shape)
            # print(broadcast_at(pattern, 1, (V,)).shape)
            out_vpolytopes = self.forward_symbolic(
                vpolytopes.reshape(N * V, *S),
                pattern = broadcast_at(pattern, 1, (V,)).reshape(N * V, *S),
            )
            out_vpolytopes = out_vpolytopes.reshape(N, V, *out_vpolytopes.shape[1:])
            return out_vpolytopes

            # return lightning.stack(tuple(
            #     lightning.relu(vertices, pattern, mask=vertices.mask)
            #     for vertices in vpolytopes
            # ))

        else:
            raise NotImplementedError

    def forward_symbolic(self:T, input, pattern):
        if isinstance(input, Tensor):
            return F.relu(input)

        elif isinstance(input, SymbolicGurobiArray):
            solver = input.solver
            output = input.copy()
            mask_on = pattern == 1
            mask_off = pattern == 0
            mask_zero = pattern == -1
            output[mask_off] = solver._zero
            output[mask_zero] = solver._zero
            constrs = np.empty(output.shape, dtype=object)
            # NOTE(anonymous): I think >= or > doesn't really matter because
            # imprecision at scale 1e-10 is not surprising in float16, 32, 64
            # matrix multiplication.
            constrs[mask_off] = input[mask_off] <= 0.
            constrs[mask_on] = input[mask_on] >= 0.
            constrs[mask_zero] = input[mask_zero] == 0. # NOTE(anonymous): maybe we can relax it.
            constrs = constrs.view(SymbolicGurobiConstrArray).to(input.solver)
            handle_constrs(constrs)
            return output

        elif isinstance(input, SymbolicLightningArray):
            return lightning.relu(input, pattern, mask=input.mask)

    def decouple(self: T) -> SymbolicDecoupledReLU:
        if isinstance(self, Decoupled):
            assert isinstance(self, SymbolicDecoupledReLU)
            return self
        return SymbolicDecoupledReLU(
            val = self.copy(),
            act = self.deepcopy().requires_symbolic_(False)
        )

class Hardswish(NonLinearLayer, nn.Hardswish):
    @property
    def mask(self):
        return None

    def activation_pattern(self: T, input) -> np.ndarray:
        """ Boolean array. True for >= 0, False for < 0.
        """
        # raise NotImplementedError
        with no_symbolic(), torch.no_grad():

            ap = torch.zeros(input.shape, dtype=int)
            l, r = -0.3, 0.0
            ap[input <= l] = -1
            ap[input >= r] = 1

            return ap.detach().cpu().numpy()
            # return (input >= 0.).cpu().detach().numpy()

    def forward_symbolic(self:T, input, pattern):
        if isinstance(input, Tensor):
            return F.hardswish(input)

        elif isinstance(input, SymbolicGurobiArray):
            solver = input.solver
            output = input.copy()
            mask_pos  = pattern == 1
            mask_zero = pattern == 0
            mask_neg  = pattern == -1

            print('pos :', mask_pos .astype(float).mean())
            print('zero:', mask_zero.astype(float).mean())
            print('neg :', mask_neg .astype(float).mean())

            output[mask_neg] = solver._zero
            constrs = np.empty(output.shape, dtype=object)
            # NOTE(anonymous): I think >= or > doesn't really matter because
            # imprecision at scale 1e-10 is not surprising in float16, 32, 64
            # matrix multiplication.
            # import pdb; pdb.set_trace()
            constrs[mask_pos ] = input[mask_pos ] >=  3.
            if mask_zero.any():
                output[mask_zero] = solver._zero
                constrs[mask_zero] = input[mask_zero].abs_ub() <= epsilon.eps
            # constrs[mask_zero] = input[mask_zero] == 0.
            constrs[mask_neg ] = input[mask_neg ] <= -3.
            constrs = constrs.view(SymbolicGurobiConstrArray).to(input.solver)
            handle_constrs(constrs)
            return output

        elif isinstance(input, SymbolicLightningArray):
            raise NotImplementedError

class GELU(NonLinearLayer, nn.GELU):
    @property
    def mask(self):
        return None

    def activation_pattern(self: T, input) -> np.ndarray:
        """ Boolean array. True for >= 0, False for < 0.
        """
        # raise NotImplementedError
        with no_symbolic(), torch.no_grad():

            ap = torch.zeros(input.shape, dtype=int)
            # l, r = -0.5, 0.0
            l = -2.0
            r = l
            print(f'l={l}, r={r}')
            ap[input <= l] = -1
            ap[input >= r] = 1

            return ap.detach().cpu().numpy()
            # return (input >= 0.).cpu().detach().numpy()

    def forward_symbolic(self:T, input, pattern):
        if isinstance(input, Tensor):
            return F.gelu(input)

        elif isinstance(input, SymbolicGurobiArray):
            solver = input.solver
            output = input.copy()
            mask_pos  = pattern == 1
            mask_zero = pattern == 0
            mask_neg  = pattern == -1

            print('pos :', mask_pos .astype(float).mean())
            print('zero:', mask_zero.astype(float).mean())
            print('neg :', mask_neg .astype(float).mean())

            output[mask_neg] = solver._zero
            constrs = np.empty(output.shape, dtype=object)
            # NOTE(anonymous): I think >= or > doesn't really matter because
            # imprecision at scale 1e-10 is not surprising in float16, 32, 64
            # matrix multiplication.
            # import pdb; pdb.set_trace()
            # l = -5.4
            # r = 5.4
            l = -5.
            r = 5.
            constrs[mask_pos ] = input[mask_pos ] >= r
            if mask_zero.any():
                output[mask_zero] = solver._zero
                constrs[mask_zero] = input[mask_zero].abs_ub() <= epsilon.eps
            # constrs[mask_zero] = input[mask_zero] == 0.
            constrs[mask_neg ] = input[mask_neg ] <= l
            constrs = constrs.view(SymbolicGurobiConstrArray).to(input.solver)
            handle_constrs(constrs)
            return output

        elif isinstance(input, SymbolicLightningArray):
            raise NotImplementedError

class DecoupledReLU(DecoupledNonLinearLayer):
    """ NOTE(anonymous): This is NOT a subclass of nn.ReLU """
    def forward_decoupled(self, input_val: Tuple, input_act: Tuple) -> Tuple[Any, Any]:
        output_val = input_val
        output_val[input_act < 0.] = 0.
        output_act = self.act(input_act)

        return (output_val, output_act)

class SymbolicDecoupledReLU(DecoupledReLU, NonLinearLayer): ...
