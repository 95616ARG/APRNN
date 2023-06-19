from __future__ import annotations
from typing import Iterable
import warnings
import torch.nn as nn
import torch.nn.functional as F

from sytorch.solver import *
from sytorch.solver import lightning
from ..symbolic_mode import no_symbolic

from sytorch.pervasives import *
import numpy as np
from .module import *
from .module import T

__all__ = [
    "Conv2d"
]

# NOTE(anonymous): Legacy code for the deprecated Conv2d.forward_symbolic with
# SymbolicGurobiArray.
def _pad(x, padding: Iterable[int], padding_mode='zeros'):
    leading_dims = x.ndim - len(padding)
    assert leading_dims in [1, 2]
    padding = ((0,0),)*leading_dims + tuple((p,p) for p in padding)

    if padding_mode == 'zeros':
        return np.pad(x, padding, 'constant')
    else:
        raise NotImplementedError(
            f"unsupported padding mode {padding_mode}"
        )

def __sum_bias_kernel(out, a, bias=None):
    lhs = [f"{out}"]
    rhs = 0.
    for arg in a.flat:
        if isinstance(arg, LightningVar):
            lhs.append(str(arg))
        else:
            rhs += arg

    if bias is not None:
        if isinstance(bias, LightningVar):
            lhs.append(str(bias))
        else:
            rhs += bias

    return " - ".join(lhs) + f" = {rhs}\n"

def _einsum_kernel_concrete_boxes(
        out_lb0d, out_ub0d,
        lb_approx1d, ub_approx1d,
        xlb1d, xub1d, weight1d, bias0d
    ):

    constrs = []
    for lb_approx, ub_approx, xlb, xub, w in zip(lb_approx1d.flat, ub_approx1d.flat, xlb1d.flat, xub1d.flat, weight1d.flat):
        if isinstance(w, LightningVar):
            constrs.append(f"{xlb} {w} - {lb_approx} >= 0\n")
            constrs.append(f"{xub} {w} - {lb_approx} >= 0\n")
            constrs.append(f"{xlb} {w} - {ub_approx} <= 0\n")
            constrs.append(f"{xub} {w} - {ub_approx} <= 0\n")

        # elif isinstance(w, LightningVar):
        else:
            if w >= 0.:
                clb, cub = xlb * w, xub * w
            else:
                clb, cub = xub * w, xlb * w

            constrs.append(f"{lb_approx} = {clb}\n")
            constrs.append(f"{ub_approx} = {cub}\n")

            # constrs.append(f"{lb_approx} <= {xlb * w}\n")
            # constrs.append(f"{lb_approx} <= {xub * w}\n")
            # constrs.append(f"{ub_approx} >= {xlb * w}\n")
            # constrs.append(f"{ub_approx} >= {xub * w}\n")

        # else:
        #     ...

    constrs.append(__sum_bias_kernel(out_lb0d, lb_approx1d, bias0d))
    constrs.append(__sum_bias_kernel(out_ub0d, ub_approx1d, bias0d))

    return "".join(constrs)

def conv_concrete_boxes_symbolic_weight(self, boxes, weight, bias):
    """_summary_

    Args:
        boxes (Tensor[N, I, , , 2]): _description_
        weight (SymbolicArray[O, I, J, K]): _description_
        bias (O): _description_

    Returns:
        _type_: _description_
    """

    if isinstance(weight, SymbolicLightningArray):
        solver = weight.solver
        N = boxes.shape[0]
        O, I, W, H = weight.shape

        # print(boxes.shape)

        # niwhjk
        xlb_windows = _as_strided_window_views2d(
            boxes[..., 0],
            kernel_size  = self.kernel_size,
            padding      = self.padding,
            padding_mode = self.padding_mode,
            stride       = self.stride,
            dilation     = self.dilation,
            ceil_mode    = False,
        )
        J, K = xlb_windows.shape[-2:]

        # niwhjk -> njkiwh
        xlb_windows = np.transpose(xlb_windows, (0,4,5,1,2,3))
        # njkiwh -> nojkiwh
        xlb_windows = broadcast_at(xlb_windows, 1, (O,))

        # niwhjk
        xub_windows = _as_strided_window_views2d(
            boxes[..., 1],
            kernel_size  = self.kernel_size,
            padding      = self.padding,
            padding_mode = self.padding_mode,
            stride       = self.stride,
            dilation     = self.dilation,
            ceil_mode    = False,
        )
        # niwhjk -> njkiwh
        xub_windows = np.transpose(xub_windows, (0,4,5,1,2,3))
        # njkiwh -> nojkiwh
        xub_windows = broadcast_at(xub_windows, 1, (O,))

        # oiwh -> nojkiwh
        weight = broadcast_at(weight, 0, (N,), 1, (J, K))

        # o -> nojk
        bias = broadcast_at(bias, 0, (N,), 1, (J, K))

        out_weighted_bounds = solver.reals((N, O, J, K, I, W, H, 2))
        out = solver.reals((N, O, J, K, 2))

        lightning.vectorize(
            _einsum_kernel_concrete_boxes,
            signature="(),(),(i,w,h),(i,w,h),(i,w,h),(i,w,h),(i,w,h),()->(@side_effect,@constr)"
        )(
            # nojk()
            out[...,0],
            # nojk()
            out[...,1],
            # nojk(iwh)
            out_weighted_bounds[...,0],
            # nojk(iwh)
            out_weighted_bounds[...,1],
            # nojk(iwh)
            xlb_windows,
            # nojk(iwh)
            xub_windows,
            # nojk(iwh)
            weight,
            # nojk()
            bias,
        )

        return out

    else:
        raise NotImplementedError

def _einsum_kernel_symbolic_boxes(
        out1d,
        xlb1d, xub1d, weight1d, bias0d
    ):
    """_summary_

    Args:
        out (SymbolicArray[2]): _description_
        box (SymbolicArray[I, 2]): _description_
        weight (Array[I]): _description_
        bias (Array[]): _description_

    Returns:
        Constr: _description_
    """
    lhs_lb = [f"- {out1d[0]}"]
    lhs_ub = [f"- {out1d[1]}"]
    rhs_lb = 0.
    rhs_ub = 0.
    for lb, ub, w in zip(xlb1d.flat, xub1d.flat, weight1d.flat):
        if w >= 0.:
            if isinstance(lb, LightningVar):
                lhs_lb.append(f"{w} {lb}")
            else:
                rhs_lb -= w * lb

            if isinstance(ub, LightningVar):
                lhs_ub.append(f"{w} {ub}")
            else:
                rhs_ub -= w * ub

        else:
            if isinstance(lb, LightningVar):
                lhs_ub.append(f"{w} {lb}")
            else:
                rhs_ub -= w * lb

            if isinstance(ub, LightningVar):
                lhs_lb.append(f"{w} {ub}")
            else:
                rhs_lb -= w * ub

    if bias0d is not None:
        if isinstance(bias0d, LightningVar):
            lhs_lb.append(f"{bias0d}")
            lhs_ub.append(f"{bias0d}")
        else:
            rhs_lb -= bias0d
            rhs_ub -= bias0d

    return (
        " + ".join(lhs_lb) + f" = {rhs_lb}\n" +
        " + ".join(lhs_ub) + f" = {rhs_ub}\n"
    )

def conv_symbolic_boxes_concrete_weight(self, boxes, weight, bias):
    if isinstance(boxes, SymbolicLightningArray):
        solver = boxes.solver
        N = boxes.shape[0]
        O, I, W, H = weight.shape

        # niwhjk
        xlb_windows = _as_strided_window_views2d(
            boxes[..., 0],
            kernel_size  = self.kernel_size,
            padding      = self.padding,
            padding_mode = self.padding_mode,
            stride       = self.stride,
            dilation     = self.dilation,
            ceil_mode    = False,
        )
        J, K = xlb_windows.shape[-2:]

        # niwhjk -> njkiwh
        xlb_windows = np.transpose(xlb_windows, (0,4,5,1,2,3))
        # njkiwh -> nojkiwh
        xlb_windows = broadcast_at(xlb_windows, 1, (O,))

        # niwhjk
        xub_windows = _as_strided_window_views2d(
            boxes[..., 1],
            kernel_size  = self.kernel_size,
            padding      = self.padding,
            padding_mode = self.padding_mode,
            stride       = self.stride,
            dilation     = self.dilation,
            ceil_mode    = False,
        )
        # niwhjk -> njkiwh
        xub_windows = np.transpose(xub_windows, (0,4,5,1,2,3))
        # njkiwh -> nojkiwh
        xub_windows = broadcast_at(xub_windows, 1, (O,))

        # oiwh -> nojkiwh
        weight = broadcast_at(weight, 0, (N,), 1, (J, K))

        # o -> nojk
        bias = broadcast_at(bias, 0, (N,), 1, (J, K))
        out = solver.reals((N, O, J, K, 2))

        lightning.vectorize(
            _einsum_kernel_symbolic_boxes,
            signature="(2),(i,w,h),(i,w,h),(i,w,h),()->(@side_effect,@constr)"
        )(
            # nojk()
            out,
            # nojk(iwh)
            xlb_windows,
            # nojk(iwh)
            xub_windows,
            # nojk(iwh)
            weight,
            # nojk()
            bias,
        )

        return out

    else:
        raise NotImplementedError
class Conv2d(LinearLayer, Layer2D, nn.Conv2d, ONNXCompatibleModule):

    def h(self, boxes, **kwargs):
        if isinstance(boxes, Tensor):
            return conv_concrete_boxes_symbolic_weight(
                    self, boxes.detach().cpu().numpy(), self.weight.array(), self.bias.array())
        else:
            return conv_symbolic_boxes_concrete_weight(
                    self, boxes, self.weight.array(), self.bias.array())


    def to_eran_bounds(self, boxes):
        assert boxes.ndim == 5 # NCHWB
        return torch.moveaxis(boxes, -4, -2).flatten(-4, -2)

    # def _conv_symbolic_forward(self, input: 'SymbolicArray'):
    #     warnings.warn(
    #         "Conv2d.forward_symbolic with SymbolicGurobiArray is slow hence "
    #         "deprecated. Please use SymbolicLightningArray instead."
    #     )
    #     # TODO(anonymous): handle the case when the `num_batches` axis doesn't exist.
    #     assert self.dilation == (1, 1)

    #     input = _pad(input, padding=self.padding, padding_mode=self.padding_mode)

    #     output_grid_shape = (
    #         int(np.floor(((input.shape[-2] - self.dilation[0] * (self.kernel_size[0] - 1) - 1) / self.stride[0]) + 1)),
    #         int(np.floor(((input.shape[-1] - self.dilation[1] * (self.kernel_size[1] - 1) - 1) / self.stride[1]) + 1)),
    #     )

    #     """ Create inplace sub-views of input array's underlying data with the specified strides . """
    #     assert input.data.contiguous
    #     input_strided_views = np.lib.stride_tricks.as_strided(
    #         # (num_batches, in_channels, input_width, input_height).
    #         input,

    #         # (num_batches, in_channels, kernel_width, kernel_height, output_width, output_height).
    #         shape     = input.shape[:1] + self.weight.shape[1:] + output_grid_shape,

    #         # Strides for indexing the original and output grids.
    #         strides   = input.strides + tuple(stride_length * stride_steps for stride_length, stride_steps in zip(input.strides[2:], self.stride)),

    #         # Preserve the `SymbolicArray` subtype.
    #         subok     = True,

    #         # Create read-only strided views.
    #         writeable = False
    #     )

    #     """ NOTE(anonymous): In short, when the size of any axis to be summed over
    #     (here `in_channels`, `kernel_width` or `kernel_height`) is 1,
    #     `np.einsum` fails with a "TypeError: invalid data type for einsum".
    #     Therefore the following code manually eliminates such axes.

    #     To further explain it, `np.einsum` in general doesn't support object
    #     arrays. However, the 'optimal' einsum path seach algorithm does have a
    #     good chance to find a feasible path for object arrays because many
    #     numpy ufuncs, like the matrix multiplication ufunc, do support object
    #     arrays. Unfortunately, the 'optimal' algorithm still fails if any axis
    #     to be summed over has a size 1.
    #     """
    #     iwh = ""
    #     iwh_indices = ()
    #     for dim_subscript, dim_size in [
    #         ('i', self.in_channels   ),
    #         ('w', self.kernel_size[0]),
    #         ('h', self.kernel_size[1]),
    #     ]:
    #         if dim_size > 1:
    #             iwh += dim_subscript
    #             iwh_indices += (slice(None),)
    #         else:
    #             iwh_indices += (0,)

    #     print(input_strided_views[tuple((slice(None), *iwh_indices, ...))].shape)
    #     print(self.weight.array()[tuple((slice(None), *iwh_indices, ...))].shape)

    #     if self.bias is not None:
    #         return np.einsum(
    #             # (num_batches[, in_channels, kernel_width, kernel_height], output_width, output_height),
    #             # (out_channels[, in_channels, kernel_width, kernel_height])
    #             # -> (num_batches, out_channels, output_width, output_height)
    #             f"n{iwh}jk,o{iwh}->nojk",
    #                 input_strided_views[tuple((slice(None), *iwh_indices, ...))],
    #                 self.weight.array()[tuple((slice(None), *iwh_indices, ...))],
    #             optimize='optimal'
    #         ) + self.bias.array()[None, ..., None, None] # Broadcast with bias at the `out_channels` axis.

    #     else:
    #         return np.einsum(
    #             # (num_batches[, in_channels, kernel_width, kernel_height], output_width, output_height),
    #             # (out_channels[, in_channels, kernel_width, kernel_height])
    #             # -> (num_batches, out_channels, output_width, output_height)
    #             f"n{iwh}jk,o{iwh}->nojk",
    #                 input_strided_views[tuple((slice(None), *iwh_indices, ...))],
    #                 self.weight.array()[tuple((slice(None), *iwh_indices, ...))],
    #             optimize='optimal'
    #         ) # Broadcast with bias at the `out_channels` axis.

    def forward_symbolic(self:T, input: 'Tensor' | 'SymbolicArray', pattern=None):

        if self.bias is not None:
            weight = self.weight.array_if_symbolic()
            bias = self.bias.array_if_symbolic()
            configuration = tuple(map(type, (input, weight, bias)))

        else:
            weight = self.weight.array_if_symbolic()
            bias = None
            configuration = tuple(map(type, (input, weight)))

        if all(issubclass(ty, Tensor) for ty in configuration):
            with no_symbolic():
                output = self(input)

        elif any(issubclass(ty, SymbolicGurobiArray) for ty in configuration):

            if self.row_mask is not None:
                sym_output = conv2d_symbolic(
                    input,
                    weight = weight[self.row_mask,...],
                    bias = bias[self.row_mask,...] if bias is not None else None,
                    padding      = self.padding,
                    padding_mode = self.padding_mode,
                    stride       = self.stride,
                    dilation     = self.dilation,
                    ceil_mode    = False,
                    value        = 'zero'
                )
                solver = sym_output.solver
                array_type = type(sym_output)
                output = np.empty((*input.shape[:-3], weight.shape[0], *sym_output.shape[-2:]), dtype=object)
                output[...,self.row_mask,:,:] = sym_output

                with no_symbolic():
                    output[..., ~self.row_mask,:,:] = self._conv_forward(
                        input,
                        self.weight[~self.row_mask,...],
                        self.bias[~self.row_mask,...] if bias is not None else None,
                    ).cpu().detach().numpy()

                output = output.view(array_type).to(solver)
                output.row_mask = self.row_mask
                output.mask = broadcast_at(self.row_mask, output.ndim-1, output.shape[-2:], 0, output.shape[:-3])
                output._concrete_dtype = torch_dtype_to_numpy(input.dtype)
                assert output.shape == output.mask.shape

            else:
                output = conv2d_symbolic(
                    input,
                    weight = weight,
                    bias = bias,
                    padding      = self.padding,
                    padding_mode = self.padding_mode,
                    stride       = self.stride,
                    dilation     = self.dilation,
                    ceil_mode    = False,
                    groups       = self.groups,
                    value        = 'zero'
                )

        elif any(issubclass(ty, SymbolicLightningArray) for ty in configuration):
            if self.row_mask is not None and isinstance(input, Tensor):
                # print("conv row mask")

                if bias is not None:
                    sym_output = lightning.conv2d(
                        input, weight[self.row_mask,...], bias[self.row_mask,...],
                        kernel_size  = self.kernel_size,
                        padding      = self.padding,
                        padding_mode = self.padding_mode,
                        stride       = self.stride,
                        dilation     = self.dilation,
                        ceil_mode    = False,
                        groups       = self.groups,
                        executor     = self.executor,
                    )
                    solver = sym_output.solver
                    array_type = type(sym_output)
                    output = np.empty((*input.shape[:-3], weight.shape[0], *sym_output.shape[-2:]), dtype=object)
                    output[...,self.row_mask,:,:] = sym_output

                    with no_symbolic():
                        output[..., ~self.row_mask,:,:] = self._conv_forward(
                            input, self.weight[~self.row_mask,...], self.bias[~self.row_mask,...]
                        ).cpu().detach().numpy()

                else:
                    sym_output = lightning.conv2d(
                        input, weight[self.row_mask,...], bias=None,
                        kernel_size  = self.kernel_size,
                        padding      = self.padding,
                        padding_mode = self.padding_mode,
                        stride       = self.stride,
                        dilation     = self.dilation,
                        ceil_mode    = False,
                        groups       = self.groups,
                        executor     = self.executor,
                    )
                    solver = sym_output.solver
                    array_type = type(sym_output)
                    output = np.empty((*input.shape[:-3], weight.shape[0], *sym_output.shape[-2:]), dtype=object)
                    output[...,self.row_mask,:,:] = sym_output

                    with no_symbolic():
                        output[..., ~self.row_mask,:,:] = self._conv_forward(
                            input, self.weight[~self.row_mask,...], bias=None
                        ).cpu().detach().numpy()

                output = output.view(array_type).to(solver)
                output.row_mask = self.row_mask
                output.mask = broadcast_at(self.row_mask, output.ndim-1, output.shape[-2:], 0, output.shape[:-3])
                output._concrete_dtype = torch_dtype_to_numpy(input.dtype)
                assert output.shape == output.mask.shape

            else:
                output = lightning.conv2d(
                    input, weight, bias,
                    kernel_size  = self.kernel_size,
                    padding      = self.padding,
                    padding_mode = self.padding_mode,
                    stride       = self.stride,
                    dilation     = self.dilation,
                    ceil_mode    = False,
                    groups       = self.groups,
                    executor     = self.executor,
                )

        else:
            raise NotImplementedError(
                f"unimplemented {type(self)} symbolic forward for {configuration}"
            )

        return output

    # def v(self, vpolytopes, pattern=None):

    #     if self.bias is not None:
    #         weight = self.weight.array_if_symbolic()
    #         bias = self.bias.array_if_symbolic()
    #         configuration = tuple(map(type, (vpolytopes, weight, bias)))

    #     else:
    #         weight = self.weight.array_if_symbolic()
    #         bias = None
    #         configuration = tuple(map(type, (vpolytopes, weight)))

    #     if all(issubclass(ty, Tensor) for ty in configuration):
    #         with no_symbolic():
    #             N, V, *S = vpolytopes.shape
    #             out_vpolytopes = self(vpolytopes.reshape(N * V, *S))
    #             out_vpolytopes = out_vpolytopes.reshape(N, V, *out_vpolytopes.shape[1:])
    #             return out_vpolytopes

    #     elif any(issubclass(ty, SymbolicLightningArray) for ty in configuration):
    #         N, V, *S = vpolytopes.shape
    #         # vpolytopes = vpolytopes.reshape(N * V, -1)
    #         out_vpolytopes = self.forward_symbolic(
    #             vpolytopes.reshape(N * V, *S),
    #             pattern = pattern
    #         )
    #         out_vpolytopes = out_vpolytopes.reshape(N, V, *out_vpolytopes.shape[1:])
    #         return out_vpolytopes

    #     else:
    #         raise NotImplementedError(f"{configuration}")

# from sytorch.nn.modules.linear import linear
from . import functional

def conv2d_symbolic(
    inp: Tensor | SymbolicGurobiArray,
    weight: Tensor | SymbolicGurobiArray,
    bias: Tensor | SymbolicGurobiArray | None,
    padding      = (0,0),
    padding_mode = 'constant',
    stride       = (1,1),
    dilation     = (1,1),
    ceil_mode    = False,
    groups       = 1,
    value        = 0,
) -> SymbolicGurobiArray:

    assert groups == 1
    O, I, W, H = weight.shape
    inp_views = _as_strided_window_views2d(
        inp,
        kernel_size = (W, H),
        padding = padding,
        padding_mode = padding_mode,
        stride = stride,
        dilation = dilation,
        ceil_mode = ceil_mode,
        value = value
    )

    N, I, W, H, J, K = inp_views.shape
    inp_views = inp_views.permute((0, 4, 5, 1, 2, 3))\
                         .reshape((N*J*K, I*W*H))
    weight = weight.reshape((O, I*W*H))

    # N, I, W, H, J, K,
    # O, I, W, H
    # O
    # ->
    # N, O, J, K

    # linear:
    # (N * J * K) * (I * W * H)
    # O * (I * W * H)
    # (N * J * K) * O

    # (N*J*K, O)
    # print(inp_views.shape, weight.shape)
    out = functional.linear(inp_views, weight, bias)\
            .reshape(N, J, K, O)\
            .permute((0, 3, 1, 2))

    return out
