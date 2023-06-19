from __future__ import annotations

import torch.nn as nn
import torch.nn.functional as F
from sytorch.solver import SymbolicGurobiArray, SymbolicLightningArray

from .module import *
from .module import T

from sytorch.pervasives import *
from sytorch.solver import lightning
from ...solver import *
from ..symbolic_mode import *
from .functional import *

__all__ = [
    'NormalizeInput',
    "Identity",
    "Linear",
]

class Identity(LinearLayer, nn.Identity):
    def forward_symbolic(self, input: 'Tensor' | 'SymbolicArray', pattern=None):
        return input

class NormalizeInput(LinearLayer):
    def __init__(self, mean, std):
        super().__init__()
        self.mean = nn.Parameter(mean)
        self.std = nn.Parameter(std)

    def h(self, boxes, **kwargs):
        out = boxes.clone()
        out[...,0] = self.forward(out[...,0])
        out[...,1] = self.forward(out[...,1])
        return out

    def forward(self, input):
        return (input - self.mean) / self.std

    def forward_symbolic(self, input: 'Tensor' | 'SymbolicArray', pattern=None):
        if isinstance(input, Tensor):
            return (input - self.mean) / self.std
        else:
            raise NotImplementedError

    def v(self, vpolytopes, pattern=None):
        return self(vpolytopes)

class FusedLinearArgMax: ...

# def linear_boxes_sym(boxes, weight, bias):
#     # multiply_interval_with_variable(boxes, weight)
#     N, I, _ = boxes.shape
#     O, I = weight.shape

#     out_boxes = np.empty((N, O, 2), dtype=object).view(type(weight)).to(weight.solver)

#     for out_box, box in zip(out_boxes, boxes): # over N
#         for out_box_row, weight_row, bias_row in zip(out_box, weight, bias): # over O
#             out_box_row[...] = multiply_interval_with_variable(box.numpy(), weight_row).sum(0) + bias_row

#     return out_boxes

def fused_linear_argmax_sorted(boxes, batched_weight, batched_bias):
    """first impl which sorts params."""
    warnings.warn("outdated.")
    boxes = boxes.detach().cpu().numpy()
    N, I, _ = boxes.shape
    _, O, I = batched_weight.shape

    # out = np.moveaxis((np.moveaxis(boxes, -1, 0) @ weight.T), 0, -1) # (N, O, 2)
    out_boxes = np.empty((N, O, 2), dtype=object).view(type(batched_weight)).to(batched_weight.solver)

    batched_flip_mask = batched_weight.semi_positivity() == -1

    for out_box, box, weight, flip_mask, bias in zip(out_boxes, boxes, batched_weight, batched_flip_mask, batched_bias): # over N
        for out_box_row, weight_row, bias_row, flip_mask_row in zip(out_box, weight, bias, flip_mask): # over O
            weighted_bounds = box * weight_row[:, None]
            weighted_bounds[flip_mask_row] = weighted_bounds[flip_mask_row,::-1]
            out_box_row[...] = weighted_bounds.sum(0) + bias_row

    return out_boxes


def fused_linear_argmax_exact(boxes, batched_weight, batched_bias):
    warnings.warn("outdated.")
    boxes = boxes.detach().cpu().numpy()
    N, I, _ = boxes.shape
    _, O, I = batched_weight.shape

    # out = np.moveaxis((np.moveaxis(boxes, -1, 0) @ weight.T), 0, -1) # (N, O, 2)
    out_boxes = np.empty((N, O, 2), dtype=object).view(type(batched_weight)).to(batched_weight.solver)

    for out_box, box, weight, bias in zip(out_boxes, boxes, batched_weight, batched_bias): # over N
        for out_box_row, weight_row, bias_row in zip(out_box, weight, bias): # over O
            out_box_row[...] = multiply_interval_with_variable_overapprox(box, weight_row).sum(0) + bias_row

    return out_boxes

def fused_linear_argmax(boxes, batched_weight, batched_bias):
    # boxes = boxes.detach().cpu().numpy()
    N, I, _ = boxes.shape
    N, O, I = batched_weight.shape

    # # out = np.moveaxis((np.moveaxis(boxes, -1, 0) @ weight.T), 0, -1) # (N, O, 2)
    # # Only LB
    # out_boxes = np.empty((N, O, 2), dtype=object).view(type(batched_weight)).to(batched_weight.solver)

    # for out_box, box, weight, bias in zip(out_boxes, boxes, batched_weight, batched_bias): # over N
    #     for out_box_row, weight_row, bias_row in zip(out_box, weight, bias): # over O
    #         out_box_row[...] = multiply_interval_with_variable_linearize(box, weight_row).sum(0) + bias_row

    # return out_boxes
    if isinstance(boxes, SymbolicGurobiArray):
        raise NotImplementedError

    elif isinstance(boxes, SymbolicLightningArray):
        out_boxes = lightning.vectorize(
            _symbolic_box_concrete_weight_kernel,
            signature="(b,@implicit,@real,@ret),(i,b),(i),()->(@side_effect,@constr)"
        )(
            broadcast_at(boxes, 1, (O,)) , # (N, [O,] I, 2)
            batched_weight, # (N, O, I)
            batched_bias, # (N, O)
        )
        return out_boxes

    elif isinstance(boxes, Tensor):
        boxes = boxes.detach().cpu().numpy()
        def _k(weighted_box2d, box2d, weight1d):
            """_summary_

            Args:
                weighted_box2d (SymbolicArray[I, 2]): _description_
                box2d (Array[I, 2]): _description_
                weight1d (SymbolicArray[I]): _description_

            Returns:
                Constrs: _description_
            """
            constrs = []
            for (lb_approx, ub_approx), (xlb, xub), w in zip(weighted_box2d, box2d, weight1d):
                # lb_approx <= xlb * w
                # lb_approx <= xub * w
                # ub_approx >= xlb * w
                # ub_approx >= xub * w
                if isinstance(w, LightningVar):
                    constrs.append(f"{xlb} {w} - {lb_approx} >= 0\n")
                    constrs.append(f"{xub} {w} - {lb_approx} >= 0\n")
                    constrs.append(f"{xlb} {w} - {ub_approx} <= 0\n")
                    constrs.append(f"{xub} {w} - {ub_approx} <= 0\n")

                else:
                    constrs.append(f"{lb_approx} <= {xlb * w}\n")
                    constrs.append(f"{lb_approx} <= {xub * w}\n")
                    constrs.append(f"{ub_approx} >= {xlb * w}\n")
                    constrs.append(f"{ub_approx} >= {xub * w}\n")

            return "".join(constrs)

        solver = batched_weight.solver
        out_weighted_bounds = solver.reals((N, O, I, 2))
        with _timeit2("mul batched weight"):
            lightning.vectorize(
                _k,
                signature="(i,b),(i,b),(i)->(@side_effect,@constr)"
            )(
                out_weighted_bounds,
                # (N, O, I, 2)
                broadcast_at( boxes, 1, (O,)),
                # (N, O, I, 2)
                # broadcast_at(weight, 0, (N,)),
                batched_weight,
                # (N, O, I)
            )

        with _timeit2("add batched bias"):
            out_boxes = np.concatenate((
                out_weighted_bounds,
                broadcast_at(batched_bias, 2, (1, 2))
            ), axis=-2).view(type(batched_weight)).to(batched_weight.solver)\
                .sum(axis=-2)

        return out_boxes

    else:
        raise NotImplementedError


def linear_concrete_boxes_symbolic_weight(boxes, weight, bias):
    """_summary_

    Args:
        boxes (Tensor[N, I, 2]): _description_
        weight (SymbolicArray[O, I]): _description_
        bias (SymbolicArray[I]): _description_

    Returns:
        SymbolicArray[..., O, 2]: _description_
    """

    N, I, _ = boxes.shape
    O, I = weight.shape

    boxes = boxes.detach().cpu().numpy()
    # (N, I, 2)

    if isinstance(weight, SymbolicGurobiArray):
        raise NotImplementedError
        # warnings.warn("Encoding with Gurobi objects might be slow.")
        # out_boxes = np.empty((N, O, 2), dtype=object).view(type(weight)).to(weight.solver)
        # # (N, O, 2)

        # from tqdm import tqdm
        # for out_box, box in tqdm(zip(out_boxes, boxes), total=N, leave=False): # over N
        #     # (I, 2), (O, 2)
        #     for out_box_row, weight_row, bias_row in tqdm(zip(out_box, weight, bias), total=O, leave=False): # over O
        #         #   (2,)        (I,)       (1,)
        #         out_box_row[...] = multiply_interval_with_variable_linearize(box, weight_row).sum(0) + bias_row

    elif isinstance(weight, SymbolicLightningArray):
        def _k(weighted_box2d, box2d, weight1d):
            """_summary_

            Args:
                weighted_box2d (SymbolicArray[I, 2]): _description_
                box2d (Array[I, 2]): _description_
                weight1d (SymbolicArray[I]): _description_

            Returns:
                Constrs: _description_
            """
            constrs = []
            for (lb_approx, ub_approx), (xlb, xub), w in zip(weighted_box2d, box2d, weight1d):
                # lb_approx <= xlb * w
                # lb_approx <= xub * w
                # ub_approx >= xlb * w
                # ub_approx >= xub * w
                if isinstance(w, LightningVar):
                    constrs.append(f"{xlb} {w} - {lb_approx} >= 0\n")
                    constrs.append(f"{xub} {w} - {lb_approx} >= 0\n")
                    constrs.append(f"{xlb} {w} - {ub_approx} <= 0\n")
                    constrs.append(f"{xub} {w} - {ub_approx} <= 0\n")

                else:
                    if w >= 0.:
                        clb, cub = xlb * w, xub * w
                    else:
                        clb, cub = xub * w, xlb * w

                    constrs.append(f"{lb_approx} = {clb}\n")
                    constrs.append(f"{ub_approx} = {cub}\n")

                # else:
                #     constrs.append(f"{lb_approx} <= {xlb * w}\n")
                #     constrs.append(f"{lb_approx} <= {xub * w}\n")
                #     constrs.append(f"{ub_approx} >= {xlb * w}\n")
                #     constrs.append(f"{ub_approx} >= {xub * w}\n")

            return "".join(constrs)

        with _timeit2('multiply conrete boxes with symbolic weight'):
            solver = weight.solver
            out_weighted_bounds = solver.reals((N, O, I, 2))
            lightning.vectorize(
                _k,
                signature="(i,b),(i,b),(i)->(@side_effect,@constr)"
            )(
                out_weighted_bounds,
                # (N, O, I, 2)
                broadcast_at( boxes, 1, (O,)),
                # (N, O, I, 2)
                broadcast_at(weight, 0, (N,)),
                # (N, O, I)
            )

        with _timeit2('adding symbolic bias'):
            out_boxes = np.concatenate((
                out_weighted_bounds,
                broadcast_at(bias, 0, (N,), 1, (1, 2))
            ), axis=-2).view(type(weight)).to(weight.solver)\
                .sum(axis=-2)

    return out_boxes

def linear_concrete_boxes_symbolic_weight_bounds(boxes, weight, bias):
    """_summary_

    Args:
        symbolic_boxes (Tensor[N, I, 2]): _description_
        weight (SymbolicArray[O, I]): _description_
        bias (SymbolicArray[I]): _description_

    Returns:
        Array[..., O, 2]: LB bound
        Array[..., O, 2]: UB bound
        Array[..., O, 2]: H bound
    """

    N, I, _ = boxes.shape
    O, I = weight.shape
    broadcast_shape = (N, O)

    boxes = boxes.detach().cpu().numpy()
    weight_bounds = weight.bounds
    bias_bounds = bias.bounds

    out_LB_bounds_flat = np.zeros((N * O, 2), dtype=boxes.dtype) # [no]2
    out_UB_bounds_flat = np.zeros((N * O, 2), dtype=boxes.dtype) # [no]2
    # out_H_bounds  = np.zeros((N, O, 2), dtype=boxes.dtype)

    boxes_flat, weight_bounds_flat, bias_bounds_flat = flatten(
        broadcast_at(boxes, 1, (O,)), # [n(o)]i2,
        broadcast_at(weight_bounds, 0, (N,)), # [(n)o]i2
        broadcast_at(bias_bounds, 0, (N,)), # [(n)o]2
        start_dim = 0,
        end_dim = len(broadcast_shape)-1
    )

    total = np.prod(broadcast_shape)
    num_chunks = get_max_workers() * get_dispatch_multiplier()
    chunksize = (total // num_chunks) + 1

    def _kernel(idx):
        lbs_bound, ubs_bound, _ = interval_mul_interval_bound_of_bound(boxes_flat[idx], weight_bounds_flat[idx])
        return (lbs_bound.sum(0) + bias_bounds_flat[idx], ubs_bound.sum(0) + bias_bounds_flat[idx])

    def _kernel_closure(start):
        end = min(start+chunksize, total)
        return tuple(_kernel(idx) for idx in range(start, end))

    with GlobalRegister(globals(), _kernel, _kernel_closure):
        executor = ProcessPoolExecutor(get_max_workers())
        constrs_future = executor.map(_kernel_closure, range(0, total, chunksize),)

        # return constrs_future
        for idx, (lb_bounds, ub_bounds) in tqdm(enumerate(itertools.chain(itertools.chain(*constrs_future)))):
            out_LB_bounds_flat[idx,:] = lb_bounds
            out_UB_bounds_flat[idx,:] = ub_bounds

    return out_LB_bounds_flat.reshape(N, O, 2), out_UB_bounds_flat.reshape(N, O, 2)

    # from tqdm import tqdm
    # for out_lb_bound, out_ub_bound, box in tqdm(zip(out_LB_bounds, out_UB_bounds, boxes), total=N, leave=False): # over N
    #     #  (O, 2)        (O, 2)    (I, 2)             (N, O, 2)       (N, O, 2)  (N, I, 2)
    #     for out_lb_bound_row, out_ub_bound_row, weight_bound_row, bias_bound_row in tqdm(zip(out_lb_bound, out_ub_bound, weight_bounds, bias_bounds), total=O, leave=False): # over O
    #         #     (2)               (2)            (I, 2)              (2,)                    (O, 2)         (O, 2)       (O, I, 2)       (O, 2)
    #         lbs_bound, ubs_bound, hs_bound = interval_mul_interval_bound_of_bound(box, weight_bound_row)
    #         #  (I, 2)    (I, 2)    (I, 2)
    #         out_lb_bound_row[...] = lbs_bound.sum(0) + bias_bound_row
    #         out_ub_bound_row[...] = ubs_bound.sum(0) + bias_bound_row
    #         # out_h_bound_row [...] =  hs_bound.sum(0) + bias_bound_row
    # warnings.warn("the calculation of H may not be correct.")
    # return out_LB_bounds, out_UB_bounds

def fused_linear_relu(boxes, weight, bias):
    """_summary_

    Args:
        boxes (Array[..., I, 2]): _description_
        weight (SymbolicArray[O, I]): _description_
        bias (SymbolicArray[I]): _description_
    """

    """ Handle Linear Layer.
    - We calculate the symbolic lower- and upper-bound of outputs
    - We calculate the concrete bounds for the symbolic lower-, upper-bounds and
      the size of (sub - slb).
    """
    if isinstance(boxes, Tensor):
        out_symbolic_boxes = linear_concrete_boxes_symbolic_weight(boxes, weight, bias)
        # (N, O, 2)
    else:
        out_symbolic_boxes = linear_symbolic_boxes(boxes, weight, bias)

    """ Handle ReLU Layer.
    - We calculate symbolic output upper-bound from symbolic input upper-bound.
        - specifically, we linearize the upper-bound of the symbolic
          upper-bound. (defined by yub >= xub and yub >= 0.) The programming
          space is a convex polyhedra.
    - We calculate symbolic output lower-bound from symbolic input lower-bound.
        - It can not be simply linearized because the programming space is
          defined by ylb <= xlb OR ylb <= 0., which is a non-convex polyhedra.
        - When xlb >= 0. (xlblb >= 0.), we have ylb = xlb
        - When xlb <= 0. (xlbub <= 0.), we have ylb = 0.
        - Otherwise (xlblb <= 0. and xubub >= 0.) we either fallback to pick one
          over-approximation, or encode an exact MILP.
          - If |xlblb| >= |xlbub|, ylb = 0.
          - If |xlblb| <  |xlbub|, ylb = xlb
          - Q: Can we make use of the bound of `xub - xlb`?
    - Optimize the bounds of weights to optimize the imprecision?
      - To do this, we can just assert the picked under-approximation, just like assert activation patterns or signs.
      - Or we can optimize it. But it may not worth it.
    """
    LB, UB = as_slice[...,0], as_slice[...,1]
    solver = out_symbolic_boxes.solver
    relu_out_symbolic_boxes = out_symbolic_boxes.copy()

    # Symbolic output upper-bound.
    relu_out_symbolic_boxes[UB] = solver.reals(out_symbolic_boxes[...,1].shape)
    solver.add_constraints(
        relu_out_symbolic_boxes[UB] >= out_symbolic_boxes[UB],
        relu_out_symbolic_boxes[UB] >= 0.,
    )

    # Symbolic output lower-bound.
    # relu_out_symbolic_boxes[LB] = out_symbolic_boxes[LB]
    relu_out_symbolic_boxes[LB] = 0.

    """_summary_

    out_LB_bounds, out_UB_bounds = linear_concrete_boxes_symbolic_weight_bounds(boxes, weight, bias)
    # (N, O, 2)     (N, O, 2)      (N, O, 2)

    ## xlblb >= 0., definitely non-negative, ylb = xlb
    non_negative_lb_mask = out_LB_bounds[LB] >= 0.
    print(f"{non_negative_lb_mask.sum()} non-negative lbs.")
    relu_out_symbolic_boxes[non_negative_lb_mask, 0] = out_symbolic_boxes[non_negative_lb_mask, 0]

    ## xlbub <= 0., definitely non-negative
    non_positive_lb_mask = out_UB_bounds[UB] <= 0.
    print(f"{non_positive_lb_mask.sum()} non-positive lbs.")
    relu_out_symbolic_boxes[non_positive_lb_mask, 0] = 0.

    ##
    undetermined_mask = ~ (non_negative_lb_mask + non_positive_lb_mask)
    print(f"{undetermined_mask.sum()} over-approx lbs.")
    for idx in zip(*np.where(undetermined_mask == True)):
        xlblb, xlbub = out_LB_bounds[idx]
        idx_lb = (*idx, 0)
        if -xlblb >= xlbub:
            relu_out_symbolic_boxes[idx_lb] = 0.
        else:
            relu_out_symbolic_boxes[idx_lb] = out_symbolic_boxes[idx_lb]
    """

    warnings.warn(
        "concrete bound of symbolic bounds after ReLU is not tracked yet. In "
        "the end we might want to just tight the bounds an carry them with "
        "variables."
    )
    return relu_out_symbolic_boxes #, out_LB_bounds, out_UB_bounds

def _symbolic_box_concrete_weight_kernel(out, box, weight, bias=None):
    """_summary_

    Args:
        out (SymbolicArray[2]): _description_
        box (SymbolicArray[I, 2]): _description_
        weight (Array[I]): _description_
        bias (Array[]): _description_

    Returns:
        Constr: _description_
    """
    lhs_lb = [f"- {out[0]}"]
    lhs_ub = [f"- {out[1]}"]
    rhs_lb = 0.
    rhs_ub = 0.
    for (lb, ub), w in zip(box, weight):
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

    if bias is not None:
        if isinstance(bias, LightningVar):
            lhs_lb.append(f"{bias}")
            lhs_ub.append(f"{bias}")
        else:
            rhs_lb -= bias
            rhs_ub -= bias

    return (
        " + ".join(lhs_lb) + f" = {rhs_lb}\n" +
        " + ".join(lhs_ub) + f" = {rhs_ub}\n"
    )

def linear_symbolic_boxes(symbolic_boxes, weight, bias):
    """_summary_

    Args:
        symbolic_boxes (SymbolicArray[N, I, 2]): _description_
        weight (Array[O, I]): _description_
        bias (SymbolicArray[I]): _description_

    Returns:
        SymbolicArray[..., O, 2]: _description_
    """

    N, I, _ = symbolic_boxes.shape
    O, I = weight.shape

    if isinstance(symbolic_boxes, SymbolicGurobiArray):
        raise NotImplementedError
        out_boxes = np.empty((N, O, 2), dtype=object).view(type(symbolic_boxes)).to(symbolic_boxes.solver)

        for out_box, sym_box in zip(out_boxes, symbolic_boxes): # over N
            for out_box_row, weight_row, bias_row in zip(out_box, weight, bias): # over O
                out_box_row[...] = multiply_symbolic_interval_with_concrete(sym_box, weight_row).sum(0) + bias_row

    elif isinstance(symbolic_boxes, SymbolicLightningArray):

        with _timeit2('linear symbolic boxes'):
            # (N, O, 2)
            out_boxes = lightning.vectorize(
                _symbolic_box_concrete_weight_kernel,
                signature="(b,@implicit,@real,@ret),(i,b),(i),()->(@side_effect,@constr)"
            )(
                broadcast_at(symbolic_boxes, 1, (O,)), # (N, O, I, 2)
                broadcast_at(weight, 0, (N,)), # (N, O, I)
                broadcast_at(bias, 0, (N,)), # (N, O)
            )

    else:
        raise NotImplementedError


    # warnings.warn(
    #     "concrete bound of symbolic bounds after ReLU is not tracked yet. In "
    #     "the end we might want to just tight the bounds an carry them with "
    #     "variables."
    # )
    return out_boxes

class EqualityEncoder:
    def __init__(self, encoder):
        self.encoder = encoder

    def __eq__(self, other):
        return self.encoder(other)

class Linear(LinearLayer, nn.Linear, ONNXCompatibleModule):

    def fuse(self, target):
        if target == 'argmax':
            ...
        elif target == 'relu':
            ...

    def assert_order(self):
        self.weight.assert_order()
        self.bias.assert_order() # DOESN'T MATTER

    def fused_linear_argmax(self, boxes, indices=None, lb=None, ub=None,):
        warnings.warn("optimize: here we only care about the lower bound.")
        if indices is not None:
            warnings.warn("only setting argmax fusion bound for weight but not bias.")
            with _timeit2("fusing weight"):
                fused_weight = self.weight.fuse_for_argmax(indices, lb=lb, ub=ub)
            with _timeit2("fusing bias"):
                fused_bias = self.bias.fuse_for_argmax(indices)
            return fused_linear_argmax(boxes, fused_weight, fused_bias)
        else:
            return EqualityEncoder(
                lambda indices: self.fused_linear_argmax(boxes, indices)
            )

    def fused_linear_relu(self, boxes):
        return fused_linear_relu(boxes, self.weight.array(), self.bias.array())

    def fuse_(self, mode):
        self._fused = mode

    @property
    def fused(self):
        return getattr(self, '_fused', None)

    def h(self, boxes, **kwargs):
        """ Cases:
        - Concrete boxes, concrete parameters -> DeepPoly concrete boxes
        - Concrete boxes, symbolic parameters -> symbolic boxes and concrete bounds
        - Symbolic boxes, concrete parameters -> symbolic boxes and concrete bounds
        - Symbolic boxes, symbolic parameters -> symbolic boxes and concrete bounds

        Args:
            boxes (_type_): _description_
        """

        # if isinstance(boxes, Tensor):
        #     return linear_concrete_boxes_symbolic_weight

        if self.fused is None:

            if isinstance(boxes, SymbolicLightningArray):
                return linear_symbolic_boxes(boxes, self.weight.array(), self.bias.array())

            elif isinstance(boxes, Tensor):
                assert boxes.ndim == 3
                N, I, _ = boxes.shape
                O, I = self.weight.shape
                points_mask = (boxes[...,0] == boxes[...,1]).all(-1)
                if points_mask.sum() > 0:
                    boxes_mask = ~points_mask
                    print(f"linear: {boxes_mask.sum()} boxes, {points_mask.sum()} points.")
                    out_boxes = linear_concrete_boxes_symbolic_weight(boxes[boxes_mask], self.weight.array(), self.bias.array())
                    out_points = self.forward_symbolic(boxes[points_mask][...,0])
                    out = np.empty((N, O, 2), dtype=object).view(type(out_boxes)).to(out_boxes.solver)
                    points_mask = points_mask.cpu()
                    boxes_mask = boxes_mask.cpu()
                    out[boxes_mask] = out_boxes
                    out[points_mask,...,0] = out_points
                    out[points_mask,...,1] = out_points
                    return out
                else:
                    return linear_concrete_boxes_symbolic_weight(boxes, self.weight.array(), self.bias.array())

            else:
                raise NotImplementedError

        elif self.fused == 'argmax':
            return self.fused_linear_argmax(boxes)

        else:
            raise NotImplementedError

    # def h(self, boxes):
    #     if not self.has_symbolic_parameter:
    #         raise NotImplementedError

    #     assert self.symbolic_mode == True
    #     weight = self.weight.symbolic()
    #     bias = self.bias.symbolic()
    #     solver = weight.solver
    #     # for all input polytope, with the same weight
    #     x1 = solver.reals(boxes.shape[:-1])
    #     x2 = solver.reals(boxes.shape[:-1])
    #     solver.add_constraints(boxes[...,0] <= x1, x1 <= boxes[...,1])
    #     solver.add_constraints(boxes[...,0] <= x2, x2 <= boxes[...,1])
    #     y1 = self(x1)
    #     y2 = self(x2)
    #     hy = y1 - y2
    #     out_lbs = []
    #     out_ubs = []
    #     out_hs = []
    #     for o in range(self.weight.shape[0]):
    #         assert solver.solve(minimize=y1[...,o].sum().alias())
    #         out_lbs.append(y1[..., o].evaluate())
    #         assert solver.solve(maximize=y1[...,o].sum().alias())
    #         out_ubs.append(y1[..., o].evaluate())
    #         assert solver.solve(maximize=hy[...,o].sum().alias())
    #         out_hs.append(hy[...,o].evaluate())

    #     out_boxes = torch.stack((
    #         torch.stack(out_lbs, -1),
    #         torch.stack(out_ubs, -1),
    #         torch.stack(out_hs, -1),
    #     ), -1)
    #     return out_boxes

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
    #             N, V = vpolytopes.shape[:2]
    #             out_vpolytopes = self(vpolytopes.reshape(N * V, -1))
    #             out_vpolytopes = out_vpolytopes.reshape(N, V, *out_vpolytopes.shape[1:])
    #             return out_vpolytopes

    #     elif any(issubclass(ty, SymbolicLightningArray) for ty in configuration):
    #         N, V = vpolytopes.shape[:2]
    #         # vpolytopes = vpolytopes.reshape(N * V, -1)
    #         out_vpolytopes = self.forward_symbolic(
    #             vpolytopes.reshape(N * V, -1),
    #             pattern = pattern
    #         )
    #         out_vpolytopes = out_vpolytopes.reshape(N, V, *out_vpolytopes.shape[1:])
    #         return out_vpolytopes

    #     else:
    #         raise NotImplementedError

    def forward_symbolic(self:T, input: Tensor | SymbolicArray, pattern=None):

        if self.bias is not None:
            weight = self.weight.array_if_symbolic()
            bias = self.bias.array_if_symbolic()
            configuration = tuple(map(type, (input, weight, bias)))

        else:
            weight = self.weight.array_if_symbolic()
            bias = None
            configuration = tuple(map(type, (input, weight)))

        if all(issubclass(ty, Tensor) for ty in configuration):
            """ Concrete execution. """
            with no_symbolic():
                output = self(input)

        elif any(issubclass(ty, SymbolicGurobiArray) for ty in configuration):
            """ Symbolic gurobi execution. """
            return linear(input, weight, bias)
            # if bias is not None:
            #     output = (input @ weight.T + bias[None, :]).alias()
            # else:
            #     output = (input @ weight.T).alias()

        elif any(issubclass(ty, SymbolicLightningArray) for ty in configuration):
            """ Symbolic lightning execution. """
            if self.row_mask is not None and isinstance(input, Tensor):
                # print("linear row mask")
                output = np.empty((*input.shape[:-1], weight.shape[0]), dtype=object)
                if bias is not None:
                    sym_output = lightning.linear(input, weight[self.row_mask,:], bias[self.row_mask])
                    solver = sym_output.solver
                    array_type = type(sym_output)
                    output[..., self.row_mask] = sym_output
                    with no_symbolic():
                        output[..., ~self.row_mask] = F.linear(
                            input, self.weight[~self.row_mask,:], self.bias[~self.row_mask]
                        ).cpu().detach().numpy()

                else:
                    sym_output = lightning.linear(input, weight[self.row_mask,:], bias=None)
                    solver = sym_output.solver
                    array_type = type(sym_output)
                    output[..., self.row_mask] = sym_output
                    with no_symbolic():
                        output[..., ~self.row_mask] = F.linear(
                            input, self.weight[~self.row_mask,:], bias=None
                        ).cpu().detach().numpy()

                output = output.view(array_type).to(solver)
                output.row_mask = self.row_mask
                output.mask = broadcast_at(self.row_mask, 0, output.shape[:-1])
                output._concrete_dtype = torch_dtype_to_numpy(input.dtype)
                assert output.shape == output.mask.shape

            else:
                output = lightning.linear(input, weight, bias)

        else:
            raise NotImplementedError(
                f"unimplemented Linear symbolic forward for {configuration}"
            )

        return output
