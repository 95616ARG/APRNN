from datetime import datetime
import os
import re
from typing import Iterable, overload
import random
import warnings
import numpy as np
import torch
from numpy import ndarray
from torch import Tensor
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import asyncio
import itertools
from itertools import chain
import tqdm as _tqdm
from tqdm.auto import tqdm
import gurobipy as gp
from gurobipy import GRB
import asyncio

from timeit import default_timer as timer
import itertools

__all__ = [
    'cat',
    'timeit',
    '_timeit',
    '_timeit2',
    '_timeit_debug',
    'vtqdm',
    'vtqdm2',
    'timer',
    'tqdm',
    'gp',
    'GRB',

    'asyncio',
    'map_with_kwargs',

    '_get_rng',
    'random_mask_1d',
    'random_mask_2d',
    'set_chunksize',
    'is_debugging',
    'enable_debug',
    'disable_debug',
    'get_executor',
    'get_max_workers',
    '_get_timestamp',
    'as_global',
    'GlobalRegister',
    'set_all_seed',
    'np',
    'ndarray',
    'torch',
    'Tensor',
    'ProcessPoolExecutor',
    'ThreadPoolExecutor',
    'asyncio',
    'itertools',

    'add_method_to_class',
    'split_and_broadcast_for_argmax',
    'get_dispatch_multiplier',
    'set_dispatch_multiplier',

    'torch_dtype_to_numpy',
    'numpy_dtype_to_torch',

    '_raise',
    '_assert',

    '_as_strided_window_views2d',
    '_as_strided_window_views2d_numpy',
    '_as_strided_window_views2d_torch',

    'as_slice',
    '_name_to_indices',
    'flatten',
    '_iter_nested_iterable',
    '_iter_over_leading_axes',
    'broadcast_at',

    'to_shared_array',
    'optimal_chunksize',

    'isin_vpoly',
    'onnx_forward',
    'unit_box',
    'points_to_hboxes',
    'hboxes_to_vboxes',
    'points_to_vboxes',
    'intersect_hbox',
    'sample_hbox',
    'partition_hbox',
    'center_of_hboxes',
    'meet_patterns',
    'stack_patterns',
    'cat_patterns',
    'calculate_vertices_and_pattern',
    'dissection_into_simplices',
    'centroid_of_vpolytopes',
    'sample_around_points',
]

def cat(arrays, *args, **kwargs):
    ty = None
    for arr in arrays:
        if not isinstance(arr, Tensor):
            ty = type(arr)

    if ty is None:
        return torch.cat(arrays, **args, **kwargs)

    else:
        return ty.concatenate(arrays, *args, **kwargs)

def sample_around_points(points, eps, nsamples, dims=None, flatten=False, validate=True):
    N, *S = points.shape

    base = (points - eps)[:,None]
    # print(f'sampling {nsamples} samples around points {points.shape} with e={eps}')
    samples = torch.rand((N, nsamples, *S), device=points.device, dtype=points.dtype) * (eps * 2) + base

    if dims is not None:
        base = torch.zeros_like(samples, dtype=points.dtype, device=points.device)
        base[:] = points[:,None]
        base[...,dims] = samples[...,dims]
        samples = base

    if validate:
        assert (points[:,None] - eps <= samples).all() and (samples <= points[:,None] + eps).all()

    if flatten:
        samples = samples.flatten(0,1)

    return samples.to(device=points.device)

def onnx_forward(path, x):
    import onnxruntime as ort
    with torch.no_grad():
        return ort.InferenceSession(path).run(['output'], {'input': x.numpy()})

def order_mask(box, ord):
    mask = torch.ones(box.shape[0], dtype=bool)
    for i, j in zip(ord[:-1], ord[1:]):
        mask *= (box[...,i] <= box[...,j])
    return mask

def unit_box(ndim, dims=None, groups=None):
    """_summary_

    Args:
        ndim (int): _description_

    Returns:
        Tensor[2^ndim, ndim]: _description_
    """
    if dims is not None:
        assert groups is None
        l = [torch.zeros((1,)) for _ in range(ndim)]
        for d in dims:
            l[d] = torch.tensor([-1., 1.])

        return torch.stack(
            torch.meshgrid( *l )
        ).permute(tuple(range(ndim,-1,-1))).reshape(-1, ndim)

    elif groups is not None:
        uniq_verts = unit_box(ndim=len(groups))
        vertices = torch.zeros( (uniq_verts.shape[0], ndim) )
        for gi, g in enumerate(groups):
            vertices[:,g] = uniq_verts[:, [gi]]

        return vertices

    else:
        l = (torch.linspace(-1., 1., 2),)*ndim

        return torch.stack(
            torch.meshgrid( *l )
        ).permute(tuple(range(ndim,-1,-1))).reshape(-1, ndim)

def dissection_into_simplices(ndim):
    box = unit_box(ndim)
    return torch.stack(
        tuple(
            torch.where(order_mask(box, perm))[0]
            for perm in itertools.permutations(range(ndim), ndim)
        )
    )

dissection_into_simplices_5D = dissection_into_simplices(5)

def calculate_vertices_and_pattern_dissection(boxes, pre, post, net_ref):
    """_summary_

    Args:
        boxes (_type_): _description_
        pre (_type_): _description_
        post (_type_): _description_

    Returns:
        _type_: _description_
    """
    # print('dissection')
    device = pre.device or post.device
    dtype = pre.dtype or post.dtype
    boxes = boxes.to(device,dtype)

    box_vertices = hboxes_to_vboxes(boxes, flatten=False)
    pre_box_vertices = pre(box_vertices)
    # centers = pre(center_of_hboxes(boxes))

    pattern_dict = dict()
    from tqdm.auto import tqdm
    with timeit("each", False):
        for box, vertices, pre_vertices in zip(boxes, box_vertices, pre_box_vertices):

            pre_centers = torch.stack(
                tuple(pre_vertices[simplex_indices]
                      for simplex_indices in dissection_into_simplices_5D),
                0
            ).mean(1)
            patterns = post.activation_pattern(pre_centers)
            patterns = [
                [p if len(p) == 0 else p[i] for p in patterns]
                for i in range(pre_centers.shape[0])
            ]

            for pattern, simplex_indices in zip(patterns, dissection_into_simplices_5D):
                simplex_vertices = vertices[simplex_indices]
                for vert in simplex_vertices:
                    keyable = tuple(vert.tolist())
                    if keyable not in pattern_dict:
                        pattern_dict[keyable] = [pattern]
                    else:
                        pattern_dict[keyable].append(pattern)

            # for simplex_indices in dissection_into_simplices_5D:

            #     pre_simplex_vertices = pre_vertices[simplex_indices]
            #     center = pre_simplex_vertices.mean(0)
            #     pattern = post.activation_pattern(center)

            #     simplex_vertices = vertices[simplex_indices]
            #     for vert in simplex_vertices:
            #         keyable = tuple(vert.tolist())
            #         if keyable not in pattern_dict:
            #             pattern_dict[keyable] = [pattern]
            #         else:
            #             pattern_dict[keyable].append(pattern)

    with timeit("meet", False):
        pattern_dict = {
            k: meet_patterns(*v)
            for k, v in pattern_dict.items()
        }

    with timeit("post", False):
        box_vertices_unique = box_vertices.flatten(0,1).unique(dim=0)
        pre_vertices_unique = pre(box_vertices_unique)
        box_vertices_pattern = stack_patterns(*(
            pattern_dict[tuple(vert.tolist())]
            for vert in box_vertices_unique
        ))
        ref_output = net_ref(box_vertices.flatten(0, 1).unique(dim=0))
    return pre_vertices_unique, box_vertices_pattern, ref_output

def calculate_vertices_and_pattern(boxes, pre, post, net_ref, dissection=False, local_rubustness=False):
    """_summary_

    Args:
        boxes (_type_): _description_
        pre (_type_): _description_
        post (_type_): _description_

    Returns:
        _type_: _description_
    """
    assert local_rubustness == False
    if dissection:
        return calculate_vertices_and_pattern_dissection(boxes, pre, post, net_ref=net_ref)

    device = pre.device or post.device
    dtype = pre.dtype or post.dtype
    boxes = boxes.to(device,dtype)

    box_vertices = hboxes_to_vboxes(boxes, flatten=False)
    # pre_box_vertices = pre(box_vertices)
    centers = pre(center_of_hboxes(boxes))

    pattern_dict = dict()
    for box, vertices, center in zip(boxes, box_vertices, centers):
        pattern = post.activation_pattern(center)
        for vert in vertices:
            keyable = tuple(vert.tolist())
            if keyable not in pattern_dict:
                pattern_dict[keyable] = [pattern]
            else:
                pattern_dict[keyable].append(pattern)

    pattern_dict = {
        k: meet_patterns(*v)
        for k, v in pattern_dict.items()
    }

    box_vertices_unique = box_vertices.flatten(0,1).unique(dim=0)
    pre_vertices_unique = pre(box_vertices_unique)
    box_vertices_pattern = stack_patterns(*(
        pattern_dict[tuple(vert.tolist())]
        for vert in box_vertices_unique
    ))
    ref_output = net_ref(box_vertices.flatten(0, 1).unique(dim=0))

    # if local_rubustness:
    #     labels = post(centers).argmin(-1)
    #     label_dict = dict()
    #     for vertices, label in zip(box_vertices, labels):
    #         for vert in vertices:
    #             keyable = tuple(vert.tolist())
    #             if keyable not in label_dict:
    #                 label_dict[keyable] = label
    #             else:
    #                 if not (label_dict[keyable] == label).all():
    #                     print(label_dict[keyable], label)

    #     box_vertices_label = torch.stack(tuple(
    #         label_dict[tuple(vert.tolist())]
    #         for vert in box_vertices_unique
    #     ), dim=0)[...,None]
    #     print(box_vertices_label.shape)
    #     return pre_vertices_unique, box_vertices_pattern, box_vertices_label, ref_output

    # else:
    return box_vertices_unique, pre_vertices_unique, box_vertices_pattern, ref_output

class as_slice:
    def __getitem__(self, idx):
        return idx
as_slice = as_slice()

def _export(func):
    """ An helper decorator to dynamically import a function to module's
    __all__, although this can't help static analyzers. Wondering if there are
    some similar built-in decorators like @overload which is supported by
    pylance or mypy.
    """
    global __all__
    if func.__name__ not in __all__:
        __all__.append(func.__name__)
    return func

def _unwrap_closure(func):
    return func()

from contextlib import contextmanager
import sys, os

@contextmanager
def suppress_stdout():
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        try:
            yield
        finally:
            sys.stdout = old_stdout

def isin_vpoly(point, vertices):
    import gurobipy as gp
    # Check if the dimensions of point and vertices match
    if point.shape[0] != vertices.shape[1]:
        raise ValueError("The dimensions of the point and the vertices do not match.")

    from sytorch.solver import GurobiSolver

    with suppress_stdout():
        solver = GurobiSolver().verbose_(False)
        solver.solver.setParam('Method', -1)
        weights = solver.reals(vertices.shape[0])
        solver.solve(
            0. <= weights, weights.sum() <= 1.,
            minimize=(point - vertices.T @ weights).norm_ub('linf+l1_normalized'),
        )
        return solver.solver.objVal < 1e-8

    # # Create a new model
    # model = gp.Model("convex_hull")

    # # Disable Gurobi output
    # model.setParam('OutputFlag', 0)
    # model.setParam('Threads', os.cpu_count())

    # # Create the variables and constraints
    # weights = model.addVars(vertices.shape[0], lb=0, ub=1, vtype=GRB.CONTINUOUS, name="weights")
    # model.addConstr(gp.quicksum(weights[i] for i in range(vertices.shape[0])) == 1)

    # # Set the objective function
    # objective = gp.quicksum(((point - vertices.T @ weights)[i]) ** 2 for i in range(point.shape[0]))
    # model.setObjective(objective, GRB.MINIMIZE)

    # # Solve the linear programming problem
    # model.optimize()

    # # Check if the minimum distance is close enough to zero
    # return model.objVal < 1e-8

def isin_vpoly_cvxpy(point, vertices):
    import cvxpy as cp
    # Check if the dimensions of point and vertices match
    if point.shape[0] != vertices.shape[1]:
        raise ValueError("The dimensions of the point and the vertices do not match.")

    # Define the variables and constraints for linear programming
    weights = cp.Variable(vertices.shape[0])
    constraints = [
        cp.sum(weights) == 1,
        weights >= 0
    ]

    # Set the objective function
    objective = cp.Minimize(cp.norm(point - vertices.T @ weights))

    # Solve the linear programming problem
    problem = cp.Problem(objective, constraints)
    problem.solve()

    # Check if the minimum distance is close enough to zero
    return problem.value < 1e-8

def points_to_hboxes(points, size):
    return torch.stack((points-size, points+size), -1)

def points_to_vboxes(points, size, dims=None, groups=None, flatten=True):
    """_summary_

    Args:
        points (Tensor[N, C]): _description_
        size (float): _description_
        flatten (bool, optional): _description_. Defaults to True.

    Returns:
        _type_: _description_
    """
    boxes = size * unit_box(points.shape[-1], dims=dims, groups=groups)
    # boxes = broadcast_at(boxes, 0, points.shape[:1]) # (N, 2**C, C)
    vertices = points[:,None,:] + boxes
    if flatten:
        vertices = vertices.flatten(0,1)
    return vertices

def hboxes_to_vboxes(hboxes, flatten=True):
    """_summary_

    Args:
        hboxes (Tensor[N, C, 2]): _description_
        flatten (bool, optional): _description_. Defaults to True.

    Returns:
        Tensor[N, 2^C, C]: _description_
    """
    assert hboxes.ndim == 3
    ndim = hboxes.shape[-2]
    dim_order = tuple(range(ndim,-1,-1))
    vertices = torch.stack(tuple(
        torch.stack(torch.meshgrid(*hbox, indexing='ij'))\
            .permute(*dim_order)\
            .reshape(-1, ndim)
        for hbox in hboxes
    ), dim=0)
    if flatten:
        vertices = vertices.flatten(0, 1)
    return vertices

def vboxes_to_unique_vertices(vboxes):
    assert vboxes.ndim in [2, 3]
    if vboxes.ndim == 3:
        vboxes = vboxes.flatten(0, 1)
    return vboxes.unique(dim=0)

""" test

    import sytorch as st
    from sytorch.pervasives import *

    st.set_all_seed(0)
    points = st.randn(2, 5)
    boxes = st.points_to_hboxes(points, .1)
    vertices = st.hboxes_to_vboxes(boxes, flatten=False)
    v2 = st.points_to_vboxes(points, .1, flatten=False)
    assert (vertices == v2).all()

"""

def intersect_hbox(box1, box2):
    """_summary_

    Args:
        box1 (Tensor[C, 2]): _description_
        box2 (Tensor[C, 2]): _description_

    Returns:
        Optional[Tensor[C, 2]]: intersection box if intersected, otherwise None.
    """

    assert box1.ndim == 2 and box1.shape == box2.shape

    lbs = torch.stack((box1[...,0], box2[...,0]), -1)
    ubs = torch.stack((box1[...,1], box2[...,1]), -1)
    intersection = torch.stack((
        lbs.amax(dim=-1),
        ubs.amin(dim=-1),
    ), -1)

    if (intersection[...,0] <= intersection[...,1]).all():
        return intersection
    else:
        return None

def center_of_hboxes(boxes):
    return boxes.sum(-1) / 2.

def centroid_of_vpolytopes(vpolytopes, dim=1):
    # print('taking centroid of', vpolytopes.shape)
    return vpolytopes.mean(dim)

def sample_hbox(boxes, h=.1, include_center=False):
    """_summary_

    Args:
        box (Tensor[5, 2]): _description_
        h (float, optional): _description_. Defaults to .1.

    Returns:
        _type_: _description_
    """
    assert boxes.ndim in [2, 3]
    if boxes.ndim == 2:
        end_points = []
        for lb, ub in boxes:
            num = int(torch.ceil((ub - lb) / h))
            if num == 0:
                num = 1
            _points = torch.linspace(lb, ub, num+1, dtype=boxes.dtype, device=boxes.device)
            if len(_points) > 2:
                _points = _points[1:-1] # not include enc points
            elif len(_points) == 2:
                _points = _points.sum(dim=-1, keepdim=True) / 2.
            assert len(_points) > 0
            end_points.append(_points)

        points = torch.stack(torch.meshgrid(*end_points, indexing='ij'))
        points = points.permute(tuple(range(points.ndim-1, -1, -1))).reshape(-1, 5)
        if include_center:
            points = torch.cat((center_of_hboxes(boxes[None]), points), dim=0)
        return points
    else:
        return tuple(sample_hbox(box, h=h, include_center=include_center) for box in boxes)

def partition_hbox(box, h=.1, gap=.0):
    """_summary_

    Args:
        box (Tensor[5, 2]): _description_
        h (float, optional): _description_. Defaults to .1.

    Returns:
        _type_: _description_
    """
    # end_points = []
    ndim_bounds = []
    for lb, ub in box:
        num = max(int(torch.ceil((ub - lb) / h)), 1)
        end_points = torch.linspace(lb, ub, num+1, dtype=box.dtype, device=box.device)
        bounds = []
        for i in range(len(end_points) - 2):
            bounds.append((end_points[i], end_points[i+1] - gap))
        bounds.append(tuple(end_points[-2:]))
        ndim_bounds.append(bounds)

    return torch.stack(tuple(torch.tensor(i) for i in itertools.product(*ndim_bounds)))

def meet_patterns(*patterns):
    if isinstance(patterns[0], np.ndarray):
        return np.vectorize(
            (lambda ps: \
                1 if (ps == 1).all() else \
                0 if (ps == 0).all() else \
                -1),
            otypes=[int],
            signature="(n)->()"
        )(np.stack(patterns, -1))

    elif isinstance(patterns[0], list):
        if patterns[0] == []:
            return []

        return [
            meet_patterns(*patterns_elems)
            for patterns_elems in zip(*patterns)
        ]

    else:
        raise NotImplementedError

def cat_patterns(*patterns, axis=0):
    if isinstance(patterns[0], np.ndarray):
        return np.concatenate(patterns, axis=axis)

    elif isinstance(patterns[0], list):
        if patterns[0] == []:
            return []

        return [
            cat_patterns(*patterns_elems)
            for patterns_elems in zip(*patterns)
        ]

    else:
        raise NotImplementedError

def stack_patterns(*patterns, axis=0):
    if isinstance(patterns[0], np.ndarray):
        return np.stack(patterns, axis=axis)

    elif isinstance(patterns[0], list):
        if patterns[0] == []:
            return []

        return [
            stack_patterns(*patterns_elems)
            for patterns_elems in zip(*patterns)
        ]

    else:
        raise NotImplementedError

# def points_to_vboxes(points, size, flatten=True):
#     return (points + size * unit_box(points.shape[-1]))

def _get_rng(seed_or_rng):
    if isinstance(seed_or_rng, int):
        rng = np.random.default_rng(seed_or_rng)

    elif hasattr(seed_or_rng, 'shuffle'):
        rng = seed_or_rng

    else:
        assert seed_or_rng is None
        rng = np.random.random.__self__

    return rng

def random_mask_1d(nrows, n, seed=None):
    rng = _get_rng(seed)

    if isinstance(n, float):
        n = int(np.ceil(nrows * n))
    mask = np.zeros((nrows,), dtype=bool)
    mask[:n] = True
    rng.shuffle(mask)

    return mask

def random_mask_2d(shape, choice, seed=None):
    rng = _get_rng(seed)

    nrows, ncols = shape
    rows, cols = choice
    mask = np.zeros(shape, dtype=bool)
    rows_mask = random_mask_1d(nrows, rows, rng)
    mask[rows_mask] = np.stack(tuple(
        random_mask_1d(ncols, cols, rng)
        for _ in range(rows_mask.shape[0])
    ))
    return mask

_torch_dtype_to_numpy_dict = {
    torch.float16: np.float16,
    torch.float32: np.float32,
    torch.float64: np.float64,
}

def torch_dtype_to_numpy(dtype):
    return _torch_dtype_to_numpy_dict[dtype]

_numpy_dtype_to_torch_dict = {
    np.float16: torch.float16,
    np.float32: torch.float32,
    np.float64: torch.float64,
}

def numpy_dtype_to_torch(dtype):
    return _numpy_dtype_to_torch_dict[dtype]

def map_with_kwargs(fn, *args_iter, **kwargs):
    for args in zip(*args_iter):
        yield fn(*args, **kwargs)

def vtqdm2(it, *args, **kwargs):
    if is_verbose():
        return tqdm(it, *args, **kwargs)
    else:
        return it

class vtqdm(_tqdm.auto.tqdm):
    def close(self): ...

    def reset(self, total=None):
        """
        Resets to 0 iterations for repeated use.

        Consider combining with `leave=True`.

        Parameters
        ----------
        total  : int or float, optional. Total to use for the new bar.
        """

        _, pbar, _ = self.container.children
        pbar.bar_style = ''
        if total is not None:
            pbar.max = total
            if not self.total and self.ncols is None:  # no longer unknown total
                pbar.layout.width = None  # reset width

        self.n = 0
        if total is not None:
            self.total = total
        if self.disable:
            return
        self.last_print_n = 0
        # self.last_print_t = self.start_t = self._time()
        # self._ema_dn = EMA(self.smoothing)
        # self._ema_dt = EMA(self.smoothing)
        # self._ema_miniters = EMA(self.smoothing)
        self.refresh()

    def __call__(self, iterable=None, desc=None, total=None, leave=None):
        self.iterable = iter(iterable) if iterable is not None else iterable
        self.desc = desc if desc is not None else self.desc
        self.leave = leave if leave is not None else self.leave
        self.reset(total=total)
        return self

""" Debugging mode. """
_is_debugging = False
def is_debugging():
    global _is_debugging
    return _is_debugging

def enable_debug():
    global _is_debugging
    _is_debugging = True

def disable_debug():
    global _is_debugging
    _is_debugging = False

""" Verbosity. """
_is_verbose = False
def is_verbose():
    global _is_verbose
    return _is_verbose

def enable_verbosity():
    global _is_verbose
    _is_verbose = True

def disable_verbosity():
    global _is_verbose
    _is_verbose = False

_dispatch_multiplier = 1
def get_dispatch_multiplier():
    global _dispatch_multiplier
    return int(np.ceil(_dispatch_multiplier))

def set_dispatch_multiplier(v):
    global _dispatch_multiplier
    _dispatch_multiplier = v

_executor = None
def get_executor(*args, **kwargs):
    global _executor
    if _executor is None:
        _executor = ProcessPoolExecutor(os.cpu_count())
    return _executor

def get_max_workers():
    return get_executor()._max_workers

def _get_timestamp():
    return re.sub('[\s_\-\:\.]', '_', f"{datetime.now()}")

class GlobalRegister:
    """ A context manager that temporary registers objects, e.g.,
    local/lambda functions that is gonna to be pickled and sent to other
    processes, to the top-level (global) environment with an __unique__
    qualified name. It restores everything when leaving the context/scope.
    """
    def __init__(self, module_dict, *objs):
        self.module_dict = module_dict
        self.objs = set(objs)
        self.qualnames = { id(obj): obj.__qualname__ for obj in self.objs }

    def register(self):
        import __main__
        for obj in self.objs:
            timestamp = _get_timestamp()
            uid = f"__{obj.__name__}__{id(obj)}__{timestamp}__"
            obj.__qualname__ = uid
            self.module_dict[uid] = obj

            # assert getattr(__main__, uid, None) is None
            # setattr(__main__, uid, obj)

            if getattr(__main__, uid, None) is None:
                setattr(__main__, uid, obj)

    def unregister(self):
        import __main__
        for obj in self.objs:
            uid = obj.__qualname__
            obj.__qualname__ = self.qualnames[id(obj)]
            del self.module_dict[uid]

            # assert getattr(__main__, uid, None) is obj
            # delattr(__main__, uid)

            if getattr(__main__, uid, None) is obj:
                delattr(__main__, uid)

    def __enter__(self):
        self.register()

    def __exit__(self, *args, **kwargs):
        self.unregister()

as_global = GlobalRegister

def set_all_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

def _raise(err):
    raise err

def _assert(cond, msg=None):
    assert cond, msg

def as_kwargs(**kwargs):
    return kwargs

def _name_to_indices(name):
    return tuple(
        map(lambda i: \
                int(i) if i.isdigit() else i,
            name.split('.'))
    )

def _iter_nested_iterable(arr):
    if isinstance(arr, str):
        yield arr
    elif isinstance(arr, ndarray):
        yield from arr.flat
    elif isinstance(arr, (tuple, list)):
        for a in arr:
            yield from _iter_nested_iterable(a)
    else:
        yield arr

class _timeit_debug:
    def __init__(self, prompt):
        self.prompt = prompt

    def __enter__(self):
        self.enabled = is_debugging()
        if self.enabled:
            # print(self.prompt, " ... ", end='')
            self.start = timer()

    def __exit__(self, exc_type, exc_value, exc_traceback):
        # if self.enabled:
        #     print(f"{timer() - self.start:.3f}s")
        self.enabled = False


class timeit:
    def __init__(self, prompt, mode=True):
        self.prompt = prompt
        self.mode = mode

    def __enter__(self):
        if self.mode == True:
            # print(f" > {self.prompt}")
            self.start = timer()

    def __exit__(self, exc_type, exc_value, exc_traceback):
        # if self.mode == True:
        #     print(f" < {self.prompt}: {timer() - self.start:.3f}s ")
        pass

class _timeit2:
    def __init__(self, prompt):
        self.prompt = prompt

    def __enter__(self):
        # print(self.prompt, " ... ", end='')
        self.start = timer()

    def __exit__(self, exc_type, exc_value, exc_traceback):
        # print(f"{timer() - self.start:.3f}s")
        pass

class _timeit:
    def __init__(self, prompt):
        return
        self.prompt = prompt

    def __enter__(self):
        return
        print(self.prompt, " ... ", end='')
        self.start = timer()

    def __exit__(self, exc_type, exc_value, exc_traceback):
        return
        print(f"{timer() - self.start:.3f}s")

def flatten(*arrays, start_dim=0, end_dim=-1):
    if len(arrays) == 1:
        array = arrays[0]
        assert -array.ndim-1 < end_dim and end_dim < array.ndim, f"{end_dim}, {array.ndim}, {array.shape}, {array}"
        end_dim = (end_dim + array.ndim) % array.ndim
        return array.reshape(
            array.shape[:start_dim] +
            (np.prod(array.shape[start_dim:end_dim+1], dtype=int),) +
            array.shape[end_dim+1:]
        )
    else:
        return tuple(
            flatten(array, start_dim=start_dim, end_dim=end_dim)
            for array in arrays
        )

def _iter_over_leading_axes(array, leading_ndims_or_shape, num_chunks=None):
    if isinstance(leading_ndims_or_shape, int):
        leading_ndims_or_shape = tuple(array.shape[:leading_ndims_or_shape])

    if num_chunks is not None:
        num_chunks *= 2

        array = flatten(array, start_dim=0, end_dim=len(leading_ndims_or_shape))

        chunksize = (array.shape[0] // num_chunks) + 1

        for start in range(0, num_chunks, chunksize):
            end = min(start+chunksize, array.shape[0])
            yield array[start:end]

    else:
        for idx in np.ndindex(*leading_ndims_or_shape):
            yield array[idx]

def broadcast_at(array, *args, subok=True, writeable=False):
    output_shape  = array.shape
    if isinstance(array, Tensor):
        output_strides = array.stride()
    else:
        output_strides = array.strides

    if len(args) == 1 and isinstance(args[0], dict):
        spec = []
        for start, shape in args[0].items():
            spec.append(start)
            spec.append(shape)
    else:
        spec = args

    for start, shape in sorted(zip(spec[0::2], spec[1::2]), reverse=True):
        if start < 0:
            raise NotImplementedError("unimplemented negative indexing in broadcast_at.")
        output_shape   = output_shape[:start] + tuple(shape) + output_shape[start:]
        output_strides = output_strides[:start] + tuple(0 for _ in range(len(shape))) + output_strides[start:]

    if isinstance(array, Tensor):
        return torch.as_strided(
            array,
            size     = output_shape,
            stride   = output_strides,
        )

    else:
        return np.lib.stride_tricks.as_strided(
            array,
            shape     = output_shape,
            strides   = output_strides,
            subok     = subok,
            writeable = writeable,
        )

def _output_grid_shape2d(input_shape, kernel_size, padding=(0,0), stride=(1,1), dilation=(1,1), ceil_mode=False):
    rounding_func = np.ceil if ceil_mode else np.floor
    return (
        int(rounding_func(((input_shape[0] + 2 * padding[0] - dilation[0] * (kernel_size[0] - 1) - 1) / stride[0]) + 1)),
        int(rounding_func(((input_shape[1] + 2 * padding[1] - dilation[1] * (kernel_size[1] - 1) - 1) / stride[1]) + 1)),
    )

def _to_torch_padding_mode(mode):
    if mode == 'zeros':
        return 'constant'

    elif mode == 'edge':
        return 'replicate'

    elif mode == 'wrap':
        return 'circular'

    else:
        assert mode in ('constant', 'reflect', 'replicate', 'circular')
        return mode

def _to_numpy_padding_mode(mode):
    if mode == 'zeros':
        return 'constant'

    elif mode == 'replicate':
        return 'edge'

    elif mode == 'circular':
        return 'wrap'

    else:
        assert mode in ('constant', 'edge', 'reflect', 'wrap')
        return mode

def _as_strided_window_views2d_torch(
    input,
    kernel_size,
    padding=(0,0),
    padding_mode='constant',
    stride=(1,1),
    dilation=(1,1),
    ceil_mode=False,
    value=0,
):
    if value == 'zero':
        value = 0.

    from torch.nn.modules.conv import _pair
    kernel_size = _pair(kernel_size)
    padding     = _pair(padding)
    stride      = _pair(stride)
    dilation    = _pair(dilation)

    assert dilation == (1, 1)
    padding_mode = _to_torch_padding_mode(padding_mode)
    assert padding_mode in ('constant', 'reflect', 'replicate', 'circular')

    output_grid_shape = _output_grid_shape2d(input_shape=input.shape[-2:], kernel_size=kernel_size, padding=padding, stride=stride, dilation=dilation, ceil_mode=ceil_mode)

    if ceil_mode:
        padding = (
            padding[1], padding[1] + max(kernel_size[1] - input.shape[-1] % stride[1], 0),
            padding[0], padding[0] + max(kernel_size[0] - input.shape[-2] % stride[0], 0),
        )
    else:
        padding = (padding[1], padding[1], padding[0], padding[0])

    if isinstance(input, np.ndarray):
        input = torch.from_numpy(input)

    if padding_mode == 'constant':
        input = torch.nn.functional.pad(input, pad=padding, mode=padding_mode, value=value).contiguous()
    else:
        assert value == 0
        input = torch.nn.functional.pad(input, pad=padding, mode=padding_mode).contiguous()

    """ Create inplace sub-views of input array's underlying data with the specified strides . """
    return torch.as_strided(
        # (num_batches, in_channels, input_width, input_height).
        input,

        # (num_batches, in_channels, kernel_width, kernel_height, output_width, output_height).
        size     = input.shape[:-2] + kernel_size + output_grid_shape,

        # Strides for indexing the original and output grids.
        stride   = input.stride() + tuple(stride_length * stride_steps for stride_length, stride_steps in zip(input.stride()[2:], stride)),
    ) #.reshape(input.shape[:-2] + (-1,) + output_grid_shape)


def _numpy_pad_func_constant_obj(vector, pad_width, iaxis, kwargs):
    assert pad_width[0] >= 0 and pad_width[1] >= 0
    if pad_width[0] > 0:
        # assert (vector[:pad_width[0]] == 0.).all()
        vector[:pad_width[0]] = kwargs['constant_values']

    if pad_width[1] > 0:
        # assert (vector[-pad_width[1]:] == 0.).all()
        vector[-pad_width[1]:] = kwargs['constant_values']

def _as_strided_window_views2d_numpy(
    input,
    kernel_size,
    padding=(0,0),
    padding_mode='constant',
    stride=(1,1),
    dilation=(1,1),
    ceil_mode=False,
    value=0,
):
    if value == 'zero':
        if hasattr(input, 'solver'):
            value = getattr(input.solver, '_zero', 0.)
        else:
            value = 0.

    original_input = input
    from torch.nn.modules.conv import _pair
    kernel_size = _pair(kernel_size)
    padding     = _pair(padding)
    stride      = _pair(stride)
    dilation    = _pair(dilation)

    assert dilation == (1, 1)
    padding_mode = _to_numpy_padding_mode(padding_mode)
    assert padding_mode in ('constant', 'edge', 'reflect', 'wrap')

    output_grid_shape = _output_grid_shape2d(input_shape=input.shape[-2:], kernel_size=kernel_size, padding=padding, stride=stride, dilation=dilation, ceil_mode=ceil_mode)

    if ceil_mode:
        padding = (
            (padding[0], padding[0] + max(kernel_size[0] - input.shape[-2] % stride[0], 0)),
            (padding[1], padding[1] + max(kernel_size[1] - input.shape[-1] % stride[1], 0)),
        )
    else:
        padding = tuple((p, p) for p in padding)

    padding = ((0, 0), ) * (input.ndim - 2) + padding
    if isinstance(input, Tensor):
        input = input.cpu().detach().numpy()

    if padding_mode == 'constant':
        input = np.pad(input, pad_width=padding, mode=_numpy_pad_func_constant_obj, constant_values=value)
    else:
        input = np.pad(input, pad_width=padding, mode=padding_mode)

    assert input.data.contiguous

    """ Create inplace sub-views of input array's underlying data with the specified strides . """
    views = np.lib.stride_tricks.as_strided(
        # (num_batches, in_channels, input_width, input_height).
        input,

        # (num_batches, in_channels, kernel_width, kernel_height, output_width, output_height).
        shape    = input.shape[:-2] + kernel_size + output_grid_shape,

        # Strides for indexing the original and output grids.
        strides  = input.strides + tuple(stride_length * stride_steps for stride_length, stride_steps in zip(input.strides[2:], stride)),

        # Preserve the `SymbolicArray` subtype.
        subok     = True,

        # Create read-only strided views.
        writeable = False
    )

    if hasattr(original_input, 'solver'):
        views = type(original_input)(views, solver=original_input.solver)

    return views #.reshape(input.shape[:-2] + (-1,) + output_grid_shape)

def _as_strided_window_views2d(
    input,
    kernel_size,
    padding=(0,0),
    padding_mode='constant',
    stride=(1,1),
    dilation=(1,1),
    ceil_mode=False,
    value=0,
):
    if isinstance(input, Tensor):
        return _as_strided_window_views2d_torch(
            input,
            kernel_size  = kernel_size,
            padding      = padding,
            padding_mode = padding_mode,
            stride       = stride,
            dilation     = dilation,
            ceil_mode    = ceil_mode,
            value        = value,
        )
    else:
        return _as_strided_window_views2d_numpy(
            input,
            kernel_size  = kernel_size,
            padding      = padding,
            padding_mode = padding_mode,
            stride       = stride,
            dilation     = dilation,
            ceil_mode    = ceil_mode,
            value        = value,
        )

def to_shared_array(t):
    if isinstance(t, Tensor):
        return t.cpu().detach().numpy()
    return t

_chunksize = None
def set_chunksize(chunksize):
    global _chunksize
    _chunksize = chunksize

def get_chunksize():
    global _chunksize
    return _chunksize

def optimal_chunksize(
    unit_work,
    num_units,
    num_workers,
    optimal_work_per_process,
    max_workers = None,
):

    chunksize = get_chunksize() or int(np.ceil(max(
        num_units / (num_workers * 10),             # minimal chunksize to utilize all workers.
        optimal_work_per_process / unit_work # minimal chunksize to keep each worker busy.
    )))

    # if is_debugging():
    #     print(
    #         f"maxworker: {max_workers}\n"
    #         f"chunksize: {chunksize}\n"
    #         f"chunkwork: {chunksize * unit_work}\n"
    #         f"numchunks: {int(np.ceil(num_units / chunksize))}"
    #     )

    return chunksize

def split_and_broadcast_for_argmax(array, argmax, axis=0):
    assert axis == 0
    N = array.shape[axis]
    other_indices = list(idx for idx in range(N) if idx != argmax)
    other_array = array[other_indices,...]
    argmax_array = broadcast_at(array[argmax,...], 0, (N-1,))
    return argmax_array, other_array

def add_method_to_class(*classes):
    def decorator(method):
        for cls in classes:
            if getattr(cls, method.__name__, None) is not None:
                warnings.warn(
                    f"overwriting class `{cls.__name__}`'s method `{method.__name__}(...)`."
                )
            setattr(cls, method.__name__, method)
        return method
    return decorator
