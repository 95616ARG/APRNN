from __future__ import annotations
from typing import Literal, overload

import re, os, pathlib, datetime
from collections import defaultdict
import multiprocessing
import dill

import torch.nn.functional as F
import sytorch
from sytorch.pervasives import *
from sytorch.pervasives import is_verbose
from sytorch.solver.symbolic_array import *
from sytorch.solver.base import *
from sytorch.solver.gurobi import GurobiSolver
from tqdm.auto import tqdm

# https://stackoverflow.com/questions/35088139/how-to-make-a-thread-safe-global-counter-in-python
class _Counter(object):
    def __init__(self):
        # RawValue because we don't need it to create a Lock:
        self.val = multiprocessing.RawValue('i', 0)
        self.lock = multiprocessing.Lock()

    def allocate(self, n):
        with self.lock:
            offset = self.val.value
            self.val.value += n
            # if is_verbose():
            #     print(f"allocated {offset} to {offset+n}.")
            return np.arange(start=offset, stop=offset+n, step=1, dtype=int)

    def increment(self, n=1):
        with self.lock:
            self.val.value += n

    def reset(self):
        with self.lock:
            self.val.value = 0

    @property
    def value(self):
        with self.lock:
            return self.val.value

class LightningVar:
    __slots__ = ['_id', 'solver']
    def __init__(self, id, solver, lb=None, ub=None):
        self._id = id
        self.solver = solver
        if lb is not None: self.lb = lb
        if ub is not None: self.ub = ub

    @property
    def id(self):
        return self._id

    def make_semi_positive(self):
        assert self.UB is None or self.UB >= 0.
        if self.LB is None or self.LB < 0.:
            self.LB = 0.

    def make_semi_negative(self):
        assert self.LB is None or self.LB <= 0.
        if self.UB is None or self.UB > 0.:
            self.UB = 0.

    @property
    def is_semi_positive(self):
        lb = self.LB
        assert lb is not None
        return lb >= 0.

    @property
    def definitely_semi_positive(self):
        lb = self.LB
        if lb is None:
            return False
        else:
            return lb >= 0.

    @property
    def is_semi_negative(self):
        ub = self.UB
        assert ub is not None
        return ub <= 0.

    @property
    def definitely_semi_negative(self):
        ub = self.UB
        if ub is None:
            return False
        else:
            return ub <= 0.

    @property
    def lb(self):
        return self.solver.lb[self.id]

    @lb.setter
    def lb(self, value):
        self.solver.lb[self.id] = value

    @property
    def ub(self):
        return self.solver.ub[self.id]

    @ub.setter
    def ub(self, value):
        self.solver.ub[self.id] = value

    @property
    def bound(self):
        return (self.lb, self.ub)

    @bound.setter
    def bound(self, value):
        self.lb, self.ub = value

    # To be compatible with GurobiVar.
    LB = lb
    UB = ub

    def to(self, solver):
        if isinstance(solver, GurobiSolver):
            var = solver.solver.getVarByName(str(self))
            # if not ((var.LB == -np.inf and self.LB is None) or np.allclose(var.LB, self.lb)):
            #     warnings.warn(f"GRB {var.LB} vs self {self.LB}")
            # if not ((var.UB ==  np.inf and self.UB is None) or np.allclose(var.UB, self.ub)):
            #     warnings.warn(f"GRB {var.UB} vs self {self.UB}")
            return var
        else:
            raise NotImplementedError

    def write(self):
        lb, ub = self.lb, self.ub

        if lb == np.inf: lb = "Inf"
        if ub == np.inf: ub = "Inf"
        if lb == -np.inf: lb = "-Inf"
        if ub == -np.inf: ub = "-Inf"

        if lb is None and ub is None:
            return f"{self} free\n"

        elif lb is None:
            return f"{self} <= {ub}\n"

        elif ub is None:
            return f"{self} >= {lb}\n"

        else:
            return f"{lb} <= {self} <= {ub}\n"


    def __str__(self):
        raise NotImplementedError

    def __repr__(self):
        if is_verbose():
            return f"<{type(self).__name__} {str(self)} at {hex(id(self))}>"
        else:
            lb, ub = self.bound
            lb = f"{-np.inf}" if lb is None else f"{lb:.2e}"
            ub = f"{ np.inf}" if ub is None else f"{ub:.2e}"
            return f"{str(self)} in [{lb}, {ub}]"

    # def sum(self):
    #     return sefl

    def overapproximate(self, **kwargs):
        return SymbolicLightningArray(self, self.solver).overapproximate(**kwargs)

class LightningReal(LightningVar):
    def __str__(self):
        return f"C{self.id}"

class LightningBool(LightningVar):
    def __str__(self):
        return f"B{self.id}"


class LightningConstr(str): ...

def _asanyarray(array, dtype=None):
    if isinstance(array, Tensor):
        return array.cpu().detach().numpy()
    else:
        if dtype is not None:
            return np.asanyarray(array, dtype=dtype)
        else:
            return np.asanyarray(array)

def resolve_dims(shape, subscripts, dim_dict=dict()):
    for dim_subscript, dim_size in zip(subscripts, tuple(shape)):
        if dim_subscript in dim_dict:
            assert dim_size == dim_dict[dim_subscript], f"dim {dim_subscript} mismatch"
        else:
            dim_dict[dim_subscript] = dim_size
    return dim_dict

def resolve_leading_broadcast_spec(a_subscripts, leading_subscripts, subscript_dims, return_tail=False):
    i = 0
    broadcast_spec = defaultdict(lambda: [])
    for out_subscript in leading_subscripts:
        if i == len(a_subscripts) or a_subscripts[i] != out_subscript:
            broadcast_spec[i].append(subscript_dims[out_subscript])
        elif i < len(a_subscripts):
            i += 1
        else:
            continue

    if return_tail:
        return broadcast_spec, a_subscripts[i:]
    else:
        return broadcast_spec

def resolve_reorder_axes(x, out):
    lead_axes = []
    for o_subscript in out:
        for x_subscript, axis in zip(x, range(len(x))):
            if x_subscript == o_subscript:
                lead_axes.append(axis)
                break

    tail_axes = [axis for axis in range(len(x)) if axis not in lead_axes]

    return tuple(lead_axes + tail_axes)

def reorder_axes(array, subscripts, out_subscripts, return_new_axes=False):
    assert isinstance(array, np.ndarray)
    new_axes = resolve_reorder_axes(subscripts, out_subscripts)
    if return_new_axes:
        return array.transpose(new_axes), "".join([subscripts[axis] for axis in new_axes]), new_axes
    else:
        return array.transpose(new_axes), "".join([subscripts[axis] for axis in new_axes])

def einsum_path(signature, *arrays, path=None):

    # TODO(anonymous): Performance improvement: reuse the CPU copy of tensor.
    arrays = tuple(
        array.cpu().detach().numpy() if isinstance(array, Tensor) else array
        for array in arrays
    )

    if path is None:

        # TODO(anonymous): Flexibility improvement: support arbitrary arguments.
        subscripts = re.compile(
            "^\s*(?P<a>\w+)\s*[\*,]"
            "\s*(?P<b>\w+)"
            "\s*(\+\s*(?P<c>\w+))?->"
            "\s*(?P<out>\w+)\s*$"
        ).match(signature).groupdict()

        dim_dict = dict()

        a = arrays[0]; resolve_dims(a.shape, subscripts['a'], dim_dict)
        b = arrays[1]; resolve_dims(b.shape, subscripts['b'], dim_dict)
        if subscripts.get('c'):
            c = arrays[2]; resolve_dims(c.shape, subscripts['c'], dim_dict)
        else:
            c = None

        reorder_spec = []
        broadcast_spec = []

        a, subscripts['a'], a_reorder_axes = reorder_axes(a, subscripts['a'], subscripts['out'], return_new_axes=True)
        a_broadcast_spec, a_contraction_subscripts = resolve_leading_broadcast_spec(
            subscripts['a'], subscripts['out'], dim_dict, return_tail=True)
        a = broadcast_at(a, a_broadcast_spec)
        reorder_spec.append(a_reorder_axes)
        broadcast_spec.append(a_broadcast_spec)

        b, subscripts['b'], b_reorder_axes = reorder_axes(b, subscripts['b'], subscripts['out'], return_new_axes=True)
        b_broadcast_spec, b_contraction_subscripts = resolve_leading_broadcast_spec(
            subscripts['b'], subscripts['out'], dim_dict, return_tail=True)
        b = broadcast_at(b, b_broadcast_spec)
        reorder_spec.append(b_reorder_axes)
        broadcast_spec.append(b_broadcast_spec)

        assert a_contraction_subscripts == b_contraction_subscripts, \
            f"mismatch contraction subscripts '{a_contraction_subscripts}' vs "\
            f"'{b_contraction_subscripts}'; reorder contraction dims can be "\
            f"implemented with `reorder_axes(...)`."

        if subscripts.get('c') is not None:
            c, subscripts['c'], c_reorder_axes = reorder_axes(c, subscripts['c'], subscripts['out'], return_new_axes=True)
            c_broadcast_spec, c_contraction_subscripts = resolve_leading_broadcast_spec(
                subscripts['c'], subscripts['out'], dim_dict, return_tail=True)
            c = broadcast_at(c, c_broadcast_spec)
            reorder_spec.append(c_reorder_axes)
            broadcast_spec.append(c_broadcast_spec)
            assert len(c_contraction_subscripts) == 0, \
                f"expected constant to not have contraction dimensions, got "\
                f"{c_contraction_subscripts}."

        # contraction_shape = tuple(dim_dict[s] for s in b_contraction_subscripts)
        # broadcast_shape = tuple(dim_dict[subscript] for subscript in subscripts['out'])

        contraction_ndim = len(a_contraction_subscripts)

        if subscripts.get('c') is not None:
            return (a, b, c), (reorder_spec, broadcast_spec, contraction_ndim)
        else:
            return (a, b), (reorder_spec, broadcast_spec, contraction_ndim)

    else:
        reorder_spec, broadcast_spec, contraction_ndim = path
        assert len(arrays) == len(reorder_spec) and \
               len(arrays) == len(broadcast_spec)
        return (
            broadcast_at(array.transpose(axes), spec)
            for array, axes, spec in zip(arrays, reorder_spec, broadcast_spec)
        ), path

def _einsum_kernel(out, a, b, c=None):
    # NOTE(anonymous): For now I just implemented the most convenient one instead of
    # all fast paths as the POPL version.

    # out = a * b + c
    lhs = [f"- {out}"]
    rhs = 0.
    for ai, bi in zip(np.asanyarray(a).flat, np.asanyarray(b).flat):
        if isinstance(ai, LightningVar):
            if bi != 0.:
                lhs.append(f"{bi} {ai}")

        elif isinstance(bi, LightningVar):
            if ai != 0:
                lhs.append(f"{ai} {bi}")

        else:
            rhs -= ai * bi

    if c is not None:
        if isinstance(c, LightningVar):
            # a * b + c - out = 0.
            lhs.append(f"{c}")

        else:
            # a * b - out = -c
            rhs -= c

    return " + ".join(lhs) + f" = {rhs}\n"

def einsum(signature, *arrays, path=None,
           solver=None, **kwargs):

    # progress = vtqdm(desc='resolving', total=1)
    with _timeit_debug("einsum: resolving"):
        arrays, path = einsum_path(signature, *arrays, path=path)
        ref_shape = arrays[0].shape
        contraction_ndim = path[-1]
        broadcase_ndim = len(ref_shape) - contraction_ndim
        assert broadcase_ndim >= 0
        broadcast_shape = arrays[0].shape[:broadcase_ndim]
        contraction_shape = arrays[0].shape[broadcase_ndim:]

        # executor = get_executor()
        solver   = _get_solver(*arrays, solver=solver)

        if is_debugging():
            assert all(solver == getattr(array, 'solver', None) or solver for array in arrays)

    # progress(desc='create output')
    with _timeit_debug("einsum: create output"):
        # TODO(anonymous): Performance improvement: vectorize or xarange or dummy array.
        # print(broadcast_shape)
        out = solver.reals(broadcast_shape)

    # progress(desc='dispatching')
    with _timeit_debug("einsum: dispatching"):

        out_flat, *arrays = flatten(out, *arrays, start_dim=0, end_dim=len(broadcast_shape)-1)

        total = np.prod(broadcast_shape)
        num_chunks = get_max_workers() * get_dispatch_multiplier()
        chunksize = (total // num_chunks) + 1

        # global einsum_kernel_cow_closure
        def einsum_kernel_cow_closure(start):
            end = min(start+chunksize, total)

            # return end

            # """ Directly return result -- might be slower due to serialization!
            return tuple(
                _einsum_kernel(out_flat[idx], *(array[idx] for array in arrays))
                for idx in range(start, end)
            )
            # return "".join(tuple(
            #     _einsum_kernel(out_flat[idx], *(array[idx] for array in arrays))
            #     for idx in range(start, end)
            # ))
            # """

            # """ Multiprocess write version. Faster.
            filename = f".sytorch_tmp/tmp_einsum_{start}_{end}_{multiprocessing.current_process().name}_{_get_timestamp()}.lp"
            with open(filename, 'w+') as f:
                # f.write("Subject To\n")
                f.write(
                    "".join(tuple(
                        _einsum_kernel(out_flat[idx], *(array[idx] for array in arrays))
                        for idx in range(start, end)
                    ))
                )
                # f.writelines(
                #     _einsum_kernel(out_flat[idx], *(array[idx] for array in arrays))
                #     for idx in range(start, end)
                # )
                return filename
            # """

        # TODO(anonymous): Performance improvement: lower priority of non-blocking tasks for implicit constraints.

        # NOTE(anonymous): Don't forget to fork the process with new local closures to
        # utilize Linux's copy-on-write every time!
        greg = GlobalRegister(globals(), einsum_kernel_cow_closure)
        greg.register()
        executor = ProcessPoolExecutor(get_max_workers())
        constrs_future = executor.map(
            einsum_kernel_cow_closure,
            range(0, total, chunksize),
        )

        # solver._constr_file_futures.append((constrs_future, executor, greg))
        solver._constraint_futures.append((constrs_future, executor, greg))
        # _ = solver.constraints

    return out

def einsum_kernel(out1d, a1d, b1d, c1d=None):
    if c1d is None:
        return [
            _einsum_kernel(out, a, b)
            for out, a, b in zip(out1d.flat, a1d.flat, b1d.flat)
        ]

    else:
        return [
            _einsum_kernel(out, a, b, c)
            for out, a, b, c in zip(out1d.flat, a1d.flat, b1d.flat, c1d.flat)
        ]

@overload
def matmul(a, b, executor=None, max_workers=None): ...
def matmul(a, b, *args, **kwargs):
    a, b = _asanyarray(a), _asanyarray(b)
    a_shape, b_shape = a.shape, b.shape
    assert a_shape[-1] == b_shape[0]
    out = einsum(
        "ij,jk->ik",
        a.reshape(-1, a_shape[-1]),
        b.reshape(b_shape[0], -1),
        *args, **kwargs
    )
    out = out.reshape(*a_shape[:-1], *b_shape[1:])
    return out

@overload
def linear(input, weight, bias, executor=None, max_workers=None): ...
def linear(input, weight, bias, *args, **kwargs):
    if bias is not None:
        return einsum("ni,oi+o->no", input, weight, bias, *args, **kwargs)
    else:
        return einsum("ni,oi->no", input, weight, *args, **kwargs)

@overload
def conv2d(
    input, weight, bias,
    kernel_size,
    padding=(0,0),
    padding_mode='constant',
    stride=(1,1),
    dilation=(1,1),
    ceil_mode=False,
    groups=1,
    executor=None,
    max_workers=None,
): ...
def conv2d(
    input, weight, bias,
    kernel_size,
    padding=(0,0),
    padding_mode='constant',
    stride=(1,1),
    dilation=(1,1),
    ceil_mode=False,
    groups=1,
    **kwargs
):
    assert groups == 1
    if bias is None:
        return einsum(
            "niwhjk,oiwh->nojk",
            _as_strided_window_views2d(
                input,
                kernel_size  = kernel_size,
                padding      = padding,
                padding_mode = padding_mode,
                stride       = stride,
                dilation     = dilation,
                ceil_mode    = ceil_mode
            ),
            weight,
            **kwargs,
        )
    else:
        return einsum(
            "niwhjk,oiwh+o->nojk",
            _as_strided_window_views2d(
                input,
                kernel_size  = kernel_size,
                padding      = padding,
                padding_mode = padding_mode,
                stride       = stride,
                dilation     = dilation,
                ceil_mode    = ceil_mode
            ),
            weight,
            bias,
            **kwargs,
        )

def relu_kernel(a, on):
    if isinstance(a, LightningVar):
        if on:
            return f"{a} >= 0\n"

        else:
            return f"{a} <= 0\n"

    else:
        return ""

def relu(input, pattern, mask=None, **kwargs):
    if mask is not None:
        vectorize(
            relu_kernel,
            signature="(),()->(@side_effect,@constr)",
        )(input[mask], pattern[mask])

        out = input.copy()
        out[~pattern] = 0.
        out[~mask] = F.relu(
            torch.from_numpy(input[~mask].astype(input._concrete_dtype))
        ).numpy()

    else:
        vectorize(
            relu_kernel,
            signature="(),()->(@side_effect,@constr)",
        )(input, pattern)

        out = input.copy()
        out[~pattern] = 0.

    return out

def adaptive_average_pool2d_kernel(out, window):
    lhs = [f"{window.size} {out}"]
    rhs = 0.
    for arg in window.flat:
        if isinstance(arg, LightningVar):
            lhs.append(str(arg))
        else:
            rhs += arg

    return " - ".join(lhs) + f" = {rhs}\n"

def _adaptive_average_pool2d_start_index(out_idx, out_len, in_len):
    """
    https://github.com/pytorch/pytorch/blob/master/aten/src/ATen/native/quantized/cpu/AdaptiveAveragePooling.cpp#L24
    """
    return (int)(np.floor((float)(out_idx * in_len) / out_len))

def _adaptive_average_pool2d_end_index(out_idx, out_len, in_len):
    """
    https://github.com/pytorch/pytorch/blob/master/aten/src/ATen/native/quantized/cpu/AdaptiveAveragePooling.cpp#L37
    """
    return (int)(np.ceil((float)((out_idx + 1) * in_len) / out_len))

def adaptive_average_pool2d(input, output_size, **kwargs):
    N, I =  input.shape[:2]
    O = I
    J, K = output_size

    solver = input.solver

    input_height , input_width  = input.shape[-2:]
    output_height, output_width = output_size # JK
    output = solver.reals((N, O, J, K))

    """ sequential version
    for oh in range(0, output_height):
        ih0 = _adaptive_average_pool2d_start_index(oh, output_height, input_height)
        ih1 = _adaptive_average_pool2d_end_index(oh, output_height, input_height)

        for ow in range(0, output_width):
            iw0 = _adaptive_average_pool2d_start_index(ow, output_width, input_width)
            iw1 = _adaptive_average_pool2d_end_index(ow, output_width, input_width)

            # yield input[...,ih0:ih1,iw0:iw1]

            # output[...,oh,ow] = np.mean(input[...,ih0:ih1,iw0:iw1], axis=(-1,-2))
            for idx in np.ndindex((N, O)):
                output[idx][oh,ow] = input[idx][ih0:ih1,iw0:iw1].mean()
    """

    ndindices = tuple(np.ndindex((N, O, J, K)))
    total = len(ndindices)
    num_chunks = get_max_workers() * get_dispatch_multiplier()
    chunksize = (total // num_chunks) + 1

    def input_window(n, o, oh, ow):
        ih0 = _adaptive_average_pool2d_start_index(oh, output_height, input_height)
        ih1 = _adaptive_average_pool2d_end_index(oh, output_height, input_height)
        iw0 = _adaptive_average_pool2d_start_index(ow, output_width, input_width)
        iw1 = _adaptive_average_pool2d_end_index(ow, output_width, input_width)
        return input[n,o,ih0:ih1,iw0:iw1]

    def average_pool2d_closure(start):
        end = min(start+chunksize, total)
        return tuple(
            adaptive_average_pool2d_kernel(
                output[idx],
                input_window(*idx)
            )
            for idx in ndindices[start:end]
        )


        # """ Multiprocess write version. Faster.
        filename = f".sytorch_tmp/tmp_averagepool_{start}_{end}_{multiprocessing.current_process().name}_{_get_timestamp()}.lp"
        with open(filename, 'w+') as f:
            # f.write("Subject To\n")
            f.write(
                "".join(
                    tuple(
                        adaptive_average_pool2d_kernel(
                            output[idx],
                            input_window(*idx)
                        )
                        for idx in ndindices[start:end]
                    )
                )
            )
            return filename
        # """



    greg = GlobalRegister(globals(), input_window, average_pool2d_closure)
    greg.register()
    executor = ProcessPoolExecutor(get_max_workers())
    future = executor.map(
        average_pool2d_closure,
        range(0, total, chunksize),
    )

    # solver._constr_file_futures.append((future, executor, greg))
    solver._constraint_futures.append((future, executor, greg))
    # _ = solver.constraints

    return output

""" Functional batch_norm_2d """
def batch_norm_2d(
    input,
    mean,
    var,
    gamma,
    beta,
    eps,
    *args, **kwargs
):
    if gamma is not None and beta is not None:
        """ output = (input - mean) * invstd * gamma + beta
            output = input * invstd_gamma - mean_invstd_gamma + beta
            output = input * invstd_gamma + minus_mean_invstd_gamma_plus_beta
        """
        invstd_gamma = (1. / torch.sqrt(var + eps)) * gamma
        constant = (- mean * invstd_gamma) + beta
        return einsum(
            "ncwh,c+c->ncwh",
            input, invstd_gamma, constant,
            *args, **kwargs
        )

    else:
        """ output = (input - mean) * invstd
            output = input * invstd - mean_invstd
        """
        assert gamma is None and beta is None
        invstd = 1. / torch.sqrt(var + eps)
        constant = - mean * invstd
        return einsum(
            "ncwh,c+c->ncwh",
            input, invstd, constant,
            *args, **kwargs
        )

def _tile_array(shape, *arrays):

    if len(arrays) == 1:
        shape = tuple(shape)
        array = arrays[0]

        if array.shape == shape:
            return array

        elif array.size == 1:
            return broadcast_at(array.reshape(()), 0, shape)

        else:
            raise NotImplementedError(
                f"unimplemented tile array from shape {array.shape} to {shape}."
            )

    else:
        return tuple(_tile_array(shape, array) for array in arrays)

def _get_solver(*arrays, solver=None):
    if solver is not None:
        return solver
    else:
        for array in arrays:
            if hasattr(array, 'solver'):
                return array.solver
        return None

def _get_any_subtype(ty, *args):
    for arg in args:
        if isinstance(arg, ty):
            return type(arg)
    return ty


""" Begin vectorize. """

# See https://docs.scipy.org/doc/numpy/reference/c-api.generalized-ufuncs.html
_DIMENSION_NAME = r'@?\w+[\w\d]*'
_CORE_DIMENSION_LIST = '(?:{0:}(?:,{0:})*)?'.format(_DIMENSION_NAME)
_ARGUMENT = r'\({}\)'.format(_CORE_DIMENSION_LIST)
_ARGUMENT_LIST = '{0:}(?:,{0:})*'.format(_ARGUMENT)
_SIGNATURE = '^{0:}->{0:}$'.format(_ARGUMENT_LIST)

def _parse_gufunc_signature(signature):
    """
    Parse string signatures for a generalized universal function.

    Arguments
    ---------
    signature : string
        Generalized universal function signature, e.g., ``(m,n),(n,p)->(m,p)``
        for ``np.matmul``.

    Returns
    -------
    Tuple of input and output core dimensions parsed from the signature, each
    of the form List[Tuple[str, ...]].
    """
    if not re.match(_SIGNATURE, signature):
        raise ValueError(
            'not a valid gufunc signature: {}'.format(signature))
    return tuple([tuple(re.findall(_DIMENSION_NAME, arg))
                  for arg in re.findall(_ARGUMENT, arg_list)]
                 for arg_list in signature.split('->'))


def _update_dim_sizes(dim_sizes, arg, core_dims):
    """
    Incrementally check and update core dimension sizes for a single argument.

    Arguments
    ---------
    dim_sizes : Dict[str, int]
        Sizes of existing core dimensions. Will be updated in-place.
    arg : ndarray
        Argument to examine.
    core_dims : Tuple[str, ...]
        Core dimensions for this argument.
    """
    if not core_dims:
        return

    arg = np.asanyarray(arg)
    num_core_dims = len(core_dims)
    if arg.ndim < num_core_dims:
        raise ValueError(
            '%d-dimensional argument does not have enough '
            'dimensions for all core dimensions %r'
            % (arg.ndim, core_dims))

    core_shape = arg.shape[-num_core_dims:]
    for dim, size in zip(core_dims, core_shape):
        if dim in dim_sizes:
            if size != dim_sizes[dim]:
                raise ValueError(
                    'inconsistent size for core dimension %r: %r vs %r'
                    % (dim, size, dim_sizes[dim]))
        else:
            dim_sizes[dim] = size


def _parse_input_dimensions(args, input_core_dims):
    """
    Parse broadcast and core dimensions for vectorize with a signature.

    Arguments
    ---------
    args : Tuple[ndarray, ...]
        Tuple of input arguments to examine.
    input_core_dims : List[Tuple[str, ...]]
        List of core dimensions corresponding to each input.

    Returns
    -------
    broadcast_shape : Tuple[int, ...]
        Common shape to broadcast all non-core dimensions to.
    dim_sizes : Dict[str, int]
        Common sizes for named core dimensions.
    """
    broadcast_args = []
    dim_sizes = {}
    for arg, core_dims in zip(args, input_core_dims):
        _update_dim_sizes(dim_sizes, arg, core_dims)
        ndim = arg.ndim - len(core_dims)
        dummy_array = np.lib.stride_tricks.as_strided(0, arg.shape[:ndim])
        broadcast_args.append(dummy_array)
    broadcast_shape = np.lib.stride_tricks._broadcast_shape(*broadcast_args)
    return broadcast_shape, dim_sizes


def _calculate_shapes(broadcast_shape, dim_sizes, list_of_core_dims):
    """Helper for calculating broadcast shapes with core dimensions."""
    return [broadcast_shape + tuple(dim_sizes[dim] for dim in core_dims)
            for core_dims in list_of_core_dims]


def _create_arrays(broadcast_shape, dim_sizes, list_of_core_dims, dtypes, solver=None):
    """Helper for creating output arrays in vectorize."""
    shapes = _calculate_shapes(broadcast_shape, dim_sizes, list_of_core_dims)
    arrays = tuple(
            np.empty(shape, dtype=object)
                .view(SymbolicLightningArray).to(solver)
                if dtype is LightningVar else

            np.empty(shape, dtype=object)
                .view(SymbolicLightningConstrArray).to(solver)
                if dtype is LightningConstr else

            np.empty(shape, dtype=dtype)

            for shape, dtype in zip(shapes, dtypes))
    return arrays

def _handle_input_specs(dims_or_specs):
    all_dims = []
    all_specs = []
    for x in dims_or_specs:
        dims = []
        spec = defaultdict(lambda: False)
        for annotate in x:
            if annotate[0] != '@':
                dims.append(annotate)

            elif annotate == '@implicit':
                spec['implicit'] = True

            elif annotate == '@real':
                spec['vtype'] = 'real'

            elif annotate == '@ret':
                spec['ret'] = True

            else:
                raise RuntimeError(
                    f"unknown input spec {annotate}."
                )
        all_dims.append(dims)
        all_specs.append(spec)
    return all_dims, all_specs

def _handle_output_specs(dims_or_specs):
    all_dims = []
    all_specs = []
    for x in dims_or_specs:
        spec = defaultdict(lambda: False)
        spec['ret'] = True
        dims = []
        for annotate in x:
            if annotate[0] != '@':
                dims.append(annotate)

            elif annotate == '@side_effect':
                spec['ret'] = False
                spec['side_effect'] = True

            elif annotate == '@constr':
                spec['vtype'] = 'constr'
                # # FOR NOW
                # spec['ret'] = False
                # spec['side_effect'] = True

            else:
                raise RuntimeError(
                    f"unknown output spec {annotate}."
                )
        all_dims.append(dims)
        all_specs.append(spec)
    return all_dims, all_specs

class vectorize(object):
    """
    vectorize(pyfunc, otypes=None, doc=None, excluded=None, cache=False,
              signature=None)

    Generalized function class.

    Define a vectorized function which takes a nested sequence of objects or
    numpy arrays as inputs and returns a single numpy array or a tuple of numpy
    arrays. The vectorized function evaluates `pyfunc` over successive tuples
    of the input arrays like the python map function, except it uses the
    broadcasting rules of numpy.

    The data type of the output of `vectorized` is determined by calling
    the function with the first element of the input.  This can be avoided
    by specifying the `otypes` argument.

    Parameters
    ----------
    pyfunc : callable
        A python function or method.
    otypes : str or list of dtypes, optional
        The output data type. It must be specified as either a string of
        typecode characters or a list of data type specifiers. There should
        be one data type specifier for each output.

        NOTE(anonymous): Additional otypes we support:
        LightningVar: will return a SymbolicLightningArray
        LightningConstr: will return a SymbolicLightningConstrArray

    doc : str, optional
        The docstring for the function. If `None`, the docstring will be the
        ``pyfunc.__doc__``.
    excluded : set, optional
        Set of strings or integers representing the positional or keyword
        arguments for which the function will not be vectorized.  These will be
        passed directly to `pyfunc` unmodified.

        .. versionadded:: 1.7.0

    cache : bool, optional
       If `True`, then cache the first function call that determines the number
       of outputs if `otypes` is not provided.

        .. versionadded:: 1.7.0

    signature : string, optional
        Generalized universal function signature, e.g., ``(m,n),(n)->(m)`` for
        vectorized matrix-vector multiplication. If provided, ``pyfunc`` will
        be called with (and expected to return) arrays with shapes given by the
        size of corresponding core dimensions. By default, ``pyfunc`` is
        assumed to take scalars as input and output.

        .. versionadded:: 1.12.0

        NOTE(anonymous): Additional annotations we support:

        Annotations on inputs:
            @implicit: vectorize call will infer the shape and create this array
                       implicitly.
            @real    : the type of implicit input.
            @ret     : will append this input to the tuple of final outputs

        Annotations on outputs:
            @side_effect: this output will be handled by `handle_constrs` and
                          will not be returned.
            @constr

    Returns
    -------
    vectorized : callable
        Vectorized function.

    See Also
    --------
    frompyfunc : Takes an arbitrary Python function and returns a ufunc

    Notes
    -----
    The `vectorize` function is provided primarily for convenience, not for
    performance. The implementation is essentially a for loop.

    If `otypes` is not specified, then a call to the function with the
    first argument will be used to determine the number of outputs.  The
    results of this call will be cached if `cache` is `True` to prevent
    calling the function twice.  However, to implement the cache, the
    original function must be wrapped which will slow down subsequent
    calls, so only do this if your function is expensive.

    The new keyword argument interface and `excluded` argument support
    further degrades performance.

    References
    ----------
    .. [1] NumPy Reference, section `Generalized Universal Function API
           <https://docs.scipy.org/doc/numpy/reference/c-api.generalized-ufuncs.html>`_.

    Examples
    --------
    """
    def __init__(self, pyfunc, otypes=None, doc=None, excluded=None,
                 cache=False, signature=None, executor=None, max_workers=None,
                 solver=None):
        self.pyfunc = pyfunc
        self.cache = cache
        self.signature = signature
        self._ufunc = None    # Caching to improve default performance
        self.executor = executor
        self.max_workers = max_workers
        self.solver = solver

        if doc is None:
            self.__doc__ = pyfunc.__doc__
        else:
            self.__doc__ = doc

        if isinstance(otypes, str):
            for char in otypes:
                if char not in np.typecodes['All']:
                    raise ValueError("Invalid otype specified: %s" % (char,))
        elif np.iterable(otypes):
            otypes = [
                otype if otype in (LightningVar, LightningConstr) else
                np.core.numeric.dtype(otype).char
                for otype in otypes
            ]
        elif otypes is not None:
            raise ValueError("Invalid otype specification")
        self.otypes = otypes

        # Excluded variable support
        if excluded is not None:
            raise NotImplementedError(
                f"lightning.vectorize doesn't support the excluded parameters yet."
            )
        # if excluded is None:
        #     excluded = set()
        # self.excluded = set(excluded)

        if signature is not None:
            self.in_core_dims, self.out_core_dims = _parse_gufunc_signature(signature)
            self.in_core_dims, self.in_specs = _handle_input_specs(self.in_core_dims)
            self.out_core_dims, self.out_specs = _handle_output_specs(self.out_core_dims)
        else:
            self.in_core_dims, self.out_core_dims = None, None

    def __call__(self, *args, **kwargs):
        """
        Return arrays with the results of `pyfunc` broadcast (vectorized) over
        `args` and `kwargs` not in `excluded`.
        """
        func = self.pyfunc
        vargs = args
        # excluded = self.excluded
        # if not kwargs and not excluded:
        #     func = self.pyfunc
        #     vargs = args
        # else:
        #     # The wrapper accepts only positional arguments: we use `names` and
        #     # `inds` to mutate `the_args` and `kwargs` to pass to the original
        #     # function.
        #     nargs = len(args)

        #     names = [_n for _n in kwargs if _n not in excluded]
        #     inds = [_i for _i in range(nargs) if _i not in excluded]
        #     the_args = list(args)

        #     def func(*vargs):
        #         for _n, _i in enumerate(inds):
        #             the_args[_i] = vargs[_n]
        #         kwargs.update(zip(names, vargs[len(inds):]))
        #         return self.pyfunc(*the_args, **kwargs)

        #     vargs = [args[_i] for _i in inds]
        #     vargs.extend([kwargs[_n] for _n in names])

        return self._vectorize_call(
            func=func,
            args=vargs,
            executor=kwargs.get('executor') or self.executor,
            solver=kwargs.get('solver') or self.solver,
            max_workers=kwargs.get('max_workers') or self.max_workers
        )

    def _get_ufunc_and_otypes(self, func, args):
        raise NotImplementedError
        """Return (ufunc, otypes)."""
        # frompyfunc will fail if args is empty
        if not args:
            raise ValueError('args can not be empty')

        if self.otypes is not None:
            otypes = self.otypes
            nout = len(otypes)

            # Note logic here: We only *use* self._ufunc if func is self.pyfunc
            # even though we set self._ufunc regardless.
            if func is self.pyfunc and self._ufunc is not None:
                ufunc = self._ufunc
            else:
                ufunc = self._ufunc = np.frompyfunc(func, len(args), nout)
        else:
            # Get number of outputs and output types by calling the function on
            # the first entries of args.  We also cache the result to prevent
            # the subsequent call when the ufunc is evaluated.
            # Assumes that ufunc first evaluates the 0th elements in the input
            # arrays (the input values are not checked to ensure this)
            args = [np.asarray(arg) for arg in args]
            if any(arg.size == 0 for arg in args):
                raise ValueError('cannot call `vectorize` on size 0 inputs '
                                 'unless `otypes` is set')

            inputs = [arg.flat[0] for arg in args]
            outputs = func(*inputs)

            # Performance note: profiling indicates that -- for simple
            # functions at least -- this wrapping can almost double the
            # execution time.
            # Hence we make it optional.
            if self.cache:
                _cache = [outputs]

                def _func(*vargs):
                    if _cache:
                        return _cache.pop()
                    else:
                        return func(*vargs)
            else:
                _func = func

            if isinstance(outputs, tuple):
                nout = len(outputs)
            else:
                nout = 1
                outputs = (outputs,)

            otypes = ''.join([np.asarray(outputs[_k]).dtype.char
                              for _k in range(nout)])

            # Performance note: profiling indicates that creating the ufunc is
            # not a significant cost compared with wrapping so it seems not
            # worth trying to cache this.
            ufunc = np.frompyfunc(_func, len(args), nout)

        return ufunc, otypes

    def _vectorize_call(self, func, args, executor=None, max_workers=None, solver=None):
        """Vectorized call to `func` over positional `args`."""
        if self.signature is not None:
            res = self._vectorize_call_with_signature(
                func, args, executor=executor, max_workers=max_workers, solver=solver)
        elif not args:
            res = func()
        else:
            raise NotImplementedError(
                f"lightning.vectorize with no signature is WIP."
            )
            ufunc, otypes = self._get_ufunc_and_otypes(func=func, args=args)

            # Convert args to object arrays first
            inputs = [array(a, copy=False, subok=True, dtype=object)
                      for a in args]

            outputs = ufunc(*inputs)

            if ufunc.nout == 1:
                res = array(outputs, copy=False, subok=True, dtype=otypes[0])
            else:
                res = tuple([array(x, copy=False, subok=True, dtype=t)
                             for x, t in zip(outputs, otypes)])
        return res

    def _vectorize_call_with_signature(
        self, func, explicit_args,
        executor=None, max_workers=None, solver=None
    ):

        # executor = get_executor(executor, max_workers)
        solver = _get_solver(*explicit_args, solver=solver)

        """Vectorized call over positional arguments with a signature."""
        input_core_dims, output_core_dims = self.in_core_dims, self.out_core_dims
        input_specs, output_specs = self.in_specs, self.out_specs

        # Calculate broadcast_shape, dim_sizes from explicit args.
        explicit_arg_core_dims = tuple(
            core_dim
            for core_dim, spec in zip(input_core_dims, input_specs)
            if not spec['implicit']
        )

        if len(explicit_args) != len(explicit_arg_core_dims):
            # for a in explicit_args:
            #     print(type(a))
            raise TypeError('wrong number of explicit positional arguments: '
                            'expected %r, got %r'
                            % (len(explicit_arg_core_dims), len(explicit_args)))

        explicit_args = tuple(_asanyarray(a) for a in explicit_args)
        assert(type(explicit_args) == tuple)

        broadcast_shape, dim_sizes = _parse_input_dimensions(
            explicit_args, explicit_arg_core_dims)

        # Handle scalar output.
        is_scalar=False
        if len(broadcast_shape) == 0:
            broadcast_shape = (1,)
            is_scalar=True

        # Calculate input shapes for all args.
        input_shapes = _calculate_shapes(broadcast_shape, dim_sizes,
                                         input_core_dims)

        # Create implicit args.
        explicit_args = list(explicit_args)
        args = []
        for shape, spec in zip(input_shapes, input_specs):
            if spec['implicit']:
                assert spec['vtype'] == 'real'
                args.append(solver.reals(shape))
            else:
                args.append(explicit_args.pop(0))
        args = tuple(args)

        # Tile all args to valid broadcast shape.
        args_broadcasted = [np.broadcast_to(arg, shape, subok=True)
                            for arg, shape in zip(args, input_shapes)]

        raw_outputs = None
        otypes = self.otypes
        n_raw_out = len(output_core_dims)

        args_flat = tuple(flatten(a, start_dim=0, end_dim=len(broadcast_shape)-1) for a in args_broadcasted)

        total = np.prod(broadcast_shape)
        num_chunks = get_max_workers() * get_dispatch_multiplier()
        chunksize = (total // num_chunks) + 1

        # Create a local closure to leverage Linux's copy-on-write fork.
        def func_closure(start):
            end = min(start+chunksize, total)
            return tuple(
                func(*(arg[idx] for arg in args_flat))
                for idx in range(start, end)
            )

            # """ Multiprocess write version. Faster.
            filename = f".sytorch_tmp/tmp_vectorize_{start}_{end}_{multiprocessing.current_process().name}_{_get_timestamp()}.lp"
            with open(filename, 'w+') as f:
                # f.write("Subject To\n")
                f.write(
                    "".join(
                        tuple(
                            func(*(arg[idx] for arg in args_flat))
                            for idx in range(start, end)
                        )
                    )
                )
            # assert pathlib.Path(filename).exists()
            return filename
            # """

        # Register the possible lambda function `func` and the local closures
        # `func_closure` to global with a new unique name.
        greg = GlobalRegister(globals(), func, func_closure)
        greg.register()
        executor = ProcessPoolExecutor(get_max_workers())
        results_futures = executor.map(
            func_closure,
            range(0, total, chunksize),
        )
        # executor.shutdown(wait=False)

        outputs = []
        if n_raw_out == 1 and output_specs[0]['side_effect'] is True:
            # solver._constr_file_futures.append((results_futures, executor, greg))
            solver._constraint_futures.append((results_futures, executor, greg))
            # _ = solver.constraints
        else:
            # results_futures = tuple(results_futures)
            for index, results in zip(np.ndindex(*broadcast_shape), itertools.chain(*results_futures)):
                n_results = len(results) if isinstance(results, tuple) else 1

                if n_raw_out != n_results:
                    raise ValueError(
                        'wrong number of raw outputs from pyfunc: expected %r, got %r'
                        % (n_raw_out, n_results))

                if n_raw_out == 1:
                    results = (results,)

                if raw_outputs is None:
                    for result, core_dims in zip(results, output_core_dims):
                        _update_dim_sizes(dim_sizes, result, core_dims)

                    if otypes is None:
                        otypes = [
                            LightningConstr if spec['vtype'] == 'constr' else
                            LightningVar if spec['vtype'] == 'real' else
                            LightningConstr if isinstance(result, (LightningConstr, SymbolicLightningConstrArray)) else
                            LightningVar if isinstance(result, (LightningVar, SymbolicLightningArray)) else
                            np.asarray(result).dtype
                            for result, spec in zip(results, output_specs)
                        ]

                    raw_outputs = _create_arrays(broadcast_shape, dim_sizes,
                                            output_core_dims, otypes, solver=solver)

                for output, result in zip(raw_outputs, results):
                    output[index] = result

            for output, spec in zip(raw_outputs, output_specs):
                if spec['side_effect']:
                    # didn't reshape back to scalar here.
                    handle_constrs(output)
                else:
                    if is_scalar:
                        output = output.reshape(())
                    outputs.append(output)

        for arg, spec in zip(args, input_specs):
            if spec['ret']:
                if is_scalar:
                    arg = arg.reshape(())
                outputs.append(arg)

        outputs = tuple(outputs)

        if outputs is None:
            raise NotImplementedError
            # did not call the function even once
            if otypes is None:
                raise ValueError('cannot call `vectorize` on size 0 inputs '
                                 'unless `otypes` is set')
            if any(dim not in dim_sizes
                            for dims in output_core_dims
                            for dim in dims):
                raise ValueError('cannot call `vectorize` with a signature '
                                 'including new output dimensions on size 0 '
                                 'inputs')
            outputs = _create_arrays(broadcast_shape, dim_sizes,
                                     output_core_dims, otypes)

        return outputs[0] if len(outputs) == 1 else outputs

""" End vectorize. """

def _resolve_array_reduce_at_axis(array, axis):
    if axis is None:
        array = array.reshape(-1)
        axis = 0

    if axis is not None:
        axis = (array.ndim + axis) % array.ndim

    out_shape = list(array.shape)
    out_shape[axis] = 1

    return array, axis, out_shape

@overload
def apply_along_axis(
    pyfunc, axis, signature, array, doc=None, excluded=None,
    cache=False, executor=None,
): ...
def apply_along_axis(
    pyfunc, axis, signature, array, **kwargs
):
    array, axis, _ = _resolve_array_reduce_at_axis(array, axis)
    array = np.moveaxis(array, axis, -1)
    return vectorize(
        pyfunc,
        signature=signature,
        **kwargs,
    )(array)

def add_kernel(out, a, b):
    if not isinstance(a, LightningVar):
        if not isinstance(b, LightningVar):
            return f"{out} = {a + b}\n"
        else:
            return f"{b} - {out} = {-a}\n"

    elif not isinstance(b, LightningVar):
        return f"{a} - {out} = {-b}\n"

    else:
        return f"{a} + {b} - {out} = 0\n"

add = vectorize(
    add_kernel,
    signature="(@implicit,@real,@ret),(),()->(@side_effect,@constr)"
)

def subtract_kernel(out, a, b):
    if not isinstance(a, LightningVar):
        if not isinstance(b, LightningVar):
            return f"{out} = {a - b}\n"
        else:
            return f"{b} + {out} = {a}\n"

    elif not isinstance(b, LightningVar):
        return f"{a} - {out} = {b}\n"

    else:
        return f"{a} - {b} - {out} = 0\n"

subtract = vectorize(
    subtract_kernel,
    signature="(@implicit,@real,@ret),(),()->(@side_effect,@constr)"
)

def divide_kernel(out, a, b):
    if not isinstance(a, LightningVar):
        return f"{b} {out} = {a}\n"

    else:
        return f"{b} {out} - {a} = 0\n"

divide = vectorize(
    divide_kernel,
    signature="(@implicit,@real,@ret),(),()->(@side_effect,@constr)"
)

def multiply_kernel(out, a, b):
    # a * b - out = 0
    if not isinstance(a, LightningVar):
        if not isinstance(b, LightningVar):
            return f"{out} = {a * b}\n"
        else:
            return f"{a} {b} - {out} = 0\n"

    elif not isinstance(b, LightningVar):
            return f"{b} {a} - {out} = 0\n"

    else:
        return f"{a} {b} - {out} = 0\n"

multiply = vectorize(
    multiply_kernel,
    signature="(@implicit,@real,@ret),(),()->(@side_effect,@constr)"
)

def eq_kernel(a, b):
    if not isinstance(a, LightningVar):
        return f"{b} = {a}\n"

    elif not isinstance(b, LightningVar):
        return f"{a} = {b}\n"

    else:
        return f"{a} - {b} = 0\n"

eq = vectorize(
    eq_kernel,
    signature="(),()->(@constr)",
)

eq_nowait = vectorize(
    eq_kernel,
    signature="(),()->(@side_effect,@constr)",
)

# TODO(anonymous): check gurobi when lhs is 0.
def le_kernel(a, b):
    if not isinstance(a, LightningVar):
        if not isinstance(b, LightningVar):
            return f"0 >= {a - b}\n"
        else:
            return f"{b} >= {a}\n"

    elif not isinstance(b, LightningVar):
        return f"{a} <= {b}\n"

    return f"{a} - {b} <= 0\n"

le = vectorize(
    le_kernel,
    signature="(),()->(@constr)",
)

le_nowait = vectorize(
    le_kernel,
    signature="(),()->(@side_effect,@constr)",
)

def ge_kernel(a, b):
    if not isinstance(a, LightningVar):
        if not isinstance(b, LightningVar):
            return f"0 <= {a - b}\n"
        else:
            return f"{b} <= {a}\n"

    elif not isinstance(b, LightningVar):
        return f"{a} >= {b}\n"

    return f"{a} - {b} >= 0\n"

ge = vectorize(
    ge_kernel,
    signature="(),()->(@constr)",
)

ge_nowait = vectorize(
    ge_kernel,
    signature="(),()->(@side_effect,@constr)",
)

def gt_kernel(a, b, eps):
    # a >= b + eps
    if not isinstance(a, LightningVar):
        if not isinstance(b, LightningVar):
            # a - b - eps >= 0
            return f"0 <= {a - b - eps}\n"
        else:
            # a - eps >= b
            return f"{b} <= {a - eps}\n"

    elif not isinstance(b, LightningVar):
        return f"{a} >= {b + eps}\n"

    # a - b >= eps
    return f"{a} - {b} >= {eps}\n"

def gt(a, b, eps=None, **kwargs):
    eps = eps or get_epsilon()
    return vectorize(
        gt_kernel,
        signature="(),(),()->(@constr)",
    )(a, b, eps, **kwargs)

def gt_nowait(a, b, eps=None, **kwargs):
    eps = eps or get_epsilon()
    return vectorize(
        gt_kernel,
        signature="(),(),()->(@side_effect,@constr)",
    )(a, b, eps, **kwargs)

def lt(a, b, eps=None, **kwargs):
    # a <= b - eps
    # a + eps <= b
    # b >= a + eps
    return gt(b, a, eps=eps, **kwargs)

def lt_nowait(a, b, eps=None, **kwargs):
    return gt_nowait(b, a, eps=eps, **kwargs)

""" Absolute upper bound."""
def abs_ub_kernel(a_abs_ub, a):
    if isinstance(a, LightningVar):
        return f"{a} - {a_abs_ub} <= 0\n- {a} - {a_abs_ub} <= 0\n"
    else:
        return f"{a_abs_ub} >= {a}\n{a_abs_ub} >= {-a}\n"

# NOTE(anonymous): Enhancement: mask non-symbolic elements.
abs_ub = vectorize(
    abs_ub_kernel,
    signature="(@implicit,@real,@ret),()->(@side_effect,@constr)",
)

""" Maximum upper bound."""
def max_ub(array, axis=None, **kwargs):
    array, axis, out_shape = _resolve_array_reduce_at_axis(array, axis)
    out = array.solver.reals(out_shape)
    handle_constrs(out >= array)
    return out.squeeze(axis)

""" Summation. """
def sum_kernel(out, a):
    lhs = [f"{out}"]
    rhs = 0.
    for arg in a.flat:
        if isinstance(arg, LightningVar):
            lhs.append(str(arg))
        else:
            rhs += arg
    return " - ".join(lhs) + f" = {rhs}\n"

def sum(array, axis=None, **kwargs):
    return apply_along_axis(
        sum_kernel,
        signature = "(@implicit,@real,@ret),(n)->(@side_effect,@constr)",
        axis = axis,
        array = array,
        **kwargs
    )

""" Mean value. """
def mean_kernel(out, a):
    lhs = [f"{a.size} {out}"]
    rhs = 0.
    for arg in a.flat:
        if isinstance(arg, LightningVar):
            lhs.append(str(arg))
        else:
            rhs += arg
    return " - ".join(lhs) + f" = {rhs}\n"

def mean(array, axis=None, **kwargs):
    return apply_along_axis(
        mean_kernel,
        signature = "(@implicit,@real,@ret),(n)->(@side_effect,@constr)",
        axis = axis,
        array = array,
        **kwargs
    )

def _to_symbolic_lightning_array(*arrays, solver):
    return tuple(
        array.to(solver)             if isinstance(array, SymbolicLightningArray) else
        array.cpu().detach().numpy().view(SymbolicLightningArray).to(solver)
                                     if isinstance(array, Tensor) else
        np.asanyarray(array).view(SymbolicLightningArray).to(solver)
        for array in arrays
    )

def concatenate(arrays, axis=None):
    solver = _get_solver(*arrays)
    arrays = _to_symbolic_lightning_array(*arrays, solver=solver)
    assert solver is not None
    ty = _get_any_subtype(SymbolicLightningArray, *arrays)
    SymbolicLightningArray._check_is_all_compatible(arrays)
    return np.concatenate(arrays, axis=axis).view(ty).to(solver)

def stack(arrays, axis=None):
    solver = _get_solver(*arrays)
    arrays = _to_symbolic_lightning_array(*arrays, solver=solver)
    assert solver is not None
    ty = _get_any_subtype(SymbolicLightningArray, *arrays)
    SymbolicLightningArray._check_is_all_compatible(arrays)
    return np.stack(arrays, axis=axis).view(ty).to(solver)

def split_along_axis(array, indices, axis, broadcast=True):
    axis = (axis + array.ndim) % array.ndim

    indices = _asanyarray(indices, dtype=int)
    if indices.size == 1:
        indices = _tile_array(array.shape[:axis] + (1,) + array.shape[axis+1:], indices)

    assert array.shape[:axis] == indices.shape[:axis]
    assert array.shape[axis+1:] == indices.shape[axis+1:]
    assert indices.shape[axis] == 1

    array = np.moveaxis(array, axis, -1)
    indices = np.moveaxis(indices, axis, -1)

    array_other = np.empty(array.shape[:-1]+(array.shape[-1]-1,), dtype=object)
    for row in np.ndindex(array.shape[:-1]):
        argmax_index = indices[row][0]
        array_other[row][:argmax_index] = array[row][:argmax_index]
        array_other[row][argmax_index:] = array[row][argmax_index+1:]

    array_indexed = np.take_along_axis(array, indices, axis=-1)
    array_indexed = broadcast_at(array_indexed[...,0], array_indexed.ndim-1, array_other.shape[-1:])

    array_indexed = np.moveaxis(array_indexed, -1, axis)
    array_other = np.moveaxis(array_other, -1, axis)

    return array_indexed, array_other

def argmax(array, indices, axis, **kwargs):
    array_max, array_other = split_along_axis(array, indices, axis=axis)
    array_max = array_max.view(type(array)).to(array.solver)
    array_other = array_other.view(type(array)).to(array.solver)
    return gt(array_max, array_other, **kwargs)

class LightningArgMaxEncoder:
    def __init__(self, array, axis, **kwargs):
        self.array = array
        self.axis = axis
        self.kwargs = kwargs

    def __eq__(self, indices):
        return argmax(self.array, indices, axis=self.axis, **self.kwargs)

def argmin(array, indices, axis, **kwargs):
    array_min, array_other = split_along_axis(array, indices, axis=axis)
    array_min = array_min.view(type(array)).to(array.solver)
    array_other = array_other.view(type(array)).to(array.solver)
    return lt(array_min, array_other, **kwargs)

class LightningArgMinEncoder:
    def __init__(self, array, axis, **kwargs):
        self.array = array
        self.axis = axis
        self.kwargs = kwargs

    def __eq__(self, indices):
        return argmin(self.array, indices, axis=self.axis, **self.kwargs)

""" SymbolicLightningArray """
class SymbolicLightningConstrArray(SymbolicArray): ...

class SymbolicLightningArray(SymbolicArray):
    def to(self, solver):
        if isinstance(solver, GurobiSolver):
            out = SymbolicLPArray(
                np.vectorize(
                    lambda name: \
                        name.to(solver)
                        if isinstance(name, LightningVar)
                        else name,
                    signature="()->()",
                    otypes=[object]
                )(self),
                solver = solver
            )

        elif isinstance(solver, LightningSolver):
            if self.solver is None:
                self.solver = solver
            elif self.solver is solver:
                pass
            else:
                raise NotImplementedError(
                    "unimplemented translation between LightningSolver."
                )
            out = self

        elif self.solver == None and solver == None:
            out = self

        else:
            raise NotImplementedError

        return out

    def ok(self) -> bool:
        return True

    # @property
    # def mask(self):
    #     return np.vectorize(
    #         lambda v: isinstance(v, str),
    #         signature="()->()",
    #         otypes=[bool]
    #     )(self)

    def alias(self):
        return self

    """ operators """
    def __eq__(self, other):
        self._check_is_compatible_with(other)
        return eq(self, other)

    def __ne__(self, other):
        raise NotImplementedError(
            f"inequality is MIP."
        )

    def __ge__(self, other):
        self._check_is_compatible_with(other)
        return ge(self, other)

    def __le__(self, other):
        self._check_is_compatible_with(other)
        return le(self, other)

    def __gt__(self, other):
        self._check_is_compatible_with(other)
        return gt(self, other)

    def __lt__(self, other):
        self._check_is_compatible_with(other)
        return lt(self, other)

    def __add__(self, other):
        self._check_is_compatible_with(other)
        return add(self, other)

    def __radd__(self, other):
        self._check_is_compatible_with(other)
        return add(other, self)

    def __sub__(self, other):
        self._check_is_compatible_with(other)
        return subtract(self, other)

    def __rsub__(self, other):
        self._check_is_compatible_with(other)
        return subtract(other, self)

    def __mul__(self, other):
        self._check_is_compatible_with(other)
        return multiply(self, other)

    def __rmul__(self, other):
        self._check_is_compatible_with(other)
        return multiply(other, self)

    def __truediv__(self, other):
        self._check_is_compatible_with(other)
        return divide(self, other)

    def __rtruediv__(self, other):
        raise NotImplementedError("quadratic.")
        self._check_is_compatible_with(other)
        return divide(other, self)

    def __matmul__(self, other):
        self._check_is_compatible_with(other)
        return matmul(self, other)

    def __rmatmul__(self, other):
        self._check_is_compatible_with(other)
        return matmul(other, self)

    def abs_ub(self, *args, **kwargs):
        return abs_ub(self, *args, **kwargs)

    def max_ub(self, *args, **kwargs):
        return max_ub(self, *args, **kwargs)

    def sum(self, axis=None, *args, **kwargs):
        return sum(self, axis=axis, *args, **kwargs)

    def mean(self, axis=None, *args, **kwargs):
        return mean(self, axis=axis, *args, **kwargs)

    def norm_ub(self, order: Literal['l1', 'l1_normalized', 'linf', 'linf+l1_normalized']='l1', *args, **kwargs) -> SymbolicLPArray:
        """Returns the norm specified by order. The l1_normalized specification
        returns the l1 norm divided by the number of variables in the SymbolicArray.
        """
        assert self.ndim == 1
        # print(f"taking {order} norm over {self.size} variables.")
        if order == 'linf+l1_normalized':
            abs_ub = self.abs_ub(*args, **kwargs)
            return (abs_ub.sum(*args, **kwargs) / self.size) + abs_ub.max_ub(*args, **kwargs)

        elif order == 'l1':
            return self.abs_ub(*args, **kwargs).sum(*args, **kwargs)

        elif order == 'l1_normalized':
            # TODO(anonymous): there is a fast path.
            return self.norm_ub('l1', *args, **kwargs) / self.size

        elif order == 'linf':
            return self.abs_ub(*args, **kwargs).max_ub(*args, **kwargs)

        raise NotImplementedError(
            f"unsupported norm order {order} for {type(self)}, perhaps try "
            f"`.milp().norm(...)`."
        )

    def argmax(self, axis, *args, **kwargs):
        return LightningArgMaxEncoder(self, axis=axis, *args, **kwargs)

    def argmin(self, axis, *args, **kwargs):
        return LightningArgMinEncoder(self, axis=axis, *args, **kwargs)

    def evaluate(self):
        return self.solver.evaluate(self)

    def nowait(self):
        return self.view(SymbolicLightningArrayNoWait)

class SymbolicLightningArrayNoWait(SymbolicLightningArray):
    def __eq__(self, other):
        self._check_is_compatible_with(other)
        return eq_nowait(self, other)

    def __ne__(self, other):
        raise NotImplementedError(
            f"inequality is MIP."
        )

    def __ge__(self, other):
        self._check_is_compatible_with(other)
        return ge_nowait(self, other)

    def __le__(self, other):
        self._check_is_compatible_with(other)
        return le_nowait(self, other)

    def __gt__(self, other):
        self._check_is_compatible_with(other)
        return gt_nowait(self, other)

    def __lt__(self, other):
        self._check_is_compatible_with(other)
        return lt_nowait(self, other)

    def wait(self):
        return self.view(SymbolicLightningArray)

""" Lightning Solver. """
class LightningSolver(Solver):
    def __init__(self):

        # pathlib.Path('.sytorch_tmp').mkdir(parents=True, exist_ok=True)

        self._verbose = True
        self.solver = None
        self.optimization_mode = None

        self._real_var_counter = _Counter()
        self.lb = defaultdict(lambda: None)
        self.ub = defaultdict(lambda: None)

        self._constraints = []
        self._constraint_futures = []
        # self._constr_files = []
        # self._constr_file_futures = []

        self._objective = None
        super().__init__()

    def verbose_(self, mode=True):
        self._verbose = mode
        return self

    # @property
    # def constr_files(self):
    #     progress = tqdm(desc="waiting constr file futures", total=len(self._constr_file_futures), leave=True)
    #     progress.display()
    #     new_files = []
    #     while len(self._constr_file_futures) > 0:
    #         future, executor, greg = self._constr_file_futures.pop()
    #         for tmp in tqdm(future, desc="waiting future...", leave=False):
    #             # self._constr_files.append(tmp)
    #             new_files.append(tmp)
    #             yield tmp
    #         executor.shutdown(wait=True)
    #         greg.unregister()
    #         progress.update(1)

    #     for tmp in self._constr_files:
    #         yield tmp

    #     for tmp in new_files:
    #         self._constr_files.append(tmp)

    @property
    def constraints(self):
        # assert len(self._constraint_futures) == 0
        # assert len(self._constraints) == 0
        # return []
        progress = tqdm(desc="waiting futures", total=len(self._constraint_futures), leave=False)
        progress.display()
        while len(self._constraint_futures) > 0:
            future, executor, greg = self._constraint_futures.pop()
            # warnings.warn('PRDNN consumes all the memory. directly writing to disk.')
            # yield from future
            self.add_constraints(tuple(future))
            # I think aysnc close is better, but there seems to be a bug in
            # python < 3.9: https://bugs.python.org/issue39104
            executor.shutdown(wait=True)
            greg.unregister()
            progress.update(1)
        return self._constraints

    @property
    def objective(self):
        return self._objective

    @objective.setter
    def objective(self, value):
        self._objective = value

    def add_constraints(self, *constraints):
        for c in _iter_nested_iterable(constraints):
            assert isinstance(c, str), f"{type(c)}, {c}"
            self._constraints.append(c)
            # self._constr_files.append(c)

    def minimize(self, objective):
        self.optimization_mode = 'Minimize'
        self.objective = objective

    def maximize(self, objective):
        self.optimization_mode = 'Maximize'
        self.objective = objective

    def solve(self, *constraints, minimize=None, maximize=None, unlink=True, keep_dump=False, Method=2, TimeLimit=None, **kwargs):
        assert minimize is None or maximize is None
        self.add_constraints(*constraints)
        if minimize is not None:
            self.minimize(minimize)
        if maximize is not None:
            self.maximize(maximize)
        self.solver = self.gurobi(keep_dump=keep_dump, unlink=unlink)
        self.solver.solver.Params.Method = Method
        if TimeLimit is not None:
            self.solver.solver.Params.TimeLimit = TimeLimit
        for k, v in kwargs.items():
            setattr(self.solver.solver.Params, k, v)
        return self.solver.solve()

    def evaluate(self, array):
        if self.solver is None:
            raise RuntimeError(
                f"{self} doesn't have an underlying solver."
            )
        return array.to(self.solver).evaluate()

    def dump(self, path=None, unlink=True):
        if path is None:
            timestamp = re.sub('[\s_\-\:\.]', '_', f"{datetime.datetime.now()}")
            path = f"sytorch_dump_{timestamp}.lp"
        assert pathlib.Path(path).suffix == ".lp"

        # progress = tqdm(desc="writing constraints")
        with open(path, 'w+') as f:

            if self.optimization_mode is not None:
                assert self.objective is not None
                f.write(self.optimization_mode + '\n')
                f.write(str(self.objective.item(0)) + '\n')

            f.write("Subject To\n")
            # f.writelines(tqdm(_iter_nested_iterable(self.constraints), desc="writing constraints"))
            f.write(
                "".join(tuple(
                    tqdm(_iter_nested_iterable(self.constraints), desc="writing constraints", smoothing=False)
                ))
            )

            # for tmp in tqdm(self.constr_files, desc="copying constraints"):
            #     with open(tmp, 'r') as tmpfile:
            #         f.write(tmpfile.read())
            #     if unlink:
            #         pathlib.Path(tmp).unlink()

            f.write("Bounds\n")
            f.write(
                "".join(tuple(
                    LightningReal(r, solver=self).write()
                    for r in tqdm(range(self.num_reals()), total=self.num_reals(), desc='writing bounds', smoothing=False)
                ))
            )

            f.write("End\n")

        # print(path)
        return path

    def print(self):
        for c in self.constraints:
            print(c, end='')

    def computeIIS(self):
        assert self.solver is not None
        return self.solver.computeIIS()

    def gurobi(self, keep_dump=False, unlink=True):
        dump_file = self.dump(unlink=unlink)
        with _timeit2("reading lp"):
            solver = GurobiSolver(path=dump_file).verbose_(self._verbose)
        if not keep_dump:
            pathlib.Path(dump_file).unlink(missing_ok=True)

        return solver

    @property
    def supported_features(self):
        return (
            'lp',
            'qp',
            'milp',
            'miqp',
            'miqcp',
        )

    @property
    def stype(self):
        return stype.lightning

    def num_reals(self):
        return self._real_var_counter.value

    def real(self, lb=None, ub=None):
        return self.reals((1,), lb=lb, ub=ub).reshape(())

    @overload
    def reals(self, shape, **kwargs): ...

    def reals(self, shape, lb=None, ub=None, mask=None, **kwargs):
        if mask is None:
            size = np.prod(shape, dtype=int)
            ids = self._real_var_counter.allocate(size)
            return np.fromiter(
                map_with_kwargs(LightningReal, ids, solver=self, lb=lb, ub=ub),
                dtype=object,
                count=size,
            ).view(SymbolicLightningArray).to(self).reshape(shape)

        else:
            size = np.broadcast_to(np.empty(1), shape)[mask].size
            ids = self._real_var_counter.allocate(size)
            out = np.empty(shape, dtype=object)
            out[mask] = np.fromiter(
                map_with_kwargs(LightningReal, ids, solver=self, lb=lb, ub=ub),
                dtype=object,
                count=size,
            ).reshape(out[mask].shape)
            return out.view(SymbolicLightningArray).to(self).reshape(shape)

    def einsum(self, *args, **kwargs):
        return einsum(*args, **kwargs, solver=self)

    def update(self):
        return self

    def overapproximate(self, expr, inplace=False):
        assert inplace is False
        return expr.to(solver=self.gurobi().verbose_(False)).overapproximate()
