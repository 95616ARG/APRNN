from __future__ import annotations
from typing import Callable, Iterable, Tuple, TypeVar, Union
import warnings
import torch
from torch import Tensor
from tqdm.auto import tqdm
import torch.nn as nn
from torch.autograd.functional import (_autograd_grad, _grad_postprocess,
    _tuple_postprocess, _as_tuple, _check_requires_grad)

__all__ = [
    'jacobian_with_inputs_inplace',
    'jacobian_wrt_params',
]

""" This module provides functionalities to compute jacobian w.r.t. any network
parameter, given input samples.

`torch.autograd.functional.jacobian` always treat the tensors to find Jacobian
of as new detached tensors, which prevents us from finding the Jacobian of
network parameters. Therefore here we modifies
`torch.autograd.functional.jacobian` to implement the other behavior discussed
in https://github.com/pytorch/pytorch/issues/32576, which uses inputs (to
jacobian(.), which are supposed to be parameters) as-is, instead of creating a
different object.
"""

def _grad_preprocess_with_inputs_inplace(inputs, create_graph, need_graph):
    """ Modified from `torch.autograd.functional._grad_postprocess` which keeps
    inputs as-is instead of always creating new Tensor objects for our purpose
    of computing the Jacobian of module parameters. For now the `create_graph`
    and `need_graph` parameters are just placeholder to be consistent with
    `torch.autograd.functional._grad_postprocess`. Check this issue for more
    details about the difference between torch's and this implementation:
    https://github.com/pytorch/pytorch/issues/32576

    """
    res = []
    for inp in inputs:
        if create_graph and inp.requires_grad:
            raise NotImplemented
        else:
            res.append(inp)
    return tuple(res)

def jacobian_with_inputs_inplace(func, inputs, create_graph=False, strict=False, vectorize=False):
    r"""Function that computes the Jacobian of a given function.

    Args:
        func (function): a Python function that takes Tensor inputs and returns
            a tuple of Tensors or a Tensor.
        inputs (tuple of Tensors or Tensor): inputs to the function ``func``.
        create_graph (bool, optional): If ``True``, the Jacobian will be
            computed in a differentiable manner. Note that when ``strict`` is
            ``False``, the result can not require gradients or be disconnected
            from the inputs.  Defaults to ``False``.
        strict (bool, optional): If ``True``, an error will be raised when we
            detect that there exists an input such that all the outputs are
            independent of it. If ``False``, we return a Tensor of zeros as the
            jacobian for said inputs, which is the expected mathematical value.
            Defaults to ``False``.
        vectorize (bool, optional): This feature is experimental, please use at
            your own risk. When computing the jacobian, usually we invoke
            ``autograd.grad`` once per row of the jacobian. If this flag is
            ``True``, we use the vmap prototype feature as the backend to
            vectorize calls to ``autograd.grad`` so we only invoke it once
            instead of once per row. This should lead to performance
            improvements in many use cases, however, due to this feature
            being incomplete, there may be performance cliffs. Please
            use `torch._C._debug_only_display_vmap_fallback_warnings(True)`
            to show any performance warnings and file us issues if
            warnings exist for your use case. Defaults to ``False``.

    Returns:
        Jacobian (Tensor or nested tuple of Tensors): if there is a single
        input and output, this will be a single Tensor containing the
        Jacobian for the linearized inputs and output. If one of the two is
        a tuple, then the Jacobian will be a tuple of Tensors. If both of
        them are tuples, then the Jacobian will be a tuple of tuple of
        Tensors where ``Jacobian[i][j]`` will contain the Jacobian of the
        ``i``\th output and ``j``\th input and will have as size the
        concatenation of the sizes of the corresponding output and the
        corresponding input and will have same dtype and device as the
        corresponding input.

    Example:

        >>> def exp_reducer(x):
        ...   return x.exp().sum(dim=1)
        >>> inputs = torch.rand(2, 2)
        >>> jacobian(exp_reducer, inputs)
        tensor([[[1.4917, 2.4352],
                 [0.0000, 0.0000]],
                [[0.0000, 0.0000],
                 [2.4369, 2.3799]]])

        >>> jacobian(exp_reducer, inputs, create_graph=True)
        tensor([[[1.4917, 2.4352],
                 [0.0000, 0.0000]],
                [[0.0000, 0.0000],
                 [2.4369, 2.3799]]], grad_fn=<ViewBackward>)

        >>> def exp_adder(x, y):
        ...   return 2 * x.exp() + 3 * y
        >>> inputs = (torch.rand(2), torch.rand(2))
        >>> jacobian(exp_adder, inputs)
        (tensor([[2.8052, 0.0000],
                [0.0000, 3.3963]]),
         tensor([[3., 0.],
                 [0., 3.]]))
    """

    with torch.enable_grad():
        is_inputs_tuple, inputs = _as_tuple(inputs, "inputs", "jacobian")
        inputs = _grad_preprocess_with_inputs_inplace(inputs, create_graph=create_graph, need_graph=True)

        outputs = func(*inputs)
        is_outputs_tuple, outputs = _as_tuple(outputs,
                                              "outputs of the user-provided function",
                                              "jacobian")
        _check_requires_grad(outputs, "outputs", strict=strict)



        if vectorize:
            if strict:
                raise RuntimeError('torch.autograd.functional.jacobian: `strict=True` '
                                   'and `vectorized=True` are not supported together. '
                                   'Please either set `strict=False` or '
                                   '`vectorize=False`.')
            # NOTE: [Computing jacobian with vmap and grad for multiple outputs]
            #
            # Let's consider f(x) = (x**2, x.sum()) and let x = torch.randn(3).
            # It turns out we can compute the jacobian of this function with a single
            # call to autograd.grad by using vmap over the correct grad_outputs.
            #
            # Firstly, one way to compute the jacobian is to stack x**2 and x.sum()
            # into a 4D vector. E.g., use g(x) = torch.stack([x**2, x.sum()])
            #
            # To get the first row of the jacobian, we call
            # >>> autograd.grad(g(x), x, grad_outputs=torch.tensor([1, 0, 0, 0]))
            # To get the 2nd row of the jacobian, we call
            # >>> autograd.grad(g(x), x, grad_outputs=torch.tensor([0, 1, 0, 0]))
            # and so on.
            #
            # Using vmap, we can vectorize all 4 of these computations into one by
            # passing the standard basis for R^4 as the grad_output.
            # vmap(partial(autograd.grad, g(x), x))(torch.eye(4)).
            #
            # Now, how do we compute the jacobian *without stacking the output*?
            # We can just split the standard basis across the outputs. So to
            # compute the jacobian of f(x), we'd use
            # >>> autograd.grad(f(x), x, grad_outputs=_construct_standard_basis_for(...))
            # The grad_outputs looks like the following:
            # ( torch.tensor([[1, 0, 0],
            #                 [0, 1, 0],
            #                 [0, 0, 1],
            #                 [0, 0, 0]]),
            #   torch.tensor([[0],
            #                 [0],
            #                 [0],
            #                 [1]]) )
            #
            # But we're not done yet!
            # >>> vmap(partial(autograd.grad(f(x), x, grad_outputs=...)))
            # returns a Tensor of shape [4, 3]. We have to remember to split the
            # jacobian of shape [4, 3] into two:
            # - one of shape [3, 3] for the first output
            # - one of shape [   3] for the second output

            # Step 1: Construct grad_outputs by splitting the standard basis
            output_numels = tuple(output.numel() for output in outputs)
            grad_outputs = _construct_standard_basis_for(outputs, output_numels)
            flat_outputs = tuple(output.reshape(-1) for output in outputs)

            # Step 2: Call vmap + autograd.grad
            def vjp(grad_output):
                vj = list(_autograd_grad(flat_outputs, inputs, grad_output, create_graph=create_graph))
                for el_idx, vj_el in enumerate(vj):
                    if vj_el is not None:
                        continue
                    vj[el_idx] = torch.zeros_like(inputs[el_idx])
                return tuple(vj)

            jacobians_of_flat_output = _vmap(vjp)(grad_outputs)

            # Step 3: The returned jacobian is one big tensor per input. In this step,
            # we split each Tensor by output.
            jacobian_input_output = []
            for jac, input_i in zip(jacobians_of_flat_output, inputs):
                jacobian_input_i_output = []
                for jac, output_j in zip(jac.split(output_numels, dim=0), outputs):
                    jacobian_input_i_output_j = jac.view(output_j.shape + input_i.shape)
                    jacobian_input_i_output.append(jacobian_input_i_output_j)
                jacobian_input_output.append(jacobian_input_i_output)

            # Step 4: Right now, `jacobian` is a List[List[Tensor]].
            # The outer List corresponds to the number of inputs,
            # the inner List corresponds to the number of outputs.
            # We need to exchange the order of these and convert to tuples
            # before returning.
            jacobian_output_input = tuple(zip(*jacobian_input_output))

            jacobian_output_input = _grad_postprocess(jacobian_output_input, create_graph)
            return _tuple_postprocess(jacobian_output_input, (is_outputs_tuple, is_inputs_tuple))

        jacobian: Tuple[torch.Tensor, ...] = tuple()

        for i, out in enumerate(outputs):

            # mypy complains that expression and variable have different types due to the empty list
            jac_i: Tuple[List[torch.Tensor]] = tuple([] for _ in range(len(inputs)))  # type: ignore[assignment]
            for j in tqdm(range(out.nelement()), desc='computing jacobian...', leave=False):
                vj = tuple(
                    _vj if _vj is not None else _vj
                    for _vj in
                    _autograd_grad((out.reshape(-1)[j],), inputs,
                                        retain_graph=True, create_graph=create_graph)
                )

                for el_idx, (jac_i_el, vj_el, inp_el) in enumerate(zip(jac_i, vj, inputs)):
                    if vj_el is not None:
                        if strict and create_graph and not vj_el.requires_grad:
                            msg = ("The jacobian of the user-provided function is "
                                   "independent of input {}. This is not allowed in "
                                   "strict mode when create_graph=True.".format(i))
                            raise RuntimeError(msg)
                        jac_i_el.append(vj_el)
                    else:
                        if strict:
                            msg = ("Output {} of the user-provided function is "
                                   "independent of input {}. This is not allowed in "
                                   "strict mode.".format(i, el_idx))
                            raise RuntimeError(msg)
                        jac_i_el.append(torch.zeros_like(inp_el))

            jacobian += (tuple(torch.stack(jac_i_el, dim=0).view(out.size()
                         + inputs[el_idx].size()) for (el_idx, jac_i_el) in enumerate(jac_i)), )

        jacobian = _grad_postprocess(jacobian, create_graph)

        return _tuple_postprocess(jacobian, (is_outputs_tuple, is_inputs_tuple))

_tensor_or_tuple_of_tensors_T = Union[Tensor, Iterable[Tensor]]
""" Type of acceptable types/structures of parameters to find jacobian, which
    can be just one Tensor or a tuple of Tensors.
"""

_tensor_or_tuple_of_tensors_T1 = TypeVar("_tensor_or_tuple_of_tensors_T1", bound=_tensor_or_tuple_of_tensors_T)
""" A type variable of `_tensor_or_tuple_of_tensors_T`. """

def _params_in_module(params: _tensor_or_tuple_of_tensors_T, mod: nn.Module) -> bool:
    if isinstance(params, Tensor):
        params = (params)

    for param in params:
        if param not in mod.parameters():
            return False

    return True

def jacobian_wrt_params(
    func: nn.Module | Callable[[Tensor], _tensor_or_tuple_of_tensors_T],
    input,
    params: _tensor_or_tuple_of_tensors_T1,
) -> _tensor_or_tuple_of_tensors_T1:
    """ Function that computes the jacobian of a given function and given input
    samples, with respect to the given parameters of the function.

    Parameters
    ==========
    func (function): a Python function or Torch module that takes and returns a
        Tensor or tuple of Tensors.
    inputs (tuple of Tensors or Tensor): input samples to function `func`, but
        __NOT__ inputs to find jacobian for.
    params (tuple of Tensors or Tensor): parameters of `func` to find jacobian
        for.

    Returns
    =======
    Jacobian (tuple of Tensors or Tensor): jacobian w.r.t. params.

    """
    def func_wrapper(*params: _tensor_or_tuple_of_tensors_T1) -> _tensor_or_tuple_of_tensors_T1:
        return func(input)
    return jacobian_with_inputs_inplace(func_wrapper, params)
