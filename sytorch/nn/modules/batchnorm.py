from __future__ import annotations
from typing import Tuple, Any
import torch.nn as nn
import torch.nn.functional as F

from sytorch.pervasives import *
from sytorch.solver import *
from sytorch.solver import lightning
from ..symbolic_mode import no_symbolic
from .module import *
from .module import T

class BatchNorm2d(LinearLayer, nn.BatchNorm2d):

    def requires_symbolic_(self: T, mode=True, row_mask=None, mask=None, weight=True, *args, **kwargs) -> T:
        if row_mask is not None and self.affine:
            if isinstance(row_mask, int):
                row_mask = as_slice[:row_mask]

            self.row_mask = np.zeros(self.bias.shape[0], dtype=bool)
            self.row_mask[row_mask] = True
            row_mask = self.row_mask
            if weight:
                self.weight.requires_symbolic_(*args, mode=mode, mask=row_mask, **kwargs)
            self.bias  .requires_symbolic_(*args, mode=mode, mask=row_mask, **kwargs)

        else:
            for param in self.parameters():
                param.requires_symbolic_(*args, mode=mode, mask=mask, **kwargs)

        return self

    def forward_symbolic(self, input, pattern=None):

        if self.affine:
            weight = self.weight.array_if_symbolic()
            bias = self.bias.array_if_symbolic()
            configuration = tuple(map(type, (input, weight, bias)))

        else:
            weight = None
            bias = None
            configuration = tuple(map(type, (input, )))

        if all(issubclass(ty, Tensor) for ty in configuration):
            with no_symbolic():
                return self(input)

        elif any(issubclass(ty, SymbolicGurobiArray) for ty in configuration):
            assert self.row_mask is None
            return batch_norm_2d(
                input = input,
                mean  = self.running_mean,
                var   = self.running_var,
                gamma = weight,
                beta  = bias,
                eps   = self.eps
            )

        elif any(issubclass(ty, SymbolicLightningArray) for ty in configuration):
            if self.row_mask is not None:
                # print("batchnorm2d row mask")
                if isinstance(input, SymbolicLightningArray):
                    assert input.mask is not None
                    assert (input.row_mask == self.row_mask).all()
                    _input_mask = as_slice[...,self.row_mask,:,:]
                    _inverted_input_mask = as_slice[...,~self.row_mask,:,:]
                    assert input[input.mask].size == input[_input_mask].size
                    # print(input.shape, input[input.mask].shape)
                    # print(input[input.mask])
                    # print(input[~input.mask])

                    sym_output = lightning.batch_norm_2d(
                        input = input[_input_mask],
                        mean  = self.running_mean[self.row_mask],
                        var   = self.running_var[self.row_mask],
                        gamma = weight[self.row_mask],
                        beta  = bias[self.row_mask],
                        eps   = self.eps,
                        executor = self.executor,
                    )
                    solver = sym_output.solver
                    array_type = type(sym_output)

                    output = np.empty(input.shape, dtype=object)
                    output[_input_mask] = sym_output
                    with no_symbolic():
                        output[_inverted_input_mask] = F.batch_norm(
                            torch.from_numpy(
                                input[_inverted_input_mask].astype(input._concrete_dtype)
                            ).to(self.bias.device),
                            self.running_mean[~self.row_mask],
                            self.running_var[~self.row_mask],
                            self.weight[~self.row_mask],
                            self.bias[~self.row_mask],
                            False,
                            self.momentum if self.momentum is not None else 0.0,
                            self.eps,
                        ).cpu().detach().numpy()

                    output = output.view(array_type).to(solver)
                    output.mask = input.mask
                    output.row_mask = input.row_mask
                    output._concrete_dtype = input._concrete_dtype

                    return output

                else:
                    raise NotImplemented

            else:
                return lightning.batch_norm_2d(
                    input = input,
                    mean  = self.running_mean,
                    var   = self.running_var,
                    gamma = weight,
                    beta  = bias,
                    eps   = self.eps,
                    executor = self.executor,
                )

        else:
            raise NotImplementedError(
                f"unimplemented {type(self)} symbolic forward for {type(input)}"
            )

from . import functional

def batch_norm_2d(
    input,
    mean,
    var,
    gamma,
    beta,
    eps,
):
    if gamma is not None and beta is not None:
        """ output = (input - mean) * invstd * gamma + beta
            output = input * invstd_gamma - mean_invstd_gamma + beta
            output = input * invstd_gamma + minus_mean_invstd_gamma_plus_beta
        """
        invstd_gamma = (1. / torch.sqrt(var + eps)) * gamma
        constant = (- mean * invstd_gamma) + beta
        # einsum(
        #     "ncwh,c+c->ncwh",
        #     input, invstd_gamma, constant,
        #     *args, **kwargs
        # )

        # import pdb; pdb.set_trace()

        # 1. row mask

        # 1, C * C + C

        N, C, W, H = input.shape

        # import scipy.sparse
        return functional.linear(
            input.permute((0,2,3,1)).reshape((N*W*H,C)),
            # broadcast_at(invstd_gamma, 1, (C,)),
            # scipy.sparse.diags(
            #     [invstd_gamma], [0],
            #     shape=(C, C),
            #     dtype=np.float64, format="lil"
            # ),
            torch.diag(invstd_gamma),
            constant
        ).reshape(N,W,H,C).permute((0,3,1,2))

        # input = input.permute((N,W,H,C)).reshape((N*W*H,C))
        # weight = broadcast_at(invstd_gamma, 0, C) # or 1?
        # bias = constant

        # N, I = nwh, c
        # O, I = c, c
        # O    = c
        # ->
        # N, O = nwh, c

    else:
        assert False
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

