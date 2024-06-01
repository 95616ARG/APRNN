# pylint: disable=import-error

import sytorch as st
from sytorch import nn
from sytorch.pervasives import *

st.enable_debug()

dtype = torch.float64
device = torch.device('cpu')


def test_array():
    solver = st.LightningSolver()
    a, b = solver.reals((2, 3, 4))
    assert a.solver is solver

def test_eq():
    st.set_all_seed(0)
    solver = st.LightningSolver()
    a, b = solver.reals((2, 3, 4))
    t = st.randn(a.shape).to(device,dtype)
    assert solver.solve(a == b, a == t, b == t)
    assert (a.evaluate().to(device,dtype) == t).all()
    assert (b.evaluate().to(device,dtype) == t).all()

def test_ge():
    st.set_all_seed(0)
    solver = st.LightningSolver()
    a, b = solver.reals((2, 3, 4))
    t = st.randn(a.shape).to(device,dtype)
    assert solver.solve(a == t, b >= a, minimize=b.sum())
    assert (b.evaluate().to(device,dtype) == a.evaluate().to(device,dtype)).all()

def test_le():
    st.set_all_seed(0)
    solver = st.LightningSolver()
    a, b = solver.reals((2, 3, 4))
    t = st.randn(a.shape).to(device,dtype)
    assert solver.solve(a == t, b <= a, maximize=b.sum())
    assert (b.evaluate().to(device,dtype) == a.evaluate().to(device,dtype)).all()

def test_gt():
    st.set_all_seed(0)
    solver = st.LightningSolver()
    a, b = solver.reals((2, 3, 4))
    t = st.randn(a.shape).to(device,dtype)
    assert solver.solve(a == t, b > a, minimize=b.sum())
    assert (b.evaluate().to(device,dtype) > a.evaluate().to(device,dtype)).all()

def test_lt():
    st.set_all_seed(0)
    solver = st.LightningSolver()
    a, b = solver.reals((2, 3, 4))
    t = st.randn(a.shape).to(device,dtype)
    assert solver.solve(a == t, b < a, maximize=b.sum())
    assert (b.evaluate().to(device,dtype) < a.evaluate().to(device,dtype)).all()

def test_binary():
    for op in (
        '__add__',
        '__sub__',
        '__radd__',
        '__rsub__',
    ):
        st.set_all_seed(0)
        solver = st.LightningSolver()
        a, b = solver.reals((2, 3, 4))
        t1 = st.randn(a.shape).to(device,dtype)
        t2 = st.randn(b.shape).to(device,dtype)
        assert solver.solve(a == t1, getattr(a, op)(b) == getattr(t1, op)(t2))
        assert (b.evaluate().to(device,dtype) == t2).all()

    for op in (
        '__mul__',
        '__rmul__',
        '__truediv__',
        # '__rtruediv__',
    ):
        st.set_all_seed(0)
        solver = st.LightningSolver()
        a, b = solver.reals((2, 3, 4))
        t1 = st.randn(a.shape).to(device,dtype)
        t2 = st.randn(b.shape).to(device,dtype)
        assert solver.solve(a == t1, getattr(a, op)(t2) == b)
        assert (b.evaluate().to(device,dtype) == getattr(t1, op)(t2)).all()

    """ __matmul__ """
    st.set_all_seed(0)
    solver = st.LightningSolver()
    a = solver.reals((2, 3, 4))
    b = solver.reals((2, 3, 5))
    t1 = st.randn((2, 3, 4)).to(device,dtype)
    t2 = st.randn((4, 5)).to(device,dtype)
    assert solver.solve(a == t1, a @ t2 == b)
    assert (b.evaluate().to(device,dtype) == t1 @ t2).all()

    """ __rmatmul__ """
    st.set_all_seed(0)
    solver = st.LightningSolver()
    a = solver.reals((4, 5))
    b = solver.reals((2, 3, 5))
    t1 = st.randn((2, 3, 4)).to(device,dtype)
    t2 = st.randn((4, 5)).to(device,dtype)
    assert solver.solve(a == t2, t1 @ a == b)
    assert (b.evaluate().to(device,dtype) == t1 @ t2).all()

def test_abs_ub():
    st.set_all_seed(0)
    solver = st.LightningSolver()
    t = torch.randn((2,3,4)).to(device,dtype)
    a = solver.reals(t.shape)
    b = solver.reals(t.shape)
    a_abs_ub = a.abs_ub()
    b_abs_ub = b.abs_ub()
    assert solver.solve(a == t, b == -t, minimize=(a_abs_ub.sum() + b_abs_ub.sum()))
    assert (a_abs_ub.evaluate().to(device,dtype) == t.abs()).all()

def test_max_ub():
    shape = (2,3,4)
    for axis in (None, 0, 1, 2, -1, -2, -3):
        st.set_all_seed(0)
        solver = st.LightningSolver()
        t = torch.randn(shape).to(device,dtype)
        a = solver.reals(shape)
        a_max_ub = a.max_ub(axis=axis)
        t_max = t.amax(dim=axis)
        assert solver.solve(a == t, minimize=a_max_ub)
        assert tuple(a_max_ub.shape) == tuple(t_max.shape)
        assert (a_max_ub.evaluate().to(device,dtype) == t_max).all()

def test_unary_exact():
    shape = (2,3,4)
    for axis, op in itertools.product(
        (None, 0, 1, 2, -1, -2, -3),
        ('sum', 'mean'),
    ):
        dim = tuple(range(len(shape))) if axis is None else axis
        st.set_all_seed(0)
        solver = st.LightningSolver()
        t = torch.randn(shape).to(device,dtype)
        a = solver.reals(shape)
        a_out = getattr(a, op)(axis=axis)
        t_out = getattr(t, op)(dim=dim)
        assert solver.solve(a == t)
        assert tuple(a_out.shape) == tuple(t_out.shape)
        assert torch.allclose(a_out.evaluate().to(device,dtype), t_out)

def test_lightning_reals_bounds():
    shape = (2, 3)
    lb, ub = st.randn(2).numpy()
    if lb > ub:
        lb, ub = ub, lb
    for seed in range(2):
        solver = st.LightningSolver()
        st.set_all_seed(seed)
        xs = solver.reals(shape, lb=lb, ub=ub)

        for x in xs.flat:
            assert x.LB == lb and x.UB == ub
            lb2, ub2 = st.randn(2).numpy()
            x.LB, x.UB = lb2, ub2
            assert x.LB == lb2 and x.UB == ub2

        xs[0,0].LB = -np.inf
        xs[0,1].UB = np.inf
        xs[0,2].LB = -np.inf
        xs[0,2].UB = np.inf

        grb = solver.gurobi()
        for xg, x in zip(xs.to(grb).flat, xs.flat):
            assert xg.LB == x.lb and xg.UB == x.ub

def test_solver_reals_mask():
    solver = st.LightningSolver()
    shape = (4, 5)
    for seed in range(3):
        st.set_all_seed(seed)
        random_mask = st.randn(shape) > 0.
        x = solver.reals(shape, mask=random_mask)
        assert (x.view(ndarray)[~random_mask] == None).all()
        assert (x.view(ndarray)[ random_mask] != None).all()

def test_parameter_mask():
    with torch.no_grad():
        solver = st.LightningSolver()
        fc = nn.Linear(4, 5).to(solver)
        random_mask = st.randn(5, 4) > 0.
        fc.weight.requires_symbolic_(mask=random_mask)
        assert (fc.weight.concrete()[~random_mask] == fc.weight.symbolic()[~random_mask].view(ndarray)).all()
        assert all(map(lambda v: isinstance(v, st.LightningVar), fc.weight.symbolic()[random_mask].flat))

def test_linear_row_mask():
    with torch.no_grad():
        for row_mask in (
            0, 1, 2, 3, 4,
            as_slice[0:2], as_slice[-4:-2], as_slice[[1,2,3]],
            as_slice[:],
            [0, 2, 4],
        ):
            solver = st.LightningSolver()
            fc = nn.Linear(4, 5).to(solver)
            fc.requires_symbolic_(row_mask=row_mask)
            if isinstance(row_mask, int): row_mask = as_slice[:row_mask]
            mask = torch.zeros(5, dtype=bool)
            mask[row_mask] = True
            assert (fc.weight.concrete()[~mask,:] == fc.weight.symbolic()[~mask,:].view(ndarray)).all()
            assert (fc.bias.concrete()[~mask] == fc.bias.symbolic()[~mask].view(ndarray)).all()
            assert all(map(lambda v: isinstance(v, st.LightningVar), fc.weight.symbolic()[mask,:].flat))
            assert all(map(lambda v: isinstance(v, st.LightningVar), fc.bias.symbolic()[mask].flat))

def test_conv2d_row_mask():
    with torch.no_grad():
        for row_mask in (
            0, 1, 2, 3, 4,
            as_slice[0:2], as_slice[-4:-2], as_slice[[1,2,3]],
            as_slice[:],
            [0, 2, 4],
        ):
            solver = st.LightningSolver()
            conv = nn.Conv2d(4, 5, 3).to(solver)
            conv.requires_symbolic_(row_mask=row_mask)
            if isinstance(row_mask, int): row_mask = as_slice[:row_mask]
            mask = torch.zeros(5, dtype=bool)
            mask[row_mask] = True
            assert (conv.weight.concrete()[~mask,:] == conv.weight.symbolic()[~mask,:].view(ndarray)).all()
            assert (conv.bias.concrete()[~mask] == conv.bias.symbolic()[~mask].view(ndarray)).all()
            assert all(map(lambda v: isinstance(v, st.LightningVar), conv.weight.symbolic()[mask,:].flat))
            assert all(map(lambda v: isinstance(v, st.LightningVar), conv.bias.symbolic()[mask].flat))


try:
    from external.bazel_python.pytest_helper import main
    main(__name__, __file__)
except ImportError:
    pass
