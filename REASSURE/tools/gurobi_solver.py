from gurobipy import Model, GRB
import numpy as np, sys, os
# from scipy.optimize import linprog


def blockPrint():
    sys.stdout = open(os.devnull, 'w')

# Restore
def enablePrint():
    sys.stdout = sys.__stdout__

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

def solve_LP(c: np.array, A: np.array, b: np.array, bounds, is_minimum=True, DualReductions=1):
    # print("using gurobi")
    assert len(c) == len(A[0])
    assert len(A) == len(b)
    x_dim = len(c)
    blockPrint()
    m = Model()
    m.Params.Threads = os.cpu_count()
    m.Params.DualReductions = DualReductions
    m.Params.LogToConsole = 0
    enablePrint()
    if bounds:
        x = m.addMVar(x_dim, lb=bounds[0], ub=bounds[1])
    else:
        x = m.addMVar(x_dim, lb=-GRB.INFINITY, ub=GRB.INFINITY)
    if is_minimum:
        m.setObjective(c @ x, GRB.MINIMIZE)
    else:
        m.setObjective(c @ x, GRB.MAXIMIZE)
    m.addConstr(A @ x <= b, name='Ab')
    m.optimize()
    if m.status == GRB.UNBOUNDED:
        if is_minimum:
            return -np.inf
        else:
            return np.inf
    elif m.status == GRB.OPTIMAL:
        obj = m.getObjective()
        return obj.getValue()
    elif DualReductions == 1:
        return solve_LP(c, A, b, bounds, is_minimum, DualReductions=0)
    else:
        print('error')
    # if is_minimum:
    #     return linprog(c, A, b, bounds).fun
    # else:
    #     return -linprog(-c, A, b, bounds).fun


if __name__ == '__main__':
    A, b, c = [[1.0, 1.0]], [1.0], [1.0, 1.0]
    A, b, c = np.array(A), np.array(b), np.array(c)
    value = solve_LP(c, A, b, (None, None), is_minimum=False)
    print(value)





