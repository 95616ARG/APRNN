import gurobipy as gp
from gurobipy import GRB
import numpy as np
from ..solver import *

def boxify(A, sense, b, lb=-100., ub=100., buffer=1., min_range=0.01):
    """Takes a polytope defined by linear constraints of the form Ax < b. 
    Returns a box contained within the original polytope (underapproximation)
    in the form of lists of lower and upper bounds on each variable.

    This algorithm is adapted from "Inner and outer approximations of polytopes
    using boxes" by Bemporad et al.
    https://doi.org/10.1016/S0925-7721(03)00048-8 
    """
    assert(isinstance(A, np.ndarray))  # Matrix A must be numpy array
    if len(A) == 0:
        print("No constraints to boxify.")
        return None

    model = gp.Model()
    num_vars = len(A[0])
    model.Params.OutputFlag = 0.

    # Represent the LP as Ax <= b
    A, sense, b = fix_sense(A, sense, b)

    # Initialize the A+ matrix 
    print("Initializing model...")
    zero_indices = (A <= 0)
    A_plus = np.copy(A)
    A_plus[zero_indices] = 0.

    # Initialize variables
    xs = model.addVars(num_vars, lb=lb, ub=ub, name="X")
    ys = model.addVars(num_vars, lb=min_range, ub=(ub-lb), name="Y")
    lnys = model.addVars(num_vars, name="lnY")
    if min_range==0:
        ypluses = model.addVars(num_vars, lb=(buffer), ub=((ub-lb)+buffer), name="Y+")
        for y, yplus, lny in zip(ys.values(), ypluses.values(), lnys.values()):
            model.addConstr(yplus == y + buffer)
            model.addGenConstrLog(yplus, lny)
    else:
        for y, lny in zip(ys.values(), lnys.values()):
            model.addGenConstrLog(y, lny)

    # Set up boxify LP
    print("Matrix multiplication...")
    x = np.asarray(xs.values()).reshape(num_vars)
    linexprA = (A@x).tolist()

    y = np.asarray(ys.values()).reshape(num_vars)
    linexprAplus = (A_plus@y).tolist()

    print("Adding constraints...")
    for l1, l2, b0 in zip(linexprA, linexprAplus, b):
        model.addConstr(b0 >= l1 + l2)

    # Set objective: maximize range of each variable
    model.ModelSense = GRB.MAXIMIZE
    model.setObjectiveN(lnys.sum(), 0, weight=3.)

    # Secondary objective: minimize abs(midpoint) of each range (x, x+y)
    # midpoint = (x+(x+y))/2 = (2x + y)/2
    midpts = model.addVars(num_vars, lb=lb, ub=ub, name="mdpt")
    model.addConstrs(midpts[i]==(((2*x[i])+y[i])*0.5) for i in range(num_vars))
    abs = model.addVars(num_vars, lb=0., ub=ub, name="abs")
    model.addConstrs(abs[i]==gp.abs_(midpts[i]) for i in range(num_vars))
    model.setObjectiveN((-1)*abs.sum(), 1, weight=1.)

    # Solve boxify LP
    model.Params.DualReductions = 0
    model.update()
    model.optimize()
    feasible = (model.status == GRB.OPTIMAL)

    if not feasible:
        print("Unable to find feasible box solution.")
        return None, None
    else:
        # Return lower and upper bound for each variable
        lower_bound = [x.X for x in xs.values()]
        y_range = [y.X for y in ys.values()]
        upper_bound = [l + y for l,y in zip(lower_bound, y_range)]

        return lower_bound, upper_bound   

def fix_sense(A, sense, b):
    """Returns A, sense, b adjusted such that all elements of sense are '<'."""
    new_As = []
    for i in range(len(sense)):
        if sense[i]=='>':
            A[i] = A[i]*(-1)
            b[i] = b[i]*(-1)
        elif sense[i]=='=':
            new_As.append(A[i]*(-1))
            b.append(b[i]*(-1))
    if len(new_As)>0:
        A = np.vstack((A, new_As))
    
    sense = ['<' for _ in b]
    return A, sense, b