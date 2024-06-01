import warnings; warnings.filterwarnings("ignore")

""" Import our framework `sytorch`
    ==============================

    We implement our framework `sytorch`, which extends PyTorch with symbolic
    execution. It supports

    - Symbolic forward execution of concrete (or symbolic) input tensors with
      some DNN parameters considered as symbolic variables.

    - Encoding encoding of symbolic constraints on (array of) symbolic
      expressions to formulate an LP problem.

    - Optimize the LP problem using Gurobi and update the DNN parameters
      using the optimal solution.

    And `sytorch` inherits everything from `torch.*`.

    We implement the repair approaches APRNN and PRDNN experiments using
    `sytorch` in scripts, just like implementing a new training approach in a
    `torch` training script.
"""
import sytorch as st

""" Working device and data type.
    =============================

    - `device` specifies the working device. Our framework `sytorch` supports
      tensors or parameters on GPU (`cuda`) and handle them properly.

    - `dtype` specifies the working data type. Because Gurobi internally uses
      double precision (`float64`), we recommend to use `float64` when encoding
      the LP problem. After repair, user could cast the DNN back to the desired
      data type.
"""

device = st.device('cpu')
dtype = st.float64

""" Loading the buggy DNN.
    ======================

    `st.nn` extends `torch.nn` and provides the exact same interface to
    construct a DNN. For example, the following code constructs a
    fully-connected ReLU DNN.

    One could also take a PyTorch DNN `torch_dnn_object` and convert it to a
    SyTorch-compatible DNN using `st.nn.from_torch`:

    ```
    dnn = st.nn.from_torch(torch_dnn)
    ```

    Moreover, one could take an saved ONNX DNN `onnx_dnn_path` and load it as a
    SyTorch-compatible DNN using `st.nn.from_file`:

    ```
    dnn = st.nn.from_file(onnx_dnn_path)
    ```

    Although for now we only support sequential ONNX DNNs, i.e., no skip
    connections.
"""

dnn = st.nn.Sequential(
    st.nn.Linear(1, 3),
    st.nn.ReLU(),
    st.nn.Linear(3, 1),
).to(device=device, dtype=dtype)

""" Here we load the example DNN `N1` used in the overview of our paper. """
with st.no_grad():
    dnn[0].weight[:] = st.tensor([-1.0, 1.0, 0.5], device=device, dtype=dtype)[:,None]
    dnn[0].bias  [:] = st.tensor([0.0, -2.0, 0.0], device=device, dtype=dtype)
    dnn[2].weight[:] = st.tensor([0.5, -0.5, 1.,], device=device, dtype=dtype)[None,:]
    dnn[2].bias  [:] = -0.5

""" Load buggy inputs.
    ==================

    Just like PyTorch, SyTorch takes any tensor of compatible shape as input,
    and symbolically forwards it through the network. One could also use
    existing torch datasets and dataloaders from training scripts.

    Here we load the buggy input point from Section 3.1 of our paper.
"""
points = st.tensor([ [-1.5], [-0.5], ], dtype=dtype, device=device)

""" Create a new Gurobi-based solver.
    ================================

    - `solver.solver` is the underlying gurobipy solver (model).

    - `GurobiSolver` provides
       - `.reals(shape)` to create a symbolic array (`numpy.ndarray`) of
         gurobipy variables.
       - `add_constraints(constrs)` to add constraints.
       - `minimize(obj)` and `maximize(obj)` to add an objective.
       - `solve()` to solve the model.

    - A symbolic array overrides the following operators for constructing
      symbolic expressions and formulas (constraints):
      - arithmetic operators `+`, `-`, `*`, `/`,
      - matrix multiplication `@`,
      - logical operators `==`, `>`, `>=`, `<`, `<=`,
      and provides the following methods for constructing symbolic expressions
      - `sum()`,
      - `abs_ub()`,
      - `max_ub()`,
      - `norm_ub()`.
"""
solver = st.GurobiSolver()

""" Attach the DNN to the solver.
    =============================

    - `.deepcopy()` returns a deepcopy of the DNN to repair. This is optional.
    - `.to(solver)` attaches the DNN to the solver, just like how you attach a
      DNN to a device in PyTorch.
    - `.repair()` turns on the repair mode and enables symbolic execution. It
       works just like `.train()` in PyTorch, which turns on the training mode.
"""
N = dnn.deepcopy().to(solver).repair()

""" Specify the symbolic weights.
    ==============================

    SyTorch extends PyTorch's `parameter` and `module` with a
    `.requires_symbolic_(...)` method, which makes the parameter or the
    module's parameters symbolic.

    For example, the following code makes the first layer weight and bias
    symbolic:

    ```
    dnn[1].weight.requires_symbolic_(lb=-3., ub=3.)
    ```

    The following code makes the first layer weight and bias symbolic:

    ```
    dnn[1].requires_symbolic_(lb=-3., ub=3.)
    ```

    The following code makes the first layer weight and all layers' bias symbolic:

    ```
    dnn.requires_symbolic_weight_and_bias(lb=-3., ub=3.)
    ```
"""
N.requires_symbolic_weight_and_bias(lb=-3., ub=3.)

""" Calculate the original output for minimization.
    ==============================================

    - The `st.no_symbolic()` context turns off the symbolic execution, just like
      `torch.no_grad()` turns off the gradient computation in PyTorch.
"""
with st.no_symbolic(), st.no_grad():
    reference_outputs = N(points)

""" Calculate the symbolic output.
    ===========================

    `N(points)` symbolically forwards `points` through `N` with the default
    reference points `points`. It is equivalent to

    ```
    pattern = N.activation_pattern(points)
    symbolic_output = N(points, pattern=pattern)
    ```

    `symbolic_output` is a `numpy.ndarray` of symbolic expressions with the
    same shape as `reference_outputs`.
"""
symbolic_outputs = N(points)

""" Construct the minimization objective.
    ====================================

    - `param_deltas` is the concatenation of all parameter delta in a 1D array.

    - `output_deltas` is the symbolic output delta flatten into 1D array.

    - `all_deltas` is the concatenation of `output_deltas` and `param_deltas`.

    - `.alias()` creates a corresponding array of variables that equals to
      the given array of symbolic expressions.

    - `.norm_ub('linf+l1_normalized')` encodes the (upper-bound) of
      the sum of L^inf norm and normalized L^1 norm.

    - `solver.minimize(obj)` sets the minimization objective.
"""
param_deltas = N.parameter_deltas(concat=True)
output_deltas = (symbolic_outputs - reference_outputs).flatten().alias()
all_deltas = st.cat([output_deltas, param_deltas])
obj = all_deltas.norm_ub('linf+l1_normalized')

""" Solve the constraints while minimizing the objective.
    ====================================================

    Here we constrain the `symbolic_outputs` to be within `[-0.1, 0.1]`.
    `solver.solve(*constrs, minimize=obj)` solves the constraints `constrs`
    while minimizing `obj`. Here `constrs=(-0.1 <= symbolic_outputs,
    symbolic_outputs <= 0.1)`.
"""
feasible = solver.solve(
    -0.1 <= symbolic_outputs,
    symbolic_outputs <= 0.1,
    minimize=obj
)

""" Update `N` with new parameters.
    ===============================

    - `.update_()` inplace updates `N`'s parameters with the solution.

    - `.repair(False)` turns off the repair (symbolic execution) mode.
"""
if feasible:
    N.update_().repair(False)
    print("Succeed.")

    with st.no_grad():
        print(f"Original output which is outside [-0.1, 0.1]:")
        print(dnn(points))
        print(f"Repaired output which is within [-0.1, 0.1]:")
        print(N(points))

else:
    print("Infeasible.")
