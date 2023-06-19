filegroup(
    name = "all_py_srcs",
    srcs = glob(["*.py"]),
    visibility = ["//sytorch:__subpackages__"],
)

py_library(
    name = "util",
    srcs = ["__init__.py"],
    visibility = ["//sytorch:__subpackages__"],
    deps = [
        ":base",
        ":gurobi",
        ":z3",
    ],
)

py_library(
    name = "base",
    srcs = ["base.py"],
)

py_library(
    name = "gurobi",
    srcs = ["gurobi.py"],
    deps = [":base"],
)

py_library(
    name = "z3",
    srcs = ["z3.py"],
    deps = [":base"],
)
