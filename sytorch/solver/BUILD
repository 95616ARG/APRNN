load("@bazel_python//:bazel_pybind11.bzl", "pybind_extension")

filegroup(
    name = "all_py_srcs",
    srcs = glob(["*.py"]),
    visibility = ["//sytorch:__subpackages__"],
)

py_library(
    name = "solver",
    srcs = ["__init__.py"],
    visibility = ["//sytorch:__subpackages__"],
    deps = [
        ":base",
        ":default",
        ":gurobi",
        ":lightning",
        ":symbolic_array",
        ":z3",
        "//sytorch/util",
    ],
)

py_library(
    name = "base",
    srcs = ["base.py"],
)

py_library(
    name = "symbolic_array",
    srcs = ["symbolic_array.py"],
    deps = [":base"],
)

py_library(
    name = "gurobi",
    srcs = ["gurobi.py"],
    deps = [
        ":base",
        ":symbolic_array",
        "//sytorch/util",
    ],
)

py_library(
    name = "lightning",
    srcs = ["lightning.py"],
    deps = [
        ":base",
        ":symbolic_array",
    ],
)

py_library(
    name = "z3",
    srcs = ["z3.py"],
    deps = [
        ":base",
        ":symbolic_array",
        "//sytorch/util",
    ],
)

py_library(
    name = "default",
    srcs = ["default.py"],
    deps = [
        ":base",
        ":symbolic_array",
    ],
)
