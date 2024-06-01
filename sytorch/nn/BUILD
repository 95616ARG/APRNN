filegroup(
    name = "all_py_srcs",
    srcs = glob(["*.py"]),
    visibility = ["//sytorch:__subpackages__"],
)

py_library(
    name = "nn",
    srcs = ["__init__.py"],
    visibility = ["//sytorch:__subpackages__"],
    deps = [
        ":jacobian",
        ":parameter",
        ":symbolic_mode",
        ":underapprox",
        ":utils",
        "//sytorch/nn/modules",
        "//sytorch/solver",
    ],
)

py_library(
    name = "jacobian",
    srcs = ["jacobian.py"],
)

py_library(
    name = "symbolic_mode",
    srcs = ["symbolic_mode.py"],
)

py_library(
    name = "parameter",
    srcs = ["parameter.py"],
    deps = ["//sytorch/solver"],
)

py_library(
    name = "underapprox",
    srcs = ["underapprox.py"],
    deps = ["//sytorch/solver"],
)

py_library(
    name = "utils",
    srcs = ["utils.py"],
)
