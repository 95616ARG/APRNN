filegroup(
    name = "all_py_srcs",
    srcs = glob(["*.py"]),
    visibility = ["//sytorch:__subpackages__"],
)

py_library(
    name = "modules",
    srcs = [
        # "__init__.py",
        ":all_py_srcs"
    ],
    visibility = ["//sytorch:__subpackages__"],
    deps = [
        "//sytorch/solver",
    ],
)

# py_library(
#     name = "module",
#     srcs = ["module.py"],
#     deps = [
#         "//sytorch/solver",
#     ],
# )

# py_library(
#     name = "container",
#     srcs = ["container.py"],
# )

# py_library(
#     name = "linear",
#     srcs = ["linear.py"],
#     deps = [
#         "//sytorch/solver",
#     ],
# )

# py_library(
#     name = "conv",
#     srcs = ["conv.py"],
# )

# py_library(
#     name = "activation",
#     srcs = ["activation.py"],
# )

# py_library(
#     name = "dropout",
#     srcs = ["dropout.py"],
# )

# py_library(
#     name = "pooling",
#     srcs = ["pooling.py"],
# )

# py_library(
#     name = "reshape",
#     srcs = ["reshape.py"],
# )

# py_library(
#     name = "batchnorm",
#     srcs = ["batchnorm.py"],
# )
