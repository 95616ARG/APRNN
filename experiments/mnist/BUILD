filegroup(
    name = "all_py_srcs",
    srcs = glob(["*.py"]),
    visibility = ["//:__subpackages__"],
)

py_library(
    name = "mnist",
    srcs = glob(["*.py"]),
    visibility = ["//:__subpackages__"],
    deps = [
        ":evaluation",
        ":models",
        ":repair",
        "//sytorch",
        "//experiments/base",
    ],
)

py_library(
    name = "models",
    srcs = ["models.py"],
)

py_library(
    name = "repair",
    srcs = ["repair.py"],
    deps = [
        "//sytorch",
    ],
)

py_library(
    name = "evaluation",
    srcs = ["evaluation.py"],
    deps = [
        "//sytorch",
    ],
)
