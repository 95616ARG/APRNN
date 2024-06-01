filegroup(
    name = "all_py_srcs",
    srcs = glob(["*.py"]),
    visibility = ["//:__subpackages__"],
)

py_library(
    name = "acas",
    srcs = [":all_py_srcs"],
    visibility = ["//:__subpackages__"],
    deps = [
        "//sytorch",
        "//experiments/base",
    ],
)
