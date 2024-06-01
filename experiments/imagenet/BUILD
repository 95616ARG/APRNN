package(
    default_visibility  = ["//visibility:public"],
)

filegroup(
    name = "all",
    srcs = glob(["**"])
)

filegroup(
    name = "all_py_srcs",
    srcs = glob(["*.py"]),
)

py_library(
    name = "imagenet",
    srcs = glob(["*.py"]),
    deps = [
        "//experiments/base",
        "//sytorch",
    ],
)
