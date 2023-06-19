py_test(
    name = "test_from_torch",
    size = "large",
    srcs = ["test_from_torch.py"],
    deps = [
        "//sytorch",
        "//sytorch/tests/nn/modules",
        "@bazel_python//:pytest_helper",
    ],
)

py_test(
    name = "test_from_torchvision",
    size = "large",
    srcs = ["test_from_torchvision.py"],
    deps = [
        "//sytorch",
        "@bazel_python//:pytest_helper",
    ],
)

py_test(
    name = "test_resnet",
    size = "large",
    srcs = ["test_resnet.py"],
    deps = [
        "//sytorch",
        "@bazel_python//:pytest_helper",
    ],
)
