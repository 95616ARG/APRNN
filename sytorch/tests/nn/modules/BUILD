py_library(
    name = "modules",
    srcs = glob(["*.py"]),
    visibility = ["//sytorch/tests:__subpackages__"],
    deps = [
        "//sytorch",
        "@bazel_python//:pytest_helper",
    ],
)

py_test(
    name = "test_module",
    size = "small",
    srcs = ["test_module.py"],
    deps = [
        "//sytorch",
        "@bazel_python//:pytest_helper",
    ],
)

py_test(
    name = "test_module_activation_pattern",
    size = "small",
    srcs = ["test_module_activation_pattern.py"],
    deps = [
        "//sytorch",
        "@bazel_python//:pytest_helper",
    ],
)

py_test(
    name = "test_linear",
    size = "medium",
    srcs = ["test_linear.py"],
    visibility = ["//sytorch/tests:__subpackages__"],
    deps = [
        "//sytorch",
        "@bazel_python//:pytest_helper",
    ],
)

py_test(
    name = "test_conv",
    size = "medium",
    srcs = ["test_conv.py"],
    visibility = ["//sytorch/tests:__subpackages__"],
    deps = [
        "//sytorch",
        "@bazel_python//:pytest_helper",
    ],
)

py_test(
    name = "test_activation",
    size = "medium",
    srcs = ["test_activation.py"],
    visibility = ["//sytorch/tests:__subpackages__"],
    deps = [
        "//sytorch",
        "@bazel_python//:pytest_helper",
    ],
)

py_test(
    name = "test_pooling",
    size = "medium",
    srcs = ["test_pooling.py"],
    visibility = ["//sytorch/tests:__subpackages__"],
    deps = [
        "//sytorch",
        "@bazel_python//:pytest_helper",
    ],
)

py_test(
    name = "test_batchnorm",
    size = "medium",
    srcs = ["test_batchnorm.py"],
    visibility = ["//sytorch/tests:__subpackages__"],
    deps = [
        "//sytorch",
        "@bazel_python//:pytest_helper",
    ],
)

py_test(
    name = "test_dropout",
    size = "medium",
    srcs = ["test_dropout.py"],
    visibility = ["//sytorch/tests:__subpackages__"],
    deps = [
        "//sytorch",
        "@bazel_python//:pytest_helper",
    ],
)

py_test(
    name = "test_container",
    size = "medium",
    srcs = ["test_container.py"],
    visibility = ["//sytorch/tests:__subpackages__"],
    deps = [
        "//sytorch",
        "@bazel_python//:pytest_helper",
    ],
)
