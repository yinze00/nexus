package(default_visibility = ["//visibility:public"])

load("//bazel:nexus.bzl", "tf_gen_op_libs", "tf_gen_op_wrappers_cc", "tf_gen_op_wrapper_cc")

cc_library(
    name = "nexus",
    deps = [
        "//nexus/cc/kernel:ann_kernel",
        "//nexus/cc:nexus_ops",
        "//nexus/cc:ops",
        "//nexus/cc/common",
        "//nexus/turing",
        "@org_tensorflow//tensorflow/core:framework"
    ]
)