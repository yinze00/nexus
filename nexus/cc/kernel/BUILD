package(default_visibility = ["//visibility:public"])

load("//bazel:nexus.bzl", "tf_kernel_library")

tf_kernel_library(
    name = "zero_out_op",
    srcs = [
        "zero_out_op.cc",
    ],
    hdrs = ["zero_out_op.h"],
    deps = [
        "@org_tensorflow//tensorflow/core:framework",
        "@org_tensorflow//tensorflow/core:framework_headers_lib",
    ]
)

tf_kernel_library(
    name = "time_two_op",
    srcs = [
        "time_two_op.cc"
    ],
    hdrs = [
        "time_two_op.h"
    ],
    deps = [
        "@org_tensorflow//tensorflow/core:framework",
        "@org_tensorflow//tensorflow/core:framework_headers_lib",
    ]
)

tf_kernel_library(
    name = "time_three_op",
    srcs = [
        "time_three_op.cc"
    ],
    hdrs = [
        "time_three_op.h"
    ],
    deps = [
        "@org_tensorflow//tensorflow/core:framework",
        "@org_tensorflow//tensorflow/core:framework_headers_lib",
    ]
)

tf_kernel_library(
    name = "gather_neighbors",
    prefix = "gather_neighbors",
    deps = [
        "@org_tensorflow//tensorflow/core:framework",
        "@org_tensorflow//tensorflow/core:framework_headers_lib",
        "@org_tensorflow//tensorflow/core:lib",
        "//nexus/cc/common:common"
    ]
)

tf_kernel_library(
    name = "gather_embeddings",
    prefix = "gather_embeddings",
    # srcs = ["gather_embeddings_kernel.cc"],
    # hdrs = ["gather_embeddings_kernel.hh"],
    deps = [
        "@org_tensorflow//tensorflow/core:framework",
        "@org_tensorflow//tensorflow/core:framework_headers_lib",
        "@org_tensorflow//tensorflow/core:lib",
        "//nexus/cc/common:common"
    ]
)

cc_library(
    name = "ann_kernel",
    deps = [
        ":time_two_op",
        ":time_three_op",
        ":zero_out_op",
        ":gather_embeddings",
        ":gather_neighbors",
    ]
)