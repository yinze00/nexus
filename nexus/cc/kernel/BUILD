package(default_visibility = ["//visibility:public"])

load("//bazel:nexus.bzl", "tf_kernel_library")


opkernel_common_deps = [
    "//nexus/turing/common:common",
    "//nexus/cc/common:common",
    "@org_tensorflow//tensorflow/core:framework",
    "@org_tensorflow//tensorflow/core:framework_headers_lib",
]

tf_kernel_library(
    name = "zero_out_op",
    srcs = [
        "zero_out_op.cc",
    ],
    hdrs = ["zero_out_op.h"],
    deps = opkernel_common_deps
)

tf_kernel_library(
    name = "time_two_op",
    srcs = [
        "time_two_op.cc"
    ],
    hdrs = [
        "time_two_op.h"
    ],
    deps = opkernel_common_deps
)

tf_kernel_library(
    name = "time_three_op",
    srcs = [
        "time_three_op.cc"
    ],
    hdrs = [
        "time_three_op.h"
    ],
    deps = opkernel_common_deps
)

tf_kernel_library(
    name = "gather_neighbors",
    prefix = "gather_neighbors",
    deps = opkernel_common_deps, 
    # [
    #     "@org_tensorflow//tensorflow/core:framework",
    #     "@org_tensorflow//tensorflow/core:framework_headers_lib",
    #     "@org_tensorflow//tensorflow/core:lib",
    #     "//nexus/cc/common:common"
    # ]
)

tf_kernel_library(
    name = "gather_embeddings",
    prefix = "gather_embeddings",
    deps = opkernel_common_deps,
    # [
    #     "@org_tensorflow//tensorflow/core:framework",
    #     "@org_tensorflow//tensorflow/core:framework_headers_lib",
    #     "@org_tensorflow//tensorflow/core:lib",
    #     "//nexus/cc/common:common"
    # ]
)

tf_kernel_library(
    name = "request_init",
    prefix = "request_init",
    deps = opkernel_common_deps,
)

tf_kernel_library(
    name = "result_construct",
    prefix = "result_construct",
    deps = opkernel_common_deps,
)


tf_kernel_library(
    name = "indirect_sort_and_topk",
    prefix = "indirect_sort_and_topk",
    deps = opkernel_common_deps,
)


tf_kernel_library(
    name = "gemv",
    prefix = "gemv",
    deps = opkernel_common_deps,
)

cc_library(
    name = "ann_kernel",
    deps = [
        ":time_two_op",
        ":time_three_op",
        ":zero_out_op",
        ":gather_embeddings",
        ":gather_neighbors",
        ":request_init",
        ":result_construct",
        ":gemv",
        ":indirect_sort_and_topk"
    ]
)