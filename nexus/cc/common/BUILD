package(default_visibility = ["//visibility:public"])

load("//bazel:nexus.bzl", "tf_kernel_library")

cc_library(
    name = "common",
    hdrs = glob(["*.hh"]) + glob(["*.h"]),
    srcs = [    
        "embedding.cc",
        "graph.cc",
        "index.cc",
        "ANNIndexHolder.cc"
    ],
    copts = [
        "-std=c++17",
    ],
    linkopts = ["-lfaiss", "-lopenblas", "-fopenmp"],
    deps = [
        "@org_tensorflow//tensorflow/core:framework"
    ]
)

cc_test(
    name = "common_test",
    srcs = [
        "graph_test.cc",
        "index_test.cc",
    ],
    deps = [
        ":common",
        "@gtest//:gtest",
        "@gtest//:gtest_main"
    ]
)

cc_test(
    name = "adaptor_test",
    srcs = [
        "adaptor_test.cc",
    ],
    deps = [
        ":common",
        "@gtest//:gtest",
        "@gtest//:gtest_main"
    ]
)

cc_test(
    name = "heap_test",
    srcs = [
        "heap_test.cc",
    ],
    deps = [
        ":common",
        "@gtest//:gtest",
        "@gtest//:gtest_main"
    ]
)