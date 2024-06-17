package(default_visibility = ["//visibility:public"])

load("//bazel:nexus.bzl", "tf_kernel_library")

cc_library(
    name = "common",
    hdrs = glob(["*.hh"]) + glob(["*.h"]),
    srcs = [    
        "embedding.cc",
        "graph.cc",
        "index.cc",
        "IndexCommonSingleton.cc"
    ],
    copts = [
        "-std=c++17"
    ],
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