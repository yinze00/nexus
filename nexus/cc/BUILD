package(default_visibility = ["//visibility:public"])

load("//bazel:nexus.bzl", "tf_gen_op_libs", "tf_gen_op_wrappers_cc", "tf_gen_op_wrapper_cc")

tf_gen_op_libs(
    op_lib_names = [        # filenames under `ops/`
        "nexus_ops",
        "time_two_ops",
        "hnsw_ops",
    ],
    deps = [
        "@org_tensorflow//:tensorflow_cc",
    ],
)

tf_gen_op_wrappers_cc(
    name = "nexus_ops",
    api_def_srcs = ["//nexus/cc/api_def:api_def"],
    op_lib_names = [
        "nexus_ops",
        "time_two_ops",
        "hnsw_ops",
    ],
)

cc_library(
    name = "ops",
    visibility = ["//visibility:public"],
    deps = [
        "time_two_ops_op_lib",
        "nexus_ops_op_lib"
    ],
    alwayslink = 1,
)

