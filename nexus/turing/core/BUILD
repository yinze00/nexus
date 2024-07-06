package(default_visibility = ["//visibility:public"])

cc_library(
    name = "core",
    hdrs = glob(["*.hh"]),
    srcs = glob(["*.cc"]),
    deps = [
        "//nexus/turing/common:common",
        "@org_tensorflow//tensorflow/core:direct_session",
        "@org_tensorflow//tensorflow/cc/saved_model:loader"
    ]
)