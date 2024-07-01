package(default_visibility = ["//visibility:public"])

cc_library(
    name = "common",
    hdrs = glob(["*.hh"]),
    srcs = glob(["*.cc"]),
    deps = [
        "//nexus/turing/proto:graph_proto_cc",
        "//nexus/utils",
        "//nexus/cc/common",
        "@org_tensorflow//tensorflow/core:framework",
        "@org_tensorflow//tensorflow/core:core_cpu_headers_lib",
    ]
)