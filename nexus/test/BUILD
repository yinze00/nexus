cc_binary(
    name = "load_and_run",
    srcs = ["load_and_run.cc"],
    copts = [
        "-g -O1",
    ],
    linkopts = [
        "-lgflags"
    ],
    deps = [
        "//nexus:nexus",
        "//nexus/cc:time_two_ops_op_lib",
        "@org_tensorflow//tensorflow/core:framework",
        "@org_tensorflow//tensorflow/cc:client_session",
        "@org_tensorflow//tensorflow/cc:cc_ops",
        "@org_tensorflow//tensorflow/cc/saved_model:loader",
    ],
)