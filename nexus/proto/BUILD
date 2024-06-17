package(default_visibility = ["//visibility:public"])

load("@rules_cc//cc:defs.bzl", "cc_binary", "cc_library", native_cc_proto_library = "cc_proto_library")
load("@rules_proto//proto:defs.bzl", "proto_lang_toolchain", "proto_library")
load(
    "@org_tensorflow//tensorflow/core/platform:default/build_config.bzl", 
    "tf_proto_library", 
)
# load(
#     "@org_tensorflow//tensorflow:tensorflow.bzl",
#     "tf_generate_proto_text_sources",
# )
load("//bazel:nexus.bzl", "tf_generate_proto_text_sources")

GRAPH_PROTOS = [
    "error_code.proto",
    "run_graph.proto"
]

####################################### From Brpc #######################################
proto_library(
    name = "run_graph_proto",
    srcs = GRAPH_PROTOS,
    visibility = ["//visibility:public"],
    deps = [
        "@com_google_protobuf//:descriptor_proto",
    ]
)

cc_proto_library(
    name = "run_graph_proto_lib",
    visibility = ["//visibility:public"],
    deps = [":run_graph_proto"]
)
####################################### From Brpc #######################################


####################################### From Tensorflow #######################################


tf_proto_library(
    name = "graph_proto",
    srcs = glob(["*.proto"]),
    cc_api_version = 2,
    visibility = ["//visibility:public"],
    make_default_target_header_only = True,
    protodeps = [
        "@org_tensorflow//tensorflow/core:protos_all",
    ],
)

tf_generate_proto_text_sources(
    name = "protos_graph_text",
    srcs = GRAPH_PROTOS,
    srcs_relative_dir = "nexus/proto/",
    visibility = ["//visibility:public"],
    deps = [

    ]
)

cc_library(
    name = "proto_text",
    hdrs = [
        "protos_graph_text_hdrs"
    ],
    deps = [
        ":graph_proto_cc"
    ],
)

####################################### From Tensorflow #######################################