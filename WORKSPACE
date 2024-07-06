workspace(name = "nexus")

load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive", "http_file")
load("@bazel_tools//tools/build_defs/repo:git.bzl", "git_repository")


############################################### bazel commons ###############################################
git_repository(
    name = "rules_proto",
    commit = "cfdc2fa31879c0aebe31ce7702b1a9c8a4be02d2",
    remote = "https://github.com/bazelbuild/rules_proto.git",
)
############################################### bazel commons ###############################################


############################################### apache_brpc ###############################################
load("//third_party/brpc:brpc_workspace.bzl", "brpc_workspace")
brpc_workspace();
############################################### apache_brpc ###############################################

http_archive(
    name = "gtest",
    urls = ["https://github.com/google/googletest/archive/refs/tags/release-1.11.0.tar.gz"],
    strip_prefix = "googletest-release-1.11.0",
)

############################################### org_tensorflow ###############################################
# new_local_repository(
#     name = "org_tensorflow",
#     build_file = "/home/yinze/libtf1.15/BUILD",
#     path = "/home/yinze/libtf1.15",
# )

new_local_repository(
    name = "org_tensorflow",
    path = "/home/yinze/dev/zenith-turing/tensorflow-plus",
    build_file = "/home/yinze/dev/zenith-turing/tensorflow-plus/BUILD",
)

http_archive(
    name = "io_bazel_rules_closure",
    sha256 = "ddce3b3a3909f99b28b25071c40b7fec7e2e1d1d1a4b2e933f3082aa99517105",
    strip_prefix = "rules_closure-316e6133888bfc39fb860a4f1a31cfcbae485aef",
    urls = [
        "https://mirror.bazel.build/github.com/bazelbuild/rules_closure/archive/316e6133888bfc39fb860a4f1a31cfcbae485aef.tar.gz",
        "https://github.com/bazelbuild/rules_closure/archive/316e6133888bfc39fb860a4f1a31cfcbae485aef.tar.gz",  # 2019-03-21
    ],
)

http_archive(
    name = "bazel_skylib",
    sha256 = "2c62d8cd4ab1e65c08647eb4afe38f51591f43f7f0885e7769832fa137633dcb",
    strip_prefix = "bazel-skylib-0.7.0",
    urls = ["https://github.com/bazelbuild/bazel-skylib/archive/0.7.0.tar.gz"],
)


load("@org_tensorflow//third_party:repo.bzl", "tf_http_archive")
load("@org_tensorflow//tensorflow:workspace.bzl", "tf_workspace")
load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

tf_workspace(path_prefix = "", tf_repo_name = "org_tensorflow")
############################################### org_tensorflow ###############################################