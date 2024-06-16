workspace(name = "nexus")

load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive", "http_file")
load("@bazel_tools//tools/build_defs/repo:git.bzl", "git_repository")

http_archive(
    name = "gtest",
    urls = ["https://github.com/google/googletest/archive/refs/tags/release-1.11.0.tar.gz"],
    strip_prefix = "googletest-release-1.11.0",
)

# new_local_repository(
#     name = "org_tensorflow",
#     build_file = "/home/yinze/libtf1.15/BUILD",
#     path = "/home/yinze/libtf1.15",
# )

new_local_repository(
    name = "org_tensorflow",
    path = "/home/yinze/dev/zenith-turing/tensorflow-plus"
)