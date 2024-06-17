
def clean_dep(dep):
    return str(Label(dep))

def tf_gpu_library(deps = None, cuda_deps = None, copts = [], **kwargs):
    if not deps:
        deps = []
    if not cuda_deps:
        cuda_deps = []

    kwargs["features"] = kwargs.get("features", []) + ["-use_header_modules"]
    native.cc_library(
        deps = deps,
        # copts = (copts + if_cuda(["-DGOOGLE_CUDA=1"]) + if_rocm(["-DTENSORFLOW_USE_ROCM=1"]) + if_mkl(["-DINTEL_MKL=1"]) + if_mkl_open_source_only(["-DINTEL_MKL_DNN_ONLY"]) + if_enable_mkl(["-DENABLE_MKL"]) + if_tensorrt(["-DGOOGLE_TENSORRT=1"])),
        **kwargs
    )

def tf_kernel_library(
        name,
        prefix = None,
        srcs = None,
        gpu_srcs = None,
        hdrs = None,
        deps = None,
        alwayslink = 1,
        copts = None,
        gpu_copts = None,
        is_external = False,
        **kwargs):
    """A rule to build a TensorFlow OpKernel.

      May either specify srcs/hdrs or prefix.  Similar to tf_gpu_library,
      but with alwayslink=1 by default.  If prefix is specified:
        * prefix*.cc (except *.cu.cc) is added to srcs
        * prefix*.h (except *.cu.h) is added to hdrs
        * prefix*.cu.cc and prefix*.h (including *.cu.h) are added to gpu_srcs.
      With the exception that test files are excluded.
      For example, with prefix = "cast_op",
        * srcs = ["cast_op.cc"]
        * hdrs = ["cast_op.h"]
        * gpu_srcs = ["cast_op_gpu.cu.cc", "cast_op.h"]
        * "cast_op_test.cc" is excluded
      With prefix = "cwise_op"
        * srcs = ["cwise_op_abs.cc", ..., "cwise_op_tanh.cc"],
        * hdrs = ["cwise_ops.h", "cwise_ops_common.h"],
        * gpu_srcs = ["cwise_op_gpu_abs.cu.cc", ..., "cwise_op_gpu_tanh.cu.cc",
                      "cwise_ops.h", "cwise_ops_common.h",
                      "cwise_ops_gpu_common.cu.h"]
        * "cwise_ops_test.cc" is excluded
      """
    if not srcs:
        srcs = []
    if not hdrs:
        hdrs = []
    if not deps:
        deps = []
    if not copts:
        copts = []
    if not gpu_copts:
        gpu_copts = []
    textual_hdrs = []
    # copts = copts + tf_copts(is_external = is_external)

    # Override EIGEN_STRONG_INLINE to inline when
    # --define=override_eigen_strong_inline=true to avoid long compiling time.
    # See https://github.com/tensorflow/tensorflow/issues/10521
    # copts = copts + if_override_eigen_strong_inline(["/DEIGEN_STRONG_INLINE=inline"])
    if prefix:
        if native.glob([prefix + "*.cu.cc"], exclude = ["*test*"]):
            if not gpu_srcs:
                gpu_srcs = []
            gpu_srcs = gpu_srcs + native.glob(
                [prefix + "*.cu.cc", prefix + "*.h"],
                exclude = [prefix + "*test*"],
            )
        srcs = srcs + native.glob(
            [prefix + "*.cc"],
            exclude = [prefix + "*test*", prefix + "*.cu.cc"],
        )
        hdrs = hdrs + native.glob(
            [prefix + "*.h", prefix + "*.hh"],
            exclude = [prefix + "*test*", prefix + "*.cu.h", prefix + "*impl.h"],
        )
        textual_hdrs = native.glob(
            [prefix + "*impl.h"],
            exclude = [prefix + "*test*", prefix + "*.cu.h"],
        )

    tf_gpu_library(
        name = name,
        srcs = srcs,
        hdrs = hdrs,
        textual_hdrs = textual_hdrs,
        copts = copts,
        cuda_deps = [],
        linkstatic = 1,  # Needed since alwayslink is broken in bazel b/27630669
        alwayslink = alwayslink,
        deps = deps,
        **kwargs
    )

    # TODO(gunan): CUDA dependency not clear here. Fix it.
    # tf_cc_shared_object(
    #     name = "libtfkernel_%s.so" % name,
    #     srcs = srcs + hdrs,
    #     copts = copts,
    #     tags = ["manual", "notap"],
    #     deps = deps,
    # )


def tf_gen_op_libs(op_lib_names, deps = None, is_external = True):
    # Make library out of each op so it can also be used to generate wrappers
    # for various languages.
    if not deps:
        deps = []
    for n in op_lib_names:
        native.cc_library(
            name = n + "_op_lib",
            # copts = 
            srcs = ["ops/" + n + ".cc"],
            deps = deps + [clean_dep("@org_tensorflow//tensorflow/core:framework")],
            visibility = ["//visibility:public"],
            alwayslink = 1,
            linkstatic = 1,
        )

def tf_gen_op_wrapper_cc(
        name,
        out_ops_file,
        pkg = "",
        op_gen = clean_dep("@org_tensorflow//tensorflow/cc:cc_op_gen_main"),
        deps = None,
        include_internal_ops = 0,
        # ApiDefs will be loaded in the order specified in this list.
        api_def_srcs = []):
    # Construct an op generator binary for these ops.
    tool = out_ops_file + "_gen_cc"
    if deps == None:
        deps = [pkg + ":" + name + "_op_lib"]
    # print(tool)
    native.cc_binary(
        name = tool,
        copts = [],
        linkopts = ["-lm", "-Wl,-ldl"],
        linkstatic = 1,  # Faster to link this one-time-use binary dynamically
        deps = [op_gen] + deps,
    )

    srcs = api_def_srcs[:]

    if not api_def_srcs:
        api_def_args_str = ","
    else:
        api_def_args = []
        for api_def_src in api_def_srcs:
            # Add directory of the first ApiDef source to args.
            # We are assuming all ApiDefs in a single api_def_src are in the
            # same directory.
            api_def_args.append(
                " $$(dirname $$(echo $(locations " + api_def_src +
                ") | cut -d\" \" -f1))",
            )
        api_def_args_str = ",".join(api_def_args)

    print("++++++ 1 ++++++ ", tool)
    # print("+++ ", op_gen)
    # print("++++++ 2 ++++++ ", include_internal_ops)

    # ccmd = ("$(location :" + tool + ") $(location :" + out_ops_file + ".h) " +
    #            "$(location :" + out_ops_file + ".cc) " +
    #            str(include_internal_ops) + " " + api_def_args_str)

    # print("++++++ 3 ++++++ ", srcs)

    # print("++++++ 4 ++++++ ", ccmd)
    
    native.genrule(
        name = name + "_genrule",
        outs = [
            out_ops_file + ".h",
            out_ops_file + ".cc",
            out_ops_file + "_internal.h",
            out_ops_file + "_internal.cc",
        ],
        srcs = srcs,
        tools = [":" + tool] + [clean_dep("@org_tensorflow//tensorflow:tensorflow_framework")],
        cmd = ("$(location :" + tool + ") $(location :" + out_ops_file + ".h) " +
               "$(location :" + out_ops_file + ".cc) " +
               str(include_internal_ops) + " " + api_def_args_str),
    )



def tf_gen_op_wrappers_cc(
        name,
        op_lib_names = [],
        other_srcs = [],
        other_hdrs = [],
        other_srcs_internal = [],
        other_hdrs_internal = [],
        pkg = "",
        deps = [
            # clean_dep("@org_tensorflow//tensorflow:tensorflow_cc")
            clean_dep("@org_tensorflow//tensorflow/cc:ops"),
            clean_dep("@org_tensorflow//tensorflow/cc:scope"),
            clean_dep("@org_tensorflow//tensorflow/cc:const_op"),
        ],
        deps_internal = [],
        op_gen = clean_dep("@org_tensorflow//tensorflow/cc:cc_op_gen_main"),
        include_internal_ops = 0,
        visibility = None,
        # ApiDefs will be loaded in the order specified in this list.
        api_def_srcs = [],
        # Any extra dependencies that the wrapper generator might need.
        extra_gen_deps = []):
    subsrcs = other_srcs[:]
    subhdrs = other_hdrs[:]
    internalsrcs = other_srcs_internal[:]
    internalhdrs = other_hdrs_internal[:]
    for n in op_lib_names:
        tf_gen_op_wrapper_cc(
            n,
            "annops/" + n,
            api_def_srcs = api_def_srcs,
            include_internal_ops = include_internal_ops,
            op_gen = op_gen,
            pkg = pkg,
            deps = [pkg + ":" + n + "_op_lib"] + extra_gen_deps,
        )
        subsrcs += ["annops/" + n + ".cc"]
        subhdrs += ["annops/" + n + ".h"]
        internalsrcs += ["annops/" + n + "_internal.cc"]
        internalhdrs += ["annops/" + n + "_internal.h"]

    # print("SRCS = ", subsrcs)
    # print("HDRS = ", subhdrs)
    native.cc_library(
        name = name,
        srcs = subsrcs,
        hdrs = subhdrs,
        deps = deps + ["@org_tensorflow//tensorflow/core:framework"],
        # copts = tf_copts(),
        alwayslink = 1,
        visibility = visibility,
    )
    native.cc_library(
        name = name + "_internal",
        srcs = internalsrcs,
        hdrs = internalhdrs,
        deps = deps + deps_internal + ["@org_tensorflow//tensorflow/core:framework"],
        # copts = tf_copts(),
        alwayslink = 1,
        visibility = visibility,
    )

# Generates a Python library target wrapping the ops registered in "deps".
#


def tf_generate_proto_text_sources(name, srcs_relative_dir, srcs, protodeps = [], deps = [], visibility = None):
    out_hdrs = (
        [
            p.replace(".proto", ".pb_text.h")
            for p in srcs
        ] + [p.replace(".proto", ".pb_text-impl.h") for p in srcs]
    )
    out_srcs = [p.replace(".proto", ".pb_text.cc") for p in srcs]
    native.genrule(
        name = name + "_srcs",
        srcs = srcs + protodeps + [clean_dep("@org_tensorflow//tensorflow/tools/proto_text:placeholder.txt")],
        outs = out_hdrs + out_srcs,
        visibility = visibility,
        cmd =
            "$(location @org_tensorflow//tensorflow/tools/proto_text:gen_proto_text_functions) " +
            "$(@D) " + srcs_relative_dir + " $(SRCS)",
        tools = [
            clean_dep("@org_tensorflow//tensorflow/tools/proto_text:gen_proto_text_functions"),
        ],
    )

    native.filegroup(
        name = name + "_hdrs",
        srcs = out_hdrs,
        visibility = visibility,
    )

    native.cc_library(
        name = name,
        srcs = out_srcs,
        hdrs = out_hdrs,
        visibility = visibility,
        deps = deps,
    )