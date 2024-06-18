#include "tensorflow/cc/client/client_session.h"
#include "tensorflow/core/framework/api_def.pb.h"
#include "tensorflow/core/framework/op.h"
// #include "tensorflow/core/framework/op_gen_lib.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/public/session.h"

#include <memory>

#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/lib/io/path.h"
// #include "annop/kernel/ZeroOutOp.h"
#include <gflags/gflags.h>
// #include <tensorflow/cc/ops/math_ops.h>
#include "tensorflow/cc/saved_model/loader.h"

#include <iostream>

#include "nexus/cc/annops/nexus_ops.h"
#include "nexus/cc/annops/time_two_ops.h"

using namespace std;
using namespace tensorflow;

// DEFINE_string(sdf,"sdf");
DEFINE_string(modelpath, "nexus/data/times_model", "x2 x3 models path");

int main(int argc, char** argv) {
    // ops::TimeTwo tt;
    gflags::ParseCommandLineFlags(&argc, &argv, true);
    // printf("All registered ops:\n%s\n",
    //        tensorflow::OpRegistry::Global()->DebugString(false).c_str());
    auto bundle = std::make_shared<SavedModelBundle>();

    // Create dummy options.
    tensorflow::SessionOptions sessionOptions;
    tensorflow::RunOptions runOptions;

    LOG(INFO) << "Start to load model from " << FLAGS_modelpath;

    // Load the model bundle.
    const auto loadResult = tensorflow::LoadSavedModel(
        sessionOptions, runOptions, FLAGS_modelpath, {"serve"}, bundle.get());

    TF_CHECK_OK(loadResult);

    LOG(INFO) << "\n" <<  bundle->meta_graph_def.DebugString();

    tensorflow::Tensor tensor(tensorflow::DT_FLOAT,
                              tensorflow::TensorShape({2}));
    tensor.vec<float>()(0) = 20.f;
    tensor.vec<float>()(1) = 6000.f;

    // Link the data with some tags so tensorflow know where to put those data
    // entries.
    std::vector<std::pair<std::string, tensorflow::Tensor>> feedInputs = {
        {"user_emb", tensor}};
    std::vector<std::string> fetches = {"ee"};

    // We need to store the results somewhere.
    std::vector<tensorflow::Tensor> outputs;

    // Let's run the model...
    // bundle->session
    auto status = bundle->session->Run(feedInputs, fetches, {}, &outputs);

    LOG(INFO) << "Session Run for the 2rd time";
    status = bundle->session->Run(feedInputs, fetches, {}, &outputs);


    TF_CHECK_OK(status);

    // ... and print out it's predictions.
    for (const auto& record : outputs) {
        LOG(INFO) << record.DebugString(100);
    }
    return 0;
}