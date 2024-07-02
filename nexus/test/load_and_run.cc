#include "tensorflow/cc/client/client_session.h"
#include "tensorflow/core/framework/api_def.pb.h"
#include "tensorflow/core/framework/op.h"
// #include "tensorflow/core/framework/op_gen_lib.h"
#include <cstddef>
#include <memory>

#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/lib/io/path.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/public/session.h"

// #include "annop/kernel/ZeroOutOp.h"
#include <gflags/gflags.h>
// #include <tensorflow/cc/ops/math_ops.h>
#include <iostream>
#include <vector>

#include "nexus/cc/annops/nexus_ops.h"
#include "nexus/cc/annops/time_two_ops.h"
#include "nexus/turing/common/op_util.hh"
#include "tensorflow/cc/saved_model/loader.h"
#include "tensorflow/core/common_runtime/device_mgr.h"
#include "tensorflow/core/common_runtime/local_device.h"

using namespace std;
using namespace tensorflow;

// DEFINE_string(sdf,"sdf");
DEFINE_string(modelpath, "nexus/data/times_model", "x2 x3 models path");

int main(int argc, char** argv) {
    gflags::ParseCommandLineFlags(&argc, &argv, true);
    // printf("All registered ops:\n%s\n",
    //        tensorflow::OpRegistry::Global()->DebugString(false).c_str());
    auto bundle = std::make_shared<SavedModelBundle>();

    tensorflow::SessionOptions sessionOptions;
    tensorflow::RunOptions     runOptions;

    LOG(INFO) << "Start to load model from " << FLAGS_modelpath;

    TF_CHECK_OK(tensorflow::LoadSavedModel(
        sessionOptions, runOptions, FLAGS_modelpath, {"serve"}, bundle.get()));

    const DeviceMgr* dm = nullptr;

    TF_CHECK_OK(bundle->session->LocalDeviceManager(&dm));

    Device* d = nullptr;
    TF_CHECK_OK(dm->LookupDevice("CPU:0", &d));

    auto ld = dynamic_cast<LocalDevice*>(d);

    ld->session_resource_.reset(new SessionResource(10));

    // LOG(INFO) << "\n" <<  bundle->meta_graph_def.DebugString();

    tensorflow::Tensor tensor(tensorflow::DT_FLOAT,
                              tensorflow::TensorShape({2}));
    tensor.vec<float>()(0) = 20.f;
    tensor.vec<float>()(1) = 6000.f;

    std::vector<std::pair<std::string, tensorflow::Tensor>> feedInputs = {
        {"user_emb", tensor}};
    std::vector<std::string> fetches = {"recall_results"};

    std::vector<tensorflow::Tensor> outputs;

    auto status = bundle->session->Run(feedInputs, fetches, {"done"}, &outputs);

    // ... and print out it's predictions.
    for (const auto& record : outputs) {
        LOG(INFO) << record.DebugString(100);
    }

    LOG(INFO) << "Session Run for the 2rd time";
    status = bundle->session->Run(feedInputs, fetches,{"done"}, &outputs);

    TF_CHECK_OK(status);

    // ... and print out it's predictions.
    for (const auto& record : outputs) {
        LOG(INFO) << record.DebugString(100);
    }

    return 0;
}