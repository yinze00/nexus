#include "graph_biz.hh"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "tensorflow/cc/saved_model/loader.h"

namespace nexus {
namespace turing {

GraphContextArgs GraphBiz::getGraphContextArgs() {
    GraphContextArgs args;

    args.session_resource = getSessionResource();
    args.session = getSession();

    // args.run_options.set_run_id(int64_t value)

    LOG(INFO) << "args done!";

    return args;
}

tensorflow::QueryResourcePtr GraphBiz::prepareQueryResource() {
    auto qry = std::make_shared<tensorflow::QueryResource>();
    return qry;
}

Status GraphBiz::init(const std::string& name) {
    createSessionResource(1024);

    TF_CHECK_OK(loadGraph());

    auto tfss = getTFSession();

    return Status::OK();
}

Status GraphBiz::loadGraph() {
    auto tfss = std::make_shared<TFSession>();
    tfss->graphName = biz_name;

    tensorflow::RunOptions  ropt;
    tensorflow::RunMetadata metas;

    std::string hnsw_model_dir =
        "/home/yinze/dev/zenith/nexus/nexus/data/times_model";
    auto bundle = std::make_shared<tensorflow::SavedModelBundle>();

    TF_CHECK_OK(tensorflow::LoadSavedModel(options, ropt, hnsw_model_dir,
                                           {"serve"}, bundle.get()));
    {  // set session_resource to runtime Device.
        const tensorflow::DeviceMgr* dm = nullptr;

        TF_CHECK_OK(bundle->session->LocalDeviceManager(&dm));
        auto ds = dm->ListDevices();
        LOG(INFO) << "ds.size = " << ds.size();
        if (ds.empty())
            return tensorflow::Status(tensorflow::error::UNAVAILABLE,
                                      "ListDevices Empty!");

        if (session_resource_ == nullptr) {
            LOG(ERROR) << "session_resource_ = nullptr";
        }
        for (auto d : ds) {
            auto ld = dynamic_cast<tensorflow::LocalDevice*>(d);
            ld->session_resource_ = getSessionResource();
        }
    }

    {
        // run grpah if necessary...
        // tensorflow::Tensor tensor(tensorflow::DT_FLOAT,
        //                           tensorflow::TensorShape({2}));
        // tensor.vec<float>()(0) = 20.f;
        // tensor.vec<float>()(1) = 6000.f;
        // std::vector<tensorflow::Tensor> outputs;
        // auto status =
        //     bundle->session->Run({{"user_emb", tensor}}, {"ee"}, {},
        //     &outputs);
        // // LOG(INFO) << "output.size = " << outputs.size() << "\n"
        // //           << bundle->meta_graph_def.Utf8DebugString();

        // LOG(INFO) << status.ToString();

        // if (status.ok()) {
        //     for (const auto& record : outputs) {
        //         LOG(INFO) << record.DebugString(100);
        //     }
        // }
    }

    tfss->session = std::move(bundle->session);

    sessions_.emplace(biz_name, tfss);
    return Status::OK();
}

std::shared_ptr<tensorflow::Session> GraphBiz::getSession(
    const std::string& model) const {
    LOG(INFO) << model;
    auto tfss = getTFSession(model);
    LOG(INFO) << "After get model";
    return tfss->session;
}

tensorflow::SessionResourcePtr GraphBiz::createSessionResource(uint32_t count) {
    return session_resource_ =
               std::make_shared<tensorflow::SessionResource>(count);
}

// const tensorflow::RunOptions& GraphBiz::getRunOptions() const {
//     return tensorflow::RunOptions();
// }

}  // namespace turing
}  // namespace nexus