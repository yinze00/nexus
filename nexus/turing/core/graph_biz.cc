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

    args.sesson = getSession();

    // args.run_options.set_run_id(int64_t value)

    return args;
}

Status GraphBiz::init(const std::string& name) {
    TF_CHECK_OK(loadGraph());

    auto tfss = getTFSession();

    return Status::OK();
}

Status GraphBiz::loadGraph() {
    tensorflow::RunOptions ropt;
    std::string hnsw_model_dir =
        "/home/yinze/dev/zenith/nexus/nexus/data/times_model";
    auto bundle = std::make_shared<tensorflow::SavedModelBundle>();

    TF_CHECK_OK(tensorflow::LoadSavedModel(options, ropt, hnsw_model_dir,
                                           {"serve"}, bundle.get()));

    auto tfss = std::make_shared<TFSession>();
    tfss->graphName = biz_name;
    tfss->session = std::move(bundle->session);

    const tensorflow::DeviceMgr* dm = nullptr;

    TF_CHECK_OK(tfss->session->LocalDeviceManager(&dm));
    auto ds = dm->ListDevices();
    if (!ds.empty())
        return tensorflow::Status(tensorflow::error::UNAVAILABLE,
                                  "ListDevices Empty!");

    auto ld = dynamic_cast<tensorflow::LocalDevice*>(ds.front());
    ld->session_resource_ = getSessionResource();

    sessions_.emplace(biz_name, tfss);
    return Status::OK();
}

std::shared_ptr<tensorflow::Session> GraphBiz::getSession(
    const std::string& model) const {
    auto tfss = getTFSession(model);
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