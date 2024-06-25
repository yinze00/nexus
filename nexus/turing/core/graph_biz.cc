#include "graph_biz.hh"

#include <cstdint>
#include <memory>
#include <string>

#include "tensorflow/cc/saved_model/loader.h"

namespace nexus {
namespace turing {

Status GraphBiz::init(const std::string& name) {
    TF_CHECK_OK(loadGraph());

    auto tfss = getTFSession();

    return Status::OK();
}

Status GraphBiz::loadGraph() {
    tensorflow::RunOptions ropt;
    std::string hnsw_model_dir;
    auto bundle = std::make_shared<tensorflow::SavedModelBundle>();

    TF_CHECK_OK(tensorflow::LoadSavedModel(options, ropt, hnsw_model_dir,
                                           {"serve"}, bundle.get()));

    TFSession tfss{.session = std::move(bundle->session),
                   .graphName = biz_name};

    sessions_.emplace(biz_name, std::move(tfss));

    return Status::OK();
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