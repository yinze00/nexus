#include "graph_context.hh"

#include <vector>

namespace nexus {
namespace turing {

// CTOR
GraphContext::GraphContext(const GraphContextArgs& argv,
                           const GraphRequest* req, GraphResponse* rsp) {
    run_id = argv.run_options.run_id();
}

void GraphContext::run(CallBack callback) {
    std::vector<std::pair<std::string, tensorflow::Tensor>> inputs;

    ErrorInfo info;

    LOG(INFO) << "run_id: " << run_id;

    if (unlikely(fill_inputs(inputs))) {
        // return;
    }

    callback(info);
}

bool GraphContext::fill_inputs(
    std::vector<std::pair<std::string, tensorflow::Tensor>>& inputs) {
    return true;
}

}  // namespace turing
}  // namespace nexus