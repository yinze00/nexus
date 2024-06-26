#include "graph_context.hh"

#include <vector>

namespace nexus {
namespace turing {

// CTOR
GraphContext::GraphContext(const GraphContextArgs& argv,
                           const GraphRequest* req, GraphResponse* rsp) {}

void GraphContext::run(CallBack callback) {
    std::vector<std::pair<std::string, tensorflow::Tensor>> inputs;

    ErrorInfo info;

    if (unlikely(fill_inputs(inputs))) {
        return;
    }

    callback(info);
}

}  // namespace turing
}  // namespace nexus