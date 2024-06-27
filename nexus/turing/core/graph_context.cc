#include "nexus/turing/core/graph_context.hh"

#include <vector>

#include "nexus/turing/common/scope_deleter.hh"

namespace nexus {
namespace turing {

// CTOR
GraphContext::GraphContext(const GraphContextArgs& argv,
                           const GraphRequest* request, GraphResponse* response)
    : req(request), rsp(response) {
    run_id = argv.run_options.run_id();
    session_resource = argv.session_resource;
    session = argv.session;
}

/**
 * 1. getBizByName
 * 2. parseRequest
 * 3. RunOptions CTOR
 * 4. prepareQueryResource
 * 5. addQueryResource
 * 6. getPlaceholders
 * 7. fillInputs
 * 8. getSession
 * 9. setErrorInfo
 * 10. afterSearch
 * 11. dumbass
 */
void GraphContext::run(CallBack callback) {
    nexus::common::ScopeDeleter<GraphContext> deleter(this);

    ErrorInfo info;

    if (!parseRequest()) {
        callback(info);
        LOG(ERROR) << "Failed to parse Request...";
        return;
    }

    // prepareQueryResource(std::shared_ptr<GraphBiz> biz,
    //                      tensorflow::RunOptions & opts)

    std::vector<std::pair<std::string, tensorflow::Tensor>> inputs;

    LOG(INFO) << "run_id: " << run_id;

    if (unlikely(!fill_inputs(inputs))) {
        return;
    }

    callback(info);
}

bool GraphContext::fill_inputs(
    std::vector<std::pair<std::string, tensorflow::Tensor>>& inputs) {
    return true;
}

}  // namespace turing
}  // namespace nexus