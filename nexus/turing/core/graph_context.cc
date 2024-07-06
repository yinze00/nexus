#include "nexus/turing/core/graph_context.hh"

#include <cstddef>
#include <string>
#include <vector>

#include "nexus/turing/common/scope_deleter.hh"

namespace nexus {
namespace turing {

// CTOR
GraphContext::GraphContext(const GraphContextArgs& argv,
                           const GraphRequest* request, GraphResponse* response)
    : req(request), rsp(response) {
    VLOG(1) << argv.run_options.DebugString();
    run_id           = argv.run_options.run_id();
    session_resource = argv.session_resource;
    session          = argv.session;
    run_options      = argv.run_options;
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
 */
void GraphContext::run(CallBack callback) {
    nexus::common::ScopeDeleter<GraphContext> deleter(this);

    ErrorInfo info;

    if (!parseRequest()) {
        callback(info);
        LOG(ERROR) << "Failed to parse Request...";
        return;
    }

    std::vector<std::pair<std::string, tensorflow::Tensor>> inputs;

    if (unlikely(!fill_inputs(inputs))) return;

    std::vector<std::string> fetches(req->graph_info().fetches().begin(),
                                     req->graph_info().fetches().end());

    std::vector<std::string> targets(req->graph_info().targets().begin(),
                                     req->graph_info().targets().end());
    LOG(INFO) << "+++++++++ Before Session.Run +++++++++++";
    // LOG(INFO) << run_options.DebugString();
    auto status = session->Run(run_options, inputs, fetches, targets, &outputs,
                               &run_metas);

    if (status.ok()) {
        auto it = fetches.begin();
        for (const auto& tensor : outputs) {
            // LOG(INFO) << record.DebugString(100);

            // LOG(INFO) << *it << " " << tensor.DebugString(100);
            auto tp = rsp->add_outputs();
            tensor.AsProtoTensorContent(tp->mutable_tensor());
            tp->set_name(*it++);
        }

        auto meta = rsp->add_run_metas();
        meta->set_name("default");
        *meta->mutable_run_meta_data() = run_metas;

    } else {
        LOG(ERROR) << status.ToString();
    }

    callback(info);
}

bool GraphContext::fill_inputs(
    std::vector<std::pair<std::string, tensorflow::Tensor>>& inputs) {
    auto& in = req->graph_info().inputs();

    for (const auto& namedtensor : in) {
        tensorflow::Tensor tensor;
        if (unlikely(!tensor.FromProto(namedtensor.tensor()))) return false;

        inputs.emplace_back(namedtensor.name(), std::move(tensor));
    }

    return true;
}

}  // namespace turing
}  // namespace nexus