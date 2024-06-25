#include "graph_service_impl.hh"

#include <memory>

#include "graph_context.hh"

namespace nexus {
namespace turing {

// GraphServiceImpl::GraphServiceImpl()

void GraphServiceImpl::init() { biz_ = init_biz(); }

GraphBizPtr GraphServiceImpl::init_biz() {
    return std::make_shared<GraphBiz>();
}

void GraphServiceImpl::runGraph(::google::protobuf::RpcController* controller,
                                const GraphRequest* request,
                                GraphResponse* response,
                                ::google::protobuf::Closure* done) {
    auto out1 = response->add_outputs();
    out1->set_name("mmoe");

    GraphContextArgs args;

    args.run_id = run_id_allocator->get();

    // auto ctx = createContext(args, request, response);

    this->process<GraphRequest, GraphResponse>(controller, request, response,
                                               done);
}
}  // namespace turing
}  // namespace nexus