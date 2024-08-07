/**
 * @author junwei.wang
 * @email junwei.wang@zju.edu.cn
 * @create date 2024-06-27 15:52:59
 * @modify date 2024-06-27 15:52:59
 * @desc Grpah Service implementation
 */

#include "graph_service_impl.hh"

#include <memory>

#include "graph_biz.hh"
#include "graph_context.hh"

namespace nexus {
namespace turing {

// GraphServiceImpl::GraphServiceImpl()

void GraphServiceImpl::init() {
    biz_ = init_biz();
    TF_CHECK_OK(biz_->init("default"));
    run_id_allocator.reset(new common::RunIDAllocator());
    run_id_allocator->init(1024);
}

GraphBizPtr GraphServiceImpl::init_biz() {
    return std::make_shared<GraphBiz>();
}

void GraphServiceImpl::runGraph(::google::protobuf::RpcController* controller,
                                const GraphRequest* request,
                                GraphResponse* response,
                                ::google::protobuf::Closure* done) {
    // auto out1 = response->add_outputs();
    // out1->set_name("mmoe");

    this->process<GraphRequest, GraphResponse>(controller, request, response,
                                               done);
}
}  // namespace turing
}  // namespace nexus