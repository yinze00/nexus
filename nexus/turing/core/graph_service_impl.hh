#pragma once

#include <google/protobuf/stubs/callback.h>

#include <atomic>
#include <functional>

#include "graph_context.hh"
#include "graph_manager.hh"
#include "nexus/turing/common/run_id_allocator.hh"
#include "nexus/turing/core/graph_biz.hh"
#include "nexus/turing/proto/run_graph.pb.h"

namespace nexus {
namespace turing {

template <typename ReqT, typename RspT>
using CreateContextFunc =
    std::function<GraphContext*(const GraphContextArgs&, const ReqT*, RspT*)>;

class GraphServiceImpl : public GraphManager {
  public:
    GraphServiceImpl()          = default;
    virtual ~GraphServiceImpl() = default;

    void init();

    virtual GraphBizPtr init_biz();

  public:
    virtual void runGraph(::google::protobuf::RpcController* controller,
                          const GraphRequest* request, GraphResponse* response,
                          ::google::protobuf::Closure* done);

    virtual GraphContext* doCreateContext(const GraphContextArgs& args,
                                          const GraphRequest*     request,
                                          GraphResponse*          response) {
        return new GraphContext(args, request, response);
    }
    template <typename ReqT, typename RspT>
    void process(::google::protobuf::RpcController* controller,
                 const ReqT* request, RspT* respons,
                 ::google::protobuf::Closure*  done,
                 CreateContextFunc<ReqT, RspT> func = nullptr);

  protected:
    GraphContext* createContext(const GraphContextArgs& args,
                                const GraphRequest* req, GraphResponse* rsp) {
        return doCreateContext(args, req, rsp);
    }

  private:
    std::atomic_int_fast64_t  session_id_{0};
    mutable int64_t           max_session_{10};
    GraphBizPtr               biz_{nullptr};
    common::RunIDAllocatorPtr run_id_allocator;
};

template <typename ReqT, typename RspT>
void GraphServiceImpl::process(::google::protobuf::RpcController* controller,
                               const ReqT* request, RspT* response,
                               ::google::protobuf::Closure*  done,
                               CreateContextFunc<ReqT, RspT> func) {
    auto cur = session_id_.fetch_add(1, std::memory_order_relaxed);

    auto runid = run_id_allocator->get();

    GraphContextArgs             argv = biz_->getGraphContextArgs();
    tensorflow::QueryResourcePtr qrp  = biz_->prepareQueryResource();

    argv.run_options.set_run_id(runid);

    VLOG(1) << "argv.run_id = " << argv.run_options.run_id();

    argv.session_resource = biz_->getSessionResource();
    auto ctx              = createContext(argv, request, response);

    VLOG(1) << "add query_resouce for run_id: " << runid;
    ctx->addQueryResource(runid, qrp);

    ctx->run([this, response, done, runid](ErrorInfo&) -> void {
        done->Run();
        run_id_allocator->put(runid);
        session_id_.fetch_sub(1, std::memory_order_relaxed);
    });
}

}  // namespace turing
}  // namespace nexus