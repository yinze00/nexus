#pragma once
#include <google/protobuf/any.h>
#include <google/protobuf/stubs/callback.h>

#include <memory>

#include "nexus/turing/common/scope_deleter.hh"
#include "nexus/turing/proto/run_graph.pb.h"

namespace nexus {
namespace turing {

// don->run() callback closure;
// template <typename ResponseType>
class GraphClosure : public google::protobuf::Closure {
  public:
    GraphClosure(GraphRequest* graph_req, GraphResponse* graph_rsp, void* rsp,
                 google::protobuf::Closure* done)
        : ureq_(graph_req), ursp_(graph_rsp), done_(done), rsp_(rsp) {}

    void Run() override {
        nexus::common::ScopeDeleter<GraphClosure> deleter(this);

        fill_rsp();
        done_->Run();
    }

    virtual void fill_rsp() = 0;

  protected:
    std::unique_ptr<GraphRequest> ureq_;
    std::unique_ptr<GraphResponse> ursp_;
    google::protobuf::Closure* done_;
    void* rsp_;
};
}  // namespace turing
}  // namespace nexus