#pragma once
#include <google/protobuf/any.h>
#include <google/protobuf/stubs/callback.h>

#include <memory>

#include "nexus/turing/proto/run_graph.pb.h"
namespace nexus {
namespace turing {

// don->run() callback closure;
template <typename ResponseType>
class GraphClosure : public google::protobuf::Closure {
  public:
    GraphClosure(GraphRequest* graph_req, GraphResponse* graph_rsp,
                 ResponseType* rsp, google::protobuf::Closure* done)
        : ureq_(graph_req), ursp_(graph_rsp), rsp_(rsp), done_(done) {}

    void Run() override {
        fill_rsp();
        delete this;
        done_->Run();
    }

    virtual void fill_rsp() = 0;

  private:
    std::unique_ptr<GraphRequest> ureq_;
    std::unique_ptr<GraphResponse> ursp_;
    google::protobuf::Closure* done_;
    ResponseType* rsp_;
};
}  // namespace turing
}  // namespace nexus