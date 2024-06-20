#pragma once

#include "graph_manager.hh"
#include "nexus/turing/proto/run_graph.pb.h"

namespace nexus {
namespace turing {

class GraphServiceImpl : public GraphManager {
public:
  void runGraph(::google::protobuf::RpcController *controller,
                const GraphRequest *request, GraphResponse *response,
                ::google::protobuf::Closure *done);

private:
  void demo();
};

} // namespace turing
} // namespace nexus