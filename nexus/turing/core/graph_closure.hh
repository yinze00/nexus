#pragma once
#include "nexus/turing/proto/run_graph.pb.h"

#include <google/protobuf/any.h>
#include <google/protobuf/stubs/callback.h>
namespace nexus {
namespace turing {

// don->run() callback closure;
class GraphClosure : public google::protobuf::Closure {
public:
  GraphClosure(GraphRequest *req, GraphResponse *rsp);
};
} // namespace turing
} // namespace nexus