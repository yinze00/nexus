#include "graph_service_impl.hh"

namespace nexus {
namespace turing {
void GraphServiceImpl::runGraph(::google::protobuf::RpcController *controller,
                                const GraphRequest *request,
                                GraphResponse *response,
                                ::google::protobuf::Closure *done) {
  auto out1 = response->add_outputs();
  out1->set_name("mmoe");
}
} // namespace turing
} // namespace nexus