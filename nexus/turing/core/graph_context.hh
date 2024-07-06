

#pragma once

#include <cstdint>
#include <functional>
#include <memory>
#include <vector>

#include "nexus/turing/common/op_util.hh"
// #include "nexus/turing/core/graph_biz.hh"
#include "nexus/turing/proto/error_code.pb.h"
namespace nexus {
namespace turing {

class GraphBiz;

struct GraphContextArgs {
    // int64_t run_id{0};
    tensorflow::SessionResourcePtr       session_resource{nullptr};
    tensorflow::RunOptions               run_options;
    std::shared_ptr<tensorflow::Session> session{nullptr};
};

struct GraphContext {
    using CallBack = std::function<void(ErrorInfo&)>;

  public:
    GraphContext() = default;
    GraphContext(const GraphContextArgs&, const GraphRequest* = nullptr,
                 GraphResponse* = nullptr);
    virtual ~GraphContext() = default;

  public:
    virtual void run(CallBack);

    void addQueryResource(int64_t id, tensorflow::QueryResourcePtr qry) {
        session_resource->add_query_resource(id, qry);
    }

  protected:
    virtual bool doParseRequestBody() { return true; }

    bool parseRequest() { return doParseRequestBody(); }

    // void prepareQueryResource(std::shared_ptr<GraphBiz> biz,
    //                           tensorflow::RunOptions& opts);

  private:
    virtual bool fill_inputs(
        std::vector<std::pair<std::string, tensorflow::Tensor>>& inputs);

  public:
    // Graph Rep Rsp
    const GraphRequest* req{nullptr};
    GraphResponse*      rsp{nullptr};

    // TF Runtime
    int64_t                         run_id{0};
    tensorflow::RunOptions          run_options;
    tensorflow::RunMetadata         run_metas;
    std::vector<tensorflow::Tensor> outputs;

    tensorflow::SessionResourcePtr       session_resource{nullptr};
    tensorflow::QueryResourcePtr         query_resource{nullptr};
    std::shared_ptr<tensorflow::Session> session{nullptr};
};

}  // namespace turing
}  // namespace nexus