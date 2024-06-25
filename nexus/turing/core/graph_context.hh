

#pragma once

#include <cstdint>
#include <functional>
#include <vector>

#include "nexus/turing/common/op_util.hh"
#include "nexus/turing/proto/error_code.pb.h"
namespace nexus {
namespace turing {

struct GraphContextArgs {
    int64_t run_id{0};
    tensorflow::SessionResourcePtr session_resource{nullptr};
    tensorflow::RunOptions run_options;
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

  private:
    virtual bool fill_inputs(
        std::vector<std::pair<std::string, tensorflow::Tensor>>& inputs);

  public:
    const GraphRequest* req{nullptr};
    GraphResponse* rsp{nullptr};

    std::vector<tensorflow::Tensor> outputs;

    int64_t run_id{0};
    tensorflow::SessionResourcePtr session_resource{nullptr};
    tensorflow::QueryResourcePtr query_resource{nullptr};
};

}  // namespace turing
}  // namespace nexus