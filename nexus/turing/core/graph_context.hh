

#pragma once

#include <cstdint>
#include <functional>

#include "nexus/turing/common/op_util.hh"
namespace nexus {
namespace turing {

struct GraphContextArgs {
    int64_t run_id{0};
};

struct GraphContext {
    using CallBack = std::function<void()>;

  public:
    GraphContext(const GraphContextArgs&, const GraphRequest* = nullptr,
                 GraphResponse* = nullptr);
    virtual ~GraphContext() = default;

  public:
    virtual void run(CallBack);

  public:
    const GraphRequest* req{nullptr};
    GraphResponse* rsp{nullptr};

    int64_t run_id;
    tensorflow::SessionResourcePtr session_resource;
    tensorflow::QueryResourcePtr query_resource;
};

}  // namespace turing
}  // namespace nexus