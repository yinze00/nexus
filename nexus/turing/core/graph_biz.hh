#pragma once

#include <ios>
#include <memory>
#include <string>
#include <unordered_map>

#include "graph_context.hh"
#include "nexus/turing/common/session_resource.hh"
#include "nexus/turing/common/tf_session.hh"
#include "tensorflow/core/common_runtime/direct_session.h"
#include "tensorflow/core/framework/tensor.h"

namespace nexus {
namespace turing {

using tensorflow::Session;
using tensorflow::SessionResourcePtr;
using tensorflow::Status;

struct BizInfo {
    std::string biz_name;
    std::string biz_graph_conf;
};

class GraphBiz {
  public:
    GraphBiz() = default;
    virtual ~GraphBiz() = default;
    GraphBiz(const GraphBiz&) = delete;
    GraphBiz& operator=(const GraphBiz&) = delete;

  public:
    virtual tensorflow::Status init(const std::string& biz_name);

    const tensorflow::RunOptions& getRunOptions() const;

    std::shared_ptr<tensorflow::Session> getSession() const {
        return getSession(biz_name);
    };
    std::shared_ptr<tensorflow::Session> getSession(const std::string&) const;
    std::shared_ptr<TFSession> getTFSession() const {
        return getTFSession(biz_name);
    };
    std::shared_ptr<TFSession> getTFSession(const std::string& model) const {
        LOG(INFO) << "gettfsession: " << model << " sessions_.size = " << sessions_.size();
        auto it = sessions_.find(model);
        std::cout << model << std::boolalpha << (it == sessions_.end());
        if (it != sessions_.end()) {
            return it->second;
        }
    };
    void putTFSession(const std::string&, std::shared_ptr<TFSession>);

    SessionResourcePtr getSessionResource() { return session_resource_; }

    // graph

    GraphContextArgs getGraphContextArgs();

  protected:
    static std::vector<std::string> getAllPlaceholders(
        const tensorflow::GraphDef& def);

    virtual SessionResourcePtr createSessionResource(uint32_t count);
    tensorflow::Status createSession(std::shared_ptr<TFSession>& tfsession);

  private:
    tensorflow::Status loadBizInfo();
    // tensorflow::Status loadGraphDef();
    tensorflow::Status loadGraph();

  private:
    std::string biz_name{"default"};

    mutable std::atomic<size_t> session_id = {0};

    SessionResourcePtr session_resource_{nullptr};
    tensorflow::SessionOptions options;

    std::unordered_map<std::string, TFSessionPtr> sessions_;
};

using GraphBizPtr = std::shared_ptr<GraphBiz>;

}  // namespace turing
}  // namespace nexus
