#include "nexus/turing/common/session_resource.hh"
#include "nexus/turing/common/tf_session.hh"
#include "tensorflow/core/common_runtime/direct_session.h"
#include "tensorflow/core/framework/tensor.h"

namespace nexus {
namespace turing {

using tensorflow::DirectSession;
using tensorflow::SessionResourcePtr;
using tensorflow::Status;

struct BizInfo {
    std::string biz_name;
    std::string biz_graph_conf;
};

class GraphBiz {
  public:
    GraphBiz(const GraphBiz&) = delete;
    GraphBiz& operator=(const GraphBiz&) = delete;

  public:
    tensorflow::Status init(const std::string& biz_name);

    const tensorflow::RunOptions& getRunOptions() const;

    tensorflow::DirectSession* getSession() const;
    tensorflow::DirectSession* getSession(const std::string&) const;
    std::shared_ptr<TFSession> getTFSession() const;
    std::shared_ptr<TFSession> getTFSession(const std::string&) const;
    void putTFSession(const std::string&, std::shared_ptr<TFSession>);

    SessionResourcePtr getSessionResource() { return session_resource_; }

  protected:
    virtual SessionResourcePtr createSessionResource(uint32_t count);

    // for init & load
    tensorflow::Status load();

  private:
    mutable std::atomic<size_t> session_id = {0};

    SessionResourcePtr session_resource_{nullptr};
};

}  // namespace turing
}  // namespace nexus
