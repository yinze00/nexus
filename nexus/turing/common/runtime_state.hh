#pragma once
#include "nexus/turing/proto/run_graph.pb.h"

#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/protobuf/config.pb.h"
#include <memory>

namespace tensorflow {

class RuntimeState {
public:
  RuntimeState() = default;
  ~RuntimeState() = default;

  // disable assignment
  RuntimeState(const RuntimeState &) = delete;
  RuntimeState &operator=(const RuntimeState &) = delete;

public:
  //@getter
  const RunOptions &getRunOptions() const { return run_options_; }
  RunOptions &getRunOptions() { return run_options_; }

  void setRunId(int64_t run_id) {
    run_options_.set_run_id(run_id);
    run_id_ = run_id;
  }

  //@setter
  void setRunOptions(const RunOptions &run_options) {
    run_options_ = run_options;
  }

  void addRunMetaData(nexus::turing::NamedRunMetadata *);
  const std::vector<std::shared_ptr<nexus::turing::NamedRunMetadata>> &
  getRunMetadata() const {
    return run_metas_;
  }
  std::vector<std::shared_ptr<nexus::turing::NamedRunMetadata>> &
  getRunMetadata() {
    return run_metas_;
  }

private:
  std::vector<std::shared_ptr<nexus::turing::NamedRunMetadata>> run_metas_;

  RunOptions run_options_;
  int64_t run_id_;
};

using RuntimeStatePtr = std::shared_ptr<RuntimeState>;
using RuntimeStateUPtr = std::unique_ptr<RuntimeState>;

} // namespace tensorflow