#pragma once

#include <memory>
#include <string>
#include <vector>

#include "tensorflow/core/public/session.h"

namespace nexus {
namespace turing {
struct TFSession final {
  std::vector<std::shared_ptr<tensorflow::Session>> sessions;
  std::vector<std::string> inputs;
  tensorflow::GraphDef graphdef;
  tensorflow::GraphDef fullGraphDef;
  std::string graphName;
};

using TFSessionPtr = std::shared_ptr<TFSession>;
using TFSessionUPtr = std::unique_ptr<TFSession>;

} // namespace turing
} // namespace nexus