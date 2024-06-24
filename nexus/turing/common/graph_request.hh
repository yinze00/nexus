#pragma once
#include <string>
#include <utility>
#include <vector>

#include "tensorflow/core/framework/tensor.h"

namespace nexus {
namespace turing {

struct TuringRequest {
  public:
    TuringRequest(const TuringRequest&) = delete;
    TuringRequest& operator=(const TuringRequest&) = delete;

  public:
    std::vector<std::pair<std::string, tensorflow::Tensor>> inputs;
    std::vector<std::string> outputs;
}

}  // namespace turing
}  // namespace nexus