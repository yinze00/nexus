#include "nexus/turing/common/tf_session.hh"

#include "tensorflow/core/framework/tensor.h"

namespace nexus {
namespace turing {

class GraphBiz {

public:
    GraphBiz(const GraphBiz&) = delete;
    GraphBiz &operator= (const GraphBiz&) = delete;

public:

    tensorflow::Status init(const std::string& biz_name);

private:

    


};

} // namespace turing
} // namespace nexus
