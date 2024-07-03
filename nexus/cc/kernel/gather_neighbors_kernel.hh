/*
 * @Description: GatherEmbeddingsOp Kernel
 * @Author: junwei.wang@zju.edu.cn
 * @Date: 2024-06-16 17:06:53
 * @LastEditTime: 2024-06-16
 */
#pragma once

#include "nexus/cc/common/graph.hh"
#include "nexus/turing/common/op_util.hh"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor.h"

namespace tensorflow {

using CPUDEVICE = Eigen::ThreadPoolDevice;

class GatherNeighborsOp : public OpKernel {
  public:
    explicit GatherNeighborsOp(OpKernelConstruction* context)
        : OpKernel(context) {
        OP_REQUIRES_OK(context, context->GetAttr("level", &level_));
        OP_REQUIRES_OK(context, context->GetAttr("index_name", &index_name_));

        auto session_resource = GET_SESSION_RESOURCE(context);

        graph_ =
            session_resource->indexmgr_.get_index(index_name_)->neis_.get();

        to_touch_neighbors_num_ = graph_->to_touch_neighbors_at_level(level_);
        LOG(INFO) << "GatherNeighborsOp @" << level_
                  << " to_touch_neighbors_per " << to_touch_neighbors_num_;
    }

    void Compute(OpKernelContext* ctx);

  private:
    int32_t                level_;
    int32_t                to_touch_neighbors_num_;
    annop::common::HGraph* graph_{nullptr};
    std::string            index_name_;
};

}  // namespace tensorflow