/*
 * @Description: GatherEmbeddingsOp Kernel
 * @Author: junwei.wang@zju.edu.cn
 * @Date: 2024-06-16 17:06:53
 * @LastEditTime: 2024-06-16
 */
#pragma once

#include <tensorflow/core/framework/op_kernel.h>
#include <tensorflow/core/framework/tensor.h>

namespace tensorflow {

using CPUDEVICE = Eigen::ThreadPoolDevice;

class GatherNeighborsOp : public OpKernel {
public:
  explicit GatherNeighborsOp(OpKernelConstruction *ctx) : OpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("level", &level_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("index_name", &index_name_));
    LOG(INFO) << "GatherNeighborsOp @" << level_;
  }

  void Compute(OpKernelContext *ctx);

private:
  int32_t level_;
  std::string index_name_;
  mutable std::mutex mtx_;
};

} // namespace tensorflow