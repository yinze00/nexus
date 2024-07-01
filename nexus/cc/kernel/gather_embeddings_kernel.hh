/*
 * @Description: GatherEmbeddingsOp Kernel
 * @Author: junwei.wang@zju.edu.cn
 * @Date: 2024-06-16 18:12:34
 * @LastEditTime: 2024-06-16
 */
#pragma once
#include <vector>

#include "tensorflow/core/framework/op_kernel.h"

namespace tensorflow {

// using namespace tensorflow;

class GatherEmbeddingsOp : public OpKernel {
  public:
    explicit GatherEmbeddingsOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
        OP_REQUIRES_OK(ctx, ctx->GetAttr("index_name", &index_name_));
          OP_REQUIRES_OK(ctx, ctx->GetAttr("dim", &dim_));
    }

    void Compute(OpKernelContext* ctx) override;

  private:
    int32_t dim_;
    std::string index_name_;
};

}  // namespace tensorflow