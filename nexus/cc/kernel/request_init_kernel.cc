#pragma once
#include <vector>

#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"

namespace tensorflow {

class RequestInitOp : public OpKernel {
  public:
    explicit RequestInitOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
        OP_REQUIRES_OK(ctx, ctx->GetAttr("index_name", &index_name_));
    }

    void Compute(OpKernelContext* ctx) {}

  private:
    std::string index_name_;
};

}  // namespace tensorflow