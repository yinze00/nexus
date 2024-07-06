#pragma once
#include <vector>

#include "nexus/turing/common/op_util.hh"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"

namespace tensorflow {

class RequestInitOp : public OpKernel {
  public:
    explicit RequestInitOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
        OP_REQUIRES_OK(ctx, ctx->GetAttr("index_name", &index_name_));
    }

    void Compute(OpKernelContext* ctx) {
        auto session_resource = GET_SESSION_RESOURCE(ctx);

        auto entry_point =
            session_resource->get_index(index_name_)->neis_->entry_point;

        VLOG(1) << "entry_point is\t" << entry_point;

        Tensor* out = nullptr;

        OP_REQUIRES_OK(ctx, ctx->allocate_output(0, {1}, &out));

        out->flat<uint32_t>()(0) = entry_point;

        VLOG(1) << "output: " << out->DebugString(100);
    }

  private:
    std::string index_name_;
};

REGISTER_KERNEL_BUILDER(Name("RequestInitOp").Device(DEVICE_CPU), RequestInitOp)

}  // namespace tensorflow