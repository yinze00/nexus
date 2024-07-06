#pragma once
#include <vector>

#include "nexus/turing/common/op_util.hh"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"

namespace tensorflow {

class ResultConstructOp : public OpKernel {
  public:
    explicit ResultConstructOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
        OP_REQUIRES_OK(ctx, ctx->GetAttr("index_name", &index_name_));
    }

    void Compute(OpKernelContext* ctx) {
        auto session_resource = GET_SESSION_RESOURCE(ctx);
        auto query_resource   = GET_QUERY_RESOURCE(session_resource);

        auto&   rslt = query_resource->results;
        int64_t nums = rslt.size();

        Tensor* out_labels = nullptr;
        Tensor* out_scores = nullptr;
        OP_REQUIRES_OK(ctx, ctx->allocate_output(0, {nums}, &out_labels));
        OP_REQUIRES_OK(ctx, ctx->allocate_output(1, {nums}, &out_scores));

        auto lf = out_labels->flat<uint32_t>().data();
        auto sf = out_scores->flat<float>().data();

        while (!rslt.empty()) {
            auto top = rslt.top();
            *lf++    = std::get<0>(top);
            *sf++    = std::get<1>(top);
            rslt.pop();
        }

        VLOG(1) << "Labels:\t" << out_labels->DebugString(10);
        VLOG(1) << "Scores:\t" << out_scores->DebugString(10);
    }

  private:
    std::string index_name_;
};

REGISTER_KERNEL_BUILDER(Name("ResultConstructOp").Device(DEVICE_CPU),
                        ResultConstructOp)

}  // namespace tensorflow