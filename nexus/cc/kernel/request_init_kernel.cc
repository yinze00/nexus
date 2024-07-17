#pragma once
#include <memory>
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
        auto query_resouce    = GET_QUERY_RESOURCE(session_resource);
        {
            auto topk = ctx->input(0).scalar<uint32>()();
            query_resouce->candidate_labels.resize(topk << 1);
            query_resouce->candidate_scores.resize(topk << 1);
            query_resouce->result_labels.resize(topk);
            query_resouce->result_scores.resize(topk);
            // query_resouce->candidates.scores =
            //     query_resouce->candidate_scores.data();
            // query_resouce->candidates.labels =
            //     query_resouce->candidate_labels.data();
            // query_resouce->candidates.topk = topk << 1;

            query_resouce->candidates =
                std::make_unique<annop::MiniMaxHeap>(topk << 1);

            query_resouce->results = std::make_unique<annop::MiniMinHeap>(topk);

            // query_resouce->results.scores =
            // query_resouce->result_scores.data();

            // query_resouce->results.labels =
            // query_resouce->result_labels.data();

            // query_resouce->results.topk = topk;
            // query_resouce->results.heapify();
        }

        query_resouce->visited_table_ =
            std::make_shared<QueryResource::VisitedTable>(
                session_resource->get_index(index_name_)->neis_->n_);

        query_resouce->visited.resize(
            session_resource->get_index(index_name_)->neis_->n_, false);

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