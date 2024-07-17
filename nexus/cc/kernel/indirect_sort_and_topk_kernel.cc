#include <cstddef>

#include "nexus/turing/common/op_util.hh"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status.h"

using namespace tensorflow;

template <typename T, typename U>
class IndirectSortAndTopkOp : public OpKernel {
  public:
    explicit IndirectSortAndTopkOp(OpKernelConstruction* context)
        : OpKernel(context) {
        OP_REQUIRES_OK(context, context->GetAttr("topk", &topk));
    }

    void Compute(OpKernelContext* ctx) override {
        auto session_resource = GET_SESSION_RESOURCE(ctx);
        auto query_resouce    = GET_QUERY_RESOURCE(session_resource);

        auto& candidates = *query_resouce->candidates;
        auto& results    = *query_resouce->results;
        auto& res        = query_resouce->res;

        candidates.clear();

        const Tensor& k = ctx->input(0);
        const Tensor& v = ctx->input(1);

        VLOG(1) << "k: " << k.DebugString();
        VLOG(1) << "v: " << v.DebugString();

        OP_REQUIRES(
            ctx, k.shape().IsSameSize(v.shape()),
            errors::InvalidArgument("K and V must have the same shape"));

        auto k_flat = k.flat<T>();
        auto v_flat = v.flat<U>();

        int64_t origin_nums = k_flat.size();
        for (auto i = 0; i < origin_nums; ++i) {
            uint32_t label = k_flat(i);
            float    score = v_flat(i);
            VLOG(2) << SSTR(v_flat(i)) << SSTR(k_flat(i));
            // query_resouce->can.add_result(v_flat(i), k_flat(i));
            // query_resouce->candidates->push(k_flat(i), v_flat(i));

            results.push(label, score);
        }

        Tensor* sorted_k = nullptr;
        Tensor* sorted_v = nullptr;

        auto output_num = std::min((int64_t)results.size(), (int64_t)topk);

        OP_REQUIRES_OK(ctx, ctx->allocate_output(0, {output_num}, &sorted_k));
        OP_REQUIRES_OK(ctx, ctx->allocate_output(1, {output_num}, &sorted_v));

        auto sorted_k_flat = sorted_k->flat<T>();
        auto sorted_v_flat = sorted_v->flat<U>();

        for (int i = 0; i < output_num; ++i) {
            sorted_k_flat(i) = results.ids[i];
            sorted_v_flat(i) = results.dis[i];
        }

        VLOG(1) << "Sorted Labels\t" << sorted_k->DebugString(10);
        VLOG(1) << "Sorted Scores\t" << sorted_v->DebugString(10);
    }

  private:
    int topk{100};
};

#define REGISTER_KERNEL(T, U)                             \
    REGISTER_KERNEL_BUILDER(Name("IndirectSortAndTopkOp") \
                                .Device(DEVICE_CPU)       \
                                .TypeConstraint<T>("T")   \
                                .TypeConstraint<U>("U"),  \
                            IndirectSortAndTopkOp<T, U>)

REGISTER_KERNEL(uint32, float);
REGISTER_KERNEL(uint64, double);
REGISTER_KERNEL(uint64, float);
REGISTER_KERNEL(uint64, double);

#undef REGISTER_KERNEL
