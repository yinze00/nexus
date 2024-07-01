#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status.h"

using namespace tensorflow;

template <typename T, typename U>
class IndirectSortAndTopKOp : public OpKernel {
  public:
    explicit IndirectSortAndTopKOp(OpKernelConstruction* context)
        : OpKernel(context) {
        OP_REQUIRES_OK(context, context->GetAttr("topk", &topk));
    }

    void Compute(OpKernelContext* context) override {
        const Tensor& k = context->input(0);
        const Tensor& v = context->input(1);

        OP_REQUIRES(
            context, k.shape().IsSameSize(v.shape()),
            errors::InvalidArgument("K and V must have the same shape"));

        auto k_flat = k.flat<T>();
        auto v_flat = v.flat<U>();

        std::vector<std::pair<T, U>> kv_pairs;
        for (int i = 0; i < k_flat.size(); ++i) {
            kv_pairs.emplace_back(k_flat(i), v_flat(i));
        }

        std::sort(kv_pairs.begin(), kv_pairs.end(),
                  [](const std::pair<T, U>& a, const std::pair<T, U>& b) {
                      return a.second >
                             b.second;  // Sort by value in descending order
                  });

        Tensor* sorted_k = nullptr;
        Tensor* sorted_v = nullptr;
        OP_REQUIRES_OK(context,
                       context->allocate_output(0, k.shape(), &sorted_k));
        OP_REQUIRES_OK(context,
                       context->allocate_output(1, v.shape(), &sorted_v));

        auto sorted_k_flat = sorted_k->flat<T>();
        auto sorted_v_flat = sorted_v->flat<U>();

        for (int i = 0; i < std::min(static_cast<size_t>(topk), kv_pairs.size()); ++i) {
            sorted_k_flat(i) = kv_pairs[i].first;
            sorted_v_flat(i) = kv_pairs[i].second;
        }
    }

  private:
    int topk{100};
};

#define REGISTER_KERNEL(T, U)                            \
    REGISTER_KERNEL_BUILDER(Name("SortTensorsByValues")  \
                                .Device(DEVICE_CPU)      \
                                .TypeConstraint<T>("T")  \
                                .TypeConstraint<U>("U"), \
                            IndirectSortAndTopKOp<T, U>)

REGISTER_KERNEL(uint32, float);
REGISTER_KERNEL(uint32, double);
REGISTER_KERNEL(uint64, float);
REGISTER_KERNEL(uint64, double);

#undef REGISTER_KERNEL
