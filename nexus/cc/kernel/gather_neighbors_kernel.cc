#include "gather_neighbors_kernel.hh"

#include <cstdint>
#include <cstring>
#include <mutex>

// #include "nexus/cc/common/ANNIndexHolder.hh"
// #include "nexus/cc/common/singleton.h"
#include "tensorflow/core/framework/op_kernel.h"
// #include "third_party/tensorflow/core/platform/types.h"
using namespace tensorflow;

void GatherNeighborsOp::Compute(OpKernelContext* ctx) {
    auto    entry_points = ctx->input(0);
    auto    n            = entry_points.shape().dim_size(0);
    Tensor* out          = nullptr;

    OP_REQUIRES_OK(ctx,
                   ctx->allocate_output(0, {n, to_touch_neighbors_num_}, &out));

    auto inner_ids = entry_points.flat<uint32_t>();

    auto outflat = out->flat<uint32>();

    auto neis_bytes = to_touch_neighbors_num_ * sizeof(uint32_t);

    for (auto i = 0, offset = 0; i < inner_ids.size(); ++i) {
        auto l = graph_->gather_neighbors(i);
        std::memcpy(outflat.data() + offset, l, neis_bytes);
        offset += to_touch_neighbors_num_;
    }
}

REGISTER_KERNEL_BUILDER(Name("GatherNeighborsOp").Device(DEVICE_CPU),
                        GatherNeighborsOp);
