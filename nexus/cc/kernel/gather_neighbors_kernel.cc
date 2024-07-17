#include "gather_neighbors_kernel.hh"

#include <cstdint>
#include <cstring>
#include <mutex>
#include <vector>

#include "tensorflow/core/framework/op_kernel.h"

using namespace tensorflow;

void GatherNeighborsOp::Compute(OpKernelContext* ctx) {
    auto  session_resource = GET_SESSION_RESOURCE(ctx);
    auto  query_resouce    = GET_QUERY_RESOURCE(session_resource);
    auto& vt               = query_resouce->visited_table_;
    auto& visited          = query_resouce->visited;

    auto    entry_points = ctx->input(0);
    auto    n            = entry_points.shape().dim_size(0);
    Tensor* out          = nullptr;

    std::vector<uint32_t> neis;
    neis.reserve(n * to_touch_neighbors_num_);

    auto inner_ids = entry_points.flat<uint32_t>();

#if 0
    OP_REQUIRES_OK(
        ctx, ctx->allocate_output(0, {n * to_touch_neighbors_num_}, &out));

    auto outflat = out->flat<uint32>();

    auto neis_bytes = to_touch_neighbors_num_ * sizeof(uint32_t);
    for (auto i = 0, offset = 0; i < inner_ids.size(); ++i) {
        auto label = inner_ids(i);
        if (label != -1) {
            auto l = graph_->gather_neighbors(label);
            std::memcpy(outflat.data() + offset, l, neis_bytes);

        } else {
            auto p    = outflat.data() + offset;
            auto temp = to_touch_neighbors_num_;
            while (temp-- > 0) {
                *p++ = -1;
            }
        }
        offset += to_touch_neighbors_num_;
    }
#else
    // size_t cnt = 0;
    // for (bool visi : visited) {
    //     if (visi) LOG(INFO) << cnt << "\t" << std::boolalpha << visi;
    //     ++cnt;
    // }

    for (auto i = 0; i < inner_ids.size(); ++i) {
        auto label = inner_ids(i);
        VLOG(1) << "++++ label " << label;
        if (label != -1) {
            auto l = graph_->gather_neighbors(label);
            VLOG(1) << "--- visited[" << label << "] = " << std::boolalpha
                    << visited[label];
            if (true) {
                // VLOG(1) << "---- unvisited : " << label;
                // neis.push_back(label);
                // vt->set(label);
                visited[label] = true;
                for (auto ii = 0; ii < to_touch_neighbors_num_; ++ii) {
                    if (*l == -1) break;
                    if (!visited[*l]) {
                        neis.push_back(*l);
                        // vt->set(*l);
                        visited[*l] = true;
                    }
                    ++l;
                }
            }
        }
    }
    LOG(INFO) << "neis.size = " << neis.size();
    OP_REQUIRES_OK(ctx, ctx->allocate_output(
                            0, {static_cast<long long>(neis.size())}, &out));

    std::copy_n(neis.data(), neis.size(), out->flat<uint32_t>().data());
#endif
    // vt->advance();
    VLOG(1) << "Gathered Neighbors: " << out->DebugString(100);
}

REGISTER_KERNEL_BUILDER(Name("GatherNeighborsOp").Device(DEVICE_CPU),
                        GatherNeighborsOp);
