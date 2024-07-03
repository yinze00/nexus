
#include "gather_embeddings_kernel.hh"

#include "nexus/turing/common/op_util.hh"

typedef Eigen::ThreadPoolDevice CPUDevice;
typedef Eigen::GpuDevice        GPUDevice;

namespace tensorflow {

// REGISTER_OP("GatherEmbeddingsOp")
//     .Input("internal_ids: int32")
//     .Output("embeddings: float")
//     .Attr("index_name: string")
//     .Attr("dim: int32")
//     .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
//         return Status::OK();
//     });

void GatherEmbeddingsOp::Compute(OpKernelContext* ctx) {
    auto session_resource = GET_SESSION_RESOURCE(ctx);

    auto index = session_resource->get_index(index_name_);

    auto neis_t = ctx->input(0);

    auto neis_n = neis_t.shape().dim_size(0);

    LOG(INFO) << neis_t.DebugString();

    Tensor* out = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, {neis_n, dim_}, &out));
    auto oflat = out->flat<float>();
    auto neis  = neis_t.flat<uint32_t>();

    auto bytes = dim_ * sizeof(float);

    for (auto i = 0, offset = 0; i < neis_n; ++i) {
        int32_t inner_id = static_cast<int32_t>(neis(i));
        // LOG(INFO) << "neis(" << i << ") = " << inner_id;
        if (inner_id != -1) {
            auto emb = index->embedding_->gather_embedding(inner_id);
            if (!i) {
                // LOG(INFO) << emb[0];
                // LOG(INFO) << emb[1];
                // LOG(INFO) << emb[dim_];
            }
            std::memcpy(oflat.data() + offset, emb, bytes);
        }
        offset += dim_;
    }
    LOG(INFO) << "output: " << out->DebugString(100);
}

REGISTER_KERNEL_BUILDER(Name("GatherEmbeddingsOp").Device(DEVICE_CPU),
                        GatherEmbeddingsOp);

}  // namespace tensorflow
