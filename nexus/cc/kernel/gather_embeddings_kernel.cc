
#include "gather_embeddings_kernel.hh"
using namespace tensorflow;

typedef Eigen::ThreadPoolDevice CPUDevice;
typedef Eigen::GpuDevice GPUDevice;

namespace tensorflow {

void GatherEmbeddingsOp::Compute(OpKernelContext *ctx) {
    
}

REGISTER_KERNEL_BUILDER(Name("GatherEmbeddingsOp").Device(DEVICE_CPU),
                        GatherEmbeddingsOp);

} // namespace tensorflow
