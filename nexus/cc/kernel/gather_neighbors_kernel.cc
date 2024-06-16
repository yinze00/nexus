#include "gather_neighbors_kernel.hh"
#include <tensorflow/core/framework/op_kernel.h>
#include <tensorflow/core/framework/tensor.h>

#include <mutex>

namespace tensorflow {

void GatherNeighborsOp::Compute(OpKernelContext *ctx) {}

REGISTER_KERNEL_BUILDER(Name("GatherNeighborsOp").Device(DEVICE_CPU),
                        GatherNeighborsOp);

} // namespace tensorflow