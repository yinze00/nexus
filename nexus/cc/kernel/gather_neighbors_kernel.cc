#include "gather_neighbors_kernel.hh"
#include "nexus/cc/common/IndexCommonSingleton.hh"
#include "nexus/cc/common/singleton.h"
#include "tensorflow/core/framework/op_kernel.h"
#include <mutex>
using namespace tensorflow;

void GatherNeighborsOp::Compute(OpKernelContext *ctx) {}

REGISTER_KERNEL_BUILDER(Name("GatherNeighborsOp").Device(DEVICE_CPU),
                        GatherNeighborsOp);
