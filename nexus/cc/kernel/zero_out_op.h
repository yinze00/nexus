#include "tensorflow/core/framework/op_kernel.h"

using namespace tensorflow;

class ZeroOutOp : public OpKernel {
public:
  explicit ZeroOutOp(OpKernelConstruction *context) : OpKernel(context) {}
  void Compute(OpKernelContext *context) override;
};