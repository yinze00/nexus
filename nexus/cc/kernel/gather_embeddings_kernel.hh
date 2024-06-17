/*
 * @Description: GatherEmbeddingsOp Kernel
 * @Author: junwei.wang@zju.edu.cn
 * @Date: 2024-06-16 18:12:34
 * @LastEditTime: 2024-06-16
 */
#pragma once
#include "tensorflow/core/framework/op_kernel.h"
#include <vector>

namespace tensorflow {

// using namespace tensorflow;

class GatherEmbeddingsOp : public OpKernel {
public:
  explicit GatherEmbeddingsOp(OpKernelConstruction *ctx) : OpKernel(ctx) {}

  void Compute(OpKernelContext *ctx) override;

private:
};

} // namespace tensorflow