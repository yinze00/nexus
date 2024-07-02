/*
 * @Description: GemvOp Kernel
 * @Author: junwei.wang@zju.edu.cn
 * @Date: 2024-06-16 18:12:34
 * @LastEditTime: 2024-06-16
 */
#pragma once
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"
#include <vector>

namespace tensorflow {

template <typename T> class GemvOp : public OpKernel {
public:
  explicit GemvOp(OpKernelConstruction *ctx) : OpKernel(ctx) {}
  void Compute(OpKernelContext *ctx) {
    // 获取输入张量
    const Tensor &matrix = ctx->input(1);
    const Tensor &vector = ctx->input(0);

    OP_REQUIRES(ctx, matrix.dims() == 2,
                errors::InvalidArgument("matrix must be 2-dimensional"));
    OP_REQUIRES(ctx, vector.dims() == 1,
                errors::InvalidArgument("vector must be 1-dimensional"));

    int batch_size = matrix.dim_size(0);
    int num_cols = matrix.dim_size(1);
    int num_rows = vector.dim_size(0);

    Tensor *output = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, {batch_size, 1}, &output));

    auto matrix_data = matrix.flat<T>();
    auto vector_data = vector.flat<T>();
    auto output_data = output->flat<T>();

    for (int i = 0; i < batch_size; ++i) {
      T sum = static_cast<T>(0);
      for (int j = 0; j < num_cols; ++j) {
        sum += matrix_data(i * num_cols + j) * vector_data(j);
      }
      output_data(i) = sum;
    }
  }
};
#define REGISTER_KERNEL(T)                                                     \
  REGISTER_KERNEL_BUILDER(                                                     \
      Name("GemvOp").Device(DEVICE_CPU).TypeConstraint<T>("T"), GemvOp<T>)

REGISTER_KERNEL(float);
REGISTER_KERNEL(double);
// REGISTER_KERNEL(int32);
// REGISTER_KERNEL(int64);
} // namespace tensorflow