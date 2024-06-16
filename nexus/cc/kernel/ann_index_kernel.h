/*
 * @Description: ann index manager op
 * @Author: junwei.wang@zju.edu.cn
 * @Date: 2024-04-25 02:05:18
 * @LastEditTime: 2024-04-25
 */

#pragma once

#include <tensorflow/core/framework/op_kernel.h>
#include <tensorflow/core/framework/tensor.h>

namespace tensorflow {

using CPUDEVICE = Eigen::ThreadPoolDevice;

// template <typename T>
class ANNIndexOp : OpKernel {
  public:
    explicit ANNIndexOp(OpKernelConstruction* context);
    void Compute(OpKernelContext* ctx);

  private:
  
};

}  // namespace tensorflow