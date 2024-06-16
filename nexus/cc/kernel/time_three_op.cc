#include "time_three_op.h"

#include "tensorflow/core/framework/op_kernel.h"

using namespace tensorflow;

typedef Eigen::ThreadPoolDevice CPUDevice;
typedef Eigen::GpuDevice GPUDevice;

// CPU specialization of actual computation.
template <typename T> struct TimeThreeFunctor<CPUDevice, T> {
  void operator()(const CPUDevice &d, int size, const T *in, T *out) {
    for (int i = 0; i < size; ++i) {
      out[i] = 3 * in[i];
    }
  }
};

// OpKernel definition.
// template parameter <T> is the datatype of the tensors.
template <typename Device, typename T> class TimeThreeOp : public OpKernel {
public:
  explicit TimeThreeOp(OpKernelConstruction *context) : OpKernel(context) {}

  void Compute(OpKernelContext *context) override {
    // Grab the input tensor
    const Tensor &input_tensor = context->input(0);

    auto device = context->device();

    // LOG(INFO) << "device << std::endl;
    LOG(INFO) << "step_id: " << context->step_id() << std::endl;

    // Create an output tensor
    Tensor *output_tensor = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, input_tensor.shape(),
                                                     &output_tensor));

    // Do the computation.
    OP_REQUIRES(context, input_tensor.NumElements() <= tensorflow::kint32max,
                errors::InvalidArgument("Too many elements in tensor"));
    TimeThreeFunctor<Device, T>()(context->eigen_device<Device>(),
                                  static_cast<int>(input_tensor.NumElements()),
                                  input_tensor.flat<T>().data(),
                                  output_tensor->flat<T>().data());
  }
};

// Register the CPU kernels.
#define REGISTER_CPU(T)                                                        \
  REGISTER_KERNEL_BUILDER(                                                     \
      Name("TimeThree").Device(DEVICE_CPU).TypeConstraint<T>("T"),               \
      TimeThreeOp<CPUDevice, T>);

REGISTER_CPU(float);
REGISTER_CPU(int32);