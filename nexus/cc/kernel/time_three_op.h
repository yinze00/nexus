// #include "time_two_op.h"

template <typename Device, typename T>
struct TimeThreeFunctor {
  void operator()(const Device &d, int size, const T *in, T *out);
};