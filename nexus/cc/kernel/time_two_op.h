
// namespace functor {

template <typename Device, typename T>
struct TimeTwoFunctor {
  void operator()(const Device &d, int size, const T *in, T *out);
};

// } // namespace functor
