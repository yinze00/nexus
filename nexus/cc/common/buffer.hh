/*
 * @Description: class buffer for graph linked
 * @Author: junwei.wang@zju.edu.cn
 * @Date: 2024-05-13 23:19:17
 * @LastEditTime: 2024-05-13
 */
#pragma once
#include <memory>

#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/lib/core/refcount.h"

namespace annop {
namespace common {

using DataType = tensorflow::DataType;

/*
 * Memory Block Buffer which may holds elements' embeddings (array float) or
 * index-graph structrure (arrary int)
 */
class Buffer : public tensorflow::core::RefCounted {
  public:
    explicit Buffer(void* data_ptr) : data_(data_ptr) {}
    ~Buffer() = default;

    void* data() const noexcept { return data_; }

    template <typename T>
    T* base() const noexcept {
        return reinterpret_cast<T*>(data());
    }

    tensorflow::DataType type() const noexcept { return dtype_; }

  private:
    tensorflow::DataType dtype_{tensorflow::DataType::DT_INT8};
    void* const          data_;
};

using BufferPtr = std::shared_ptr<Buffer>;

}  // namespace common
}  // namespace annop