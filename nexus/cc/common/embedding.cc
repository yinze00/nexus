/*
 * @Description: class embedding impl
 * @Author: junwei.wang@zju.edu.cn
 * @Date: 2024-04-08 22:03:08
 * @LastEditTime: 2024-04-08
 */

#include "embedding.hh"

#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/platform/types.h"

namespace annop {
namespace common {

template <typename T>
class IndexBuffer : public Buffer {
  public:
    explicit IndexBuffer(int64_t n)
        : Buffer(new int8_t[n * sizeof(T)]), n_(n) {}
    ~IndexBuffer() {
        if (void* p = data()) {
            delete reinterpret_cast<int8_t*>(p);
        }
    }
    size_t size() { return n_ * sizeof(T); }

  private:
    int64_t n_;
};

using namespace tensorflow;
#define SINGLE_ARG(...) __VA_ARGS__
#define CASE(TYPE, STMTS)                           \
    case tensorflow::DataTypeToEnum<TYPE>::value: { \
        typedef TYPE T;                             \
        STMTS;                                      \
        break;                                      \
    }
#define CASES_WITH_DEFAULT(TYPE_ENUM, STMTS, INVALID, DEFAULT) \
    switch (TYPE_ENUM) {                                       \
        CASE(float, SINGLE_ARG(STMTS))                         \
        CASE(double, SINGLE_ARG(STMTS))                        \
        CASE(int32, SINGLE_ARG(STMTS))                         \
        CASE(uint8, SINGLE_ARG(STMTS))                         \
        CASE(uint16, SINGLE_ARG(STMTS))                        \
        CASE(uint32, SINGLE_ARG(STMTS))                        \
        CASE(uint64, SINGLE_ARG(STMTS))                        \
        CASE(int16, SINGLE_ARG(STMTS))                         \
        CASE(int8, SINGLE_ARG(STMTS))                          \
        CASE(int64, SINGLE_ARG(STMTS))                         \
        CASE(bool, SINGLE_ARG(STMTS))                          \
        case tensorflow::DT_INVALID:                           \
            INVALID;                                           \
            break;                                             \
        default:                                               \
            DEFAULT;                                           \
            break;                                             \
    }

#define CASES(TYPE_ENUM, STMTS)                                        \
    CASES_WITH_DEFAULT(TYPE_ENUM, STMTS, LOG(FATAL) << "Type not set"; \
                       , LOG(FATAL) << "Unexpected type: " << TYPE_ENUM;)

// ctor
EmbeddingHolder::EmbeddingHolder(DataType type, int64_t n, int dim)
    : type_(type), n_(n), dim_(dim) {
    if (n_ * dim_ > 0) {
        CASES(type_, buf_ = new IndexBuffer<T>(n_ * dim_));
    }
}

// dtor
EmbeddingHolder::~EmbeddingHolder() {
    if (buf_) {
        buf_->Unref();
    }
}

float* EmbeddingHolder::gather_embedding(int64_t offset) {
    auto ptr = buf_->base<float>();
    ptr += offset;
    return ptr;
}

}  // namespace common
}  // namespace annop