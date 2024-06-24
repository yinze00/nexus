/*
 * @Description: Graph class, containing the graph structure of edges
 * @Author: junwei.wang@zju.edu.cn
 * @Date: 2024-05-08 22:11:44
 * @LastEditTime: 2024-04-08
 */

#pragma once

#include <google/protobuf/generated_message_reflection.h>

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <tuple>
#include <type_traits>
#include <vector>

#include "buffer.hh"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/lib/core/refcount.h"
#include "tensorflow/core/platform/types.h"

namespace annop {
namespace common {

template <typename T, typename enable = void>
struct LinkedList;

template <typename T>
struct LinkedList<T, typename std::enable_if<std::is_integral<T>::value>::type>
    : public Buffer {
  public:
    explicit LinkedList(T n, int m) : n_(n), m_(m), Buffer(new T[n * m]) {}
    explicit LinkedList(T n, size_t total) : n_(n), Buffer(new T[total]) {}
    ~LinkedList() = default;
    inline std::tuple<int, T*> gather_neighbors(size_t index) {
        return std::make_tuple(m_, this->template base<T>() + index * m_);
    }

  public:
    T n_;
    int m_;
};

template <typename T>
struct LinkedListDeleter {
    void operator()(LinkedList<T>* obj) const { obj->Unref(); }
};

template <typename T>
using LinkedListUPtr = std::unique_ptr<LinkedList<T>, LinkedListDeleter<T>>;

template <typename T>
using LinkedListPtr = std::shared_ptr<LinkedList<T>>;

/*
 * Directed Acyclic Graph, Like NSW, which may has only one layer, we called
 * it level0, level0 contains the whole corpus's linked list (neighbors)
 */
class Graph {
  public:
    using LinkedListType = LinkedList<uint32_t>;
    using LinkedListUPtrType = LinkedListUPtr<uint32_t>;

  public:
    explicit Graph(DataType type, uint64_t n, int m);
    ~Graph();

    // get neis
    void get_label(uint32_t, uint64_t&);
    uint64_t get_label(uint32_t index) {
        uint64_t res;
        get_label(index, res);
        return res;
    }
    void get_labels(const std::vector<uint32_t>&, std::vector<uint64_t>&);
    std::vector<uint64_t> get_labels(const std::vector<uint32_t>& indice) {
        std::vector<uint64_t> res;
        get_labels(indice, res);
        return res;
    }
    // set neis

    uint32_t* gather_neighbors(size_t index);

    void gather_neighbors(const std::vector<size_t>& indice);

    void set_labels(std::vector<uint64_t>& labels);

    virtual void set_neis(LinkedListUPtrType&& ptr, int level = 0) {
        if (!level) linklist_ = std::move(ptr);
    }

  protected:
    uint64_t n_{0};  // number of elements
    uint32_t m_{0};  // number of neighbors per element

    LinkedListUPtrType linklist_;

    std::vector<uint64_t> labels_;
};

/* hierachy Graph like hnsw */
class HGraph : public Graph {
  public:
    using Graph::LinkedListUPtrType;

  public:
    HGraph(DataType type, uint64_t n, int m, int h) : Graph(type, n, m) {
        // assert(h > 0);
        non_0_linklist_.reserve(h - 1);
    }

    void set_neis(LinkedListUPtrType&& ptr, int level = 0) override {
        // if (!level)
        //     linklist_ = std::move(ptr);
        // else {
        //     if (level < non_0_linklist_.size()) {
        //         non_0_linklist_[level - 1] = std::move(ptr);
        //     }
        // }
    }

  protected:
    std::vector<LinkedListUPtrType> non_0_linklist_;
};

}  // namespace common
}  // namespace annop