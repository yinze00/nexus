/*
 * @Description: Graph class, containing the graph structure of edges
 * @Author: junwei.wang@zju.edu.cn
 * @Date: 2024-05-08 22:11:44
 * @LastEditTime: 2024-04-08
 */

#pragma once

// #include <google/protobuf/generated_message_reflection.h>

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
    T   n_;
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

template <typename T, typename enable = void>
struct HierachyLinkedList;

template <typename T>
struct HierachyLinkedList<
    T, typename std::enable_if<std::is_integral<T>::value>::type>
    : public Buffer {
  public:
    explicit HierachyLinkedList(T n, size_t total)
        : Buffer(new T[total]), n_(n) {}
    ~HierachyLinkedList() = default;

    inline std::tuple<int, T*> gather_neighbors(size_t index, int level) {
        return std::make_tuple(m_, this->template base<T>() + index * m_);
    }

    inline T* gather_neighbors(size_t offset) {
        return this->template base<T>() + offset;
    }

  public:
    T   n_;
    int m_;
    int m0_;
    int max_level_;
};

template <typename T>
struct HierachyLinkedListDeleter {
    void operator()(HierachyLinkedList<T>* obj) const { obj->Unref(); }
};

template <typename T>
using HierachyLinkedListUPtr =
    std::unique_ptr<HierachyLinkedList<T>, HierachyLinkedListDeleter<T>>;

/*
 * Directed Acyclic Graph, Like NSW, which may has only one layer, we called
 * it level0, level0 contains the whole corpus's linked list (neighbors)
 */
class Graph {
  public:
    using LinkedListType     = LinkedList<uint32_t>;
    using LinkedListUPtrType = LinkedListUPtr<uint32_t>;

  public:
    explicit Graph(DataType type, uint64_t n, int m);
    ~Graph();

    // get neis
    void     get_label(uint32_t, uint64_t&);
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
    virtual uint32_t* gather_neighbors(size_t index);

    // void gather_neighbors(const std::vector<size_t>& indice);

    void set_labels(std::vector<uint64_t>& labels);

    virtual void set_neis(LinkedListUPtrType&& ptr, int level = 0) {
        if (!level) linklist_ = std::move(ptr);
    }

    // graph-related
    std::pair<size_t /*begin offset*/, size_t /*end offset*/> neighbors_range(
        uint32_t idx, int level);

    inline uint32_t neighbors(uint32_t item) {
        // return linklist_->gather_neighbors(item);
        return -1;
    }

  public:
    uint64_t n_{0};  // number of elements
    uint32_t m_{0};  // number of neighbors per element

    uint32_t entry_point;

    LinkedListUPtrType    linklist_;
    std::vector<uint64_t> labels_;
    std::vector<size_t>   offsets_;  // offset of item-i in the `linklist_`
};

/* hierachy Graph like hnsw */
class HGraph : public Graph {
  public:
    using HierachyLinkedListUPtrType = HierachyLinkedListUPtr<uint32_t>;

  public:
    explicit HGraph(DataType type, uint64_t n, int m, int h)
        : Graph(type, n, m) {
        h_linklist_.reset(new HierachyLinkedList<uint32_t>(n_, m_));
    }

    std::pair<size_t /*begin offset*/, size_t /*end offset*/> neighbors_range(
        uint32_t idx, int level);

    void set_h_linklist(HierachyLinkedListUPtrType&& ptr) {
        h_linklist_ = std::move(ptr);
    }

    uint32_t* gather_neighbors(size_t index) override;

    inline int to_touch_neighbors_at_level(size_t level) const {
        return to_touch_neighbors_at_level_[level];
    }

  protected:
    HierachyLinkedListUPtrType h_linklist_;  // hierachy linklist
    std::vector<int>           levels_;      // all item's layer N.O.
    std::vector<int>           to_touch_neighbors_at_level_;  // 2M, M, M, ..

    // std::vector<LinkedListUPtrType> non_0_linklist_;
};

}  // namespace common
}  // namespace annop