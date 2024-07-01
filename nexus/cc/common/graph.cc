/*
 * @Description: graph impl
 * @Author: junwei.wang@zju.edu.cn
 * @Date: 2024-05-13 23:39:22
 * @LastEditTime: 2024-04-13
 */
#include "graph.hh"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <utility>
#include <vector>

namespace annop {
namespace common {

// ctor
Graph::Graph(DataType type, uint64_t n, int m) : n_(n), m_(m) {
    labels_.reserve(n);
    linklist_.reset(new LinkedListType((uint32_t)n_, (size_t)m_));
}

// dtor
Graph::~Graph() {
    if (linklist_) {
        linklist_->Unref();
    }
}

void Graph::get_label(uint32_t index, uint64_t& label) {}

void Graph::get_labels(const std::vector<uint32_t>& indice,
                       std::vector<uint64_t>&       labels) {}

void Graph::set_labels(std::vector<uint64_t>& labels) { labels_.swap(labels); }

uint32_t* Graph::gather_neighbors(size_t index) {
    auto [l, r] = neighbors_range(index, 0);

    return nullptr;
}

std::pair<size_t, size_t> Graph::neighbors_range(uint32_t idx, int level) {
    auto offset = offsets_[idx];
    return std::make_pair(offset, offset + m_);
}

//// HGraph
uint32_t* HGraph::gather_neighbors(size_t idx) {
    auto [l, r] = neighbors_range(idx, levels_[idx]);
    return h_linklist_->gather_neighbors(l);
}

std::pair<size_t, size_t> HGraph::neighbors_range(uint32_t idx, int level) {
    auto offset = offsets_[idx];
    auto l      = offset + to_touch_neighbors_at_level(level);
    auto r      = offset + to_touch_neighbors_at_level(level + 1);
    return std::make_pair(l, r);
}

}  // namespace common
}  // namespace annop