/*
 * @Description: GraphIdex impl
 * @Author: junwei.wang@zju.edu.cn
 * @Date: 2024-04-25 02:26:43
 * @LastEditTime: 2024-04-25
 */

#include "graph_index.hh"

#include <cstdint>

namespace annop {
namespace common {

template struct GraphIndex<uint8_t, uint32_t>;

template struct GraphIndex<float, uint64_t>;
template struct GraphIndex<float, uint32_t>;

template struct GraphIndex<double, uint64_t>;
template struct GraphIndex<double, uint32_t>;

}  // namespace common
}  // namespace annop
