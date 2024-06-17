/*
 * @Description: step id generator
 * @Author: junwei.wang@zju.edu.cn
 * @Date: 2024-06-16 17:06:53
 * @LastEditTime: 2024-06-16
 */

#pragma once

#include <memory>
#include <mutex>

namespace nexus {
namespace common {

struct RunIDAllocator {
public:
  RunIDAllocator(const RunIDAllocator &) = delete;
  RunIDAllocator &operator=(const RunIDAllocator &) = delete;

public:
  void init(size_t);
  
};

} // namespace common
} // namespace nexus