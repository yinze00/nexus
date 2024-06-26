/*
 * @Description: step id generator
 * @Author: junwei.wang@zju.edu.cn
 * @Date: 2024-06-16 17:06:53
 * @LastEditTime: 2024-06-16
 */

#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <vector>

namespace nexus {
namespace common {

struct RunIDAllocator {
  public:
    RunIDAllocator(const RunIDAllocator&) = delete;
    RunIDAllocator& operator=(const RunIDAllocator&) = delete;

  public:
    void init(size_t);
    int64_t get();
    void put(int64_t);

  private:
    size_t max_session_{1024};
    // std::vector<int64_t> idle_sess_ids_;

    std::vector<bool> bitmap_;

    mutable std::mutex mtx_;
};

using RunIDAllocatorPtr = std::shared_ptr<RunIDAllocator>;

}  // namespace common
}  // namespace nexus