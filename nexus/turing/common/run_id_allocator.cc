#include "run_id_allocator.hh"

#include <cstdint>
#include <string>

#include "session_resource.hh"

namespace nexus {
namespace common {
void RunIDAllocator::init(size_t d) {
    max_session_ = d;
    bitmap_.resize(max_session_ + 1, false);
}

int64_t RunIDAllocator::get() {
    std::lock_guard<std::mutex> lock(mtx_);
    for (int64_t i = 1; i <= max_session_; ++i) {
        if (!bitmap_[i]) {
            bitmap_[i] = true;
            LOG(INFO) << "get " << i;
            return i;
        }
    }
    return -1;
}

void RunIDAllocator::put(int64_t id) {
    std::lock_guard<std::mutex> lock(mtx_);

    LOG(INFO) << "put " << id;

    if (likely(id >= 0 && id < max_session_ && bitmap_[id])) {
        bitmap_[id] = false;
        return;
    }

    throw std::runtime_error("Invalid ID " + std::to_string(id));
}

}  // namespace common
}  // namespace nexus