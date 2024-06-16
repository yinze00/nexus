/*
 * @Description: ann index data holder
 * @Author: junwei.wang@zju.edu.cn
 * @Date: 2024-04-01 16:17:13
 * @LastEditTime: 2024-04-01
 */
#include <iostream>
#include <mutex>
#include <string_view>
#include <unordered_map>

#include "index.hh"
namespace annop {
namespace common {

// will be a singleton...
struct ANNIndexHolder {
  public:
    virtual bool set_index(const std::string& index_name, AIndexPtr& index) {
        std::lock_guard<std::mutex> lock(mtx_);
        bool is_updated = index_map_.find(index_name) != index_map_.end();
        index_map_[index_name] = index;
        return is_updated;
    }
    virtual AIndexPtr get_index(const std::string& index_name) {
        if (auto it = index_map_.find(index_name); it != index_map_.end()) {
            return it-> second;
        }
        return nullptr;
    }
    virtual bool get_index(const std::string& index_name, AIndexPtr& index) {
        index = get_index(index_name);
        return nullptr == index;
    }

  private:
    std::mutex mtx_;
    std::unordered_map<std::string, AIndexPtr> index_map_ GUARDED_BY(mtx_);
};

}  // namespace common
}  // namespace annop