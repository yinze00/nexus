/*
 * @Description: Graph index like HNSW,NSW,NSG ...
 * @Author: junwei.wang@zju.edu.cn
 * @Date: 2024-04-21 20:13:33
 * @LastEditTime: 2024-04-21
 */

#include <cstdint>
#include <memory>
#include <string>

#include "index.h"
namespace annop {
namespace common {

template <typename T, typename U>
class IVFIndex : public Index<T, U> {
  public:
    IVFIndex(const std::string& name) : Index<T, U>(name) {}

  private:
};
}  // namespace common
}  // namespace annop