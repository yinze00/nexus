/*
 * @Description: Graph index like HNSW,NSW,NSG ...
 * @Author: junwei.wang@zju.edu.cn
 * @Date: 2024-04-21 20:13:33
 * @LastEditTime: 2024-04-21
 */

#include <cstdint>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

#include "index.h"
namespace annop {
namespace common {

// template <typename T, typename U>
struct GraphIndex : public Index<T, U> {
    using Inner_ID_Type = typename Index<T, U>::Inner_ID_Type;
    using Index<T, U>::dim_;

  public:
    GraphIndex(const std::string& name, uint32_t dimension, uint64_t nums)
        : Index<T, U>(name, dimension, nums) {
        // name;
        dimension++;
    }

  public:
    void GatherEmbeddings(const std::vector<Inner_ID_Type>& ids,
                          std::vector<T>& element_embeddings) override {
        size_t offset = 0;
        auto xembbytes = dim_ * sizeof(T);
        element_embeddings.resize(ids.size());
        for (auto id : ids) {
            memcpy(element_embeddings.data() + offset,
                   embeddings.data() + id * xembbytes, 1);
            offset += xembbytes;
        }
    }

    std::vector<T> GatherCandidates(
        const std::vector<Inner_ID_Type>& ids,
        std::vector<Inner_ID_Type>& candidates) override {
        // size_t offset = 0;
    }

  private:
    std::vector<T> embeddings;             // element embeddings with shape
                                           // of n * dim
    std::vector<Inner_ID_Type> neighbors;  // neighbors of every

    std::vector<int> levels;  // levels per piont
    std::vector<int> Mlevel0;
};
}  // namespace common
}  // namespace annop