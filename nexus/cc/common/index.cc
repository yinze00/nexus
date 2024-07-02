/*
 * @Description: index.cc
 * @Author: junwei.wang@zju.edu.cn
 * @Date: 2024-04-29 00:08:25
 * @LastEditTime: 2024-04-29
 */
#include "index.hh"

#include <cstdint>
#include <memory>

#include "buffer.hh"
#include "embedding.hh"
#include "graph.hh"
#include "tensorflow/core/framework/types.h"

namespace annop {
namespace common {

AIndex::AIndex(const std::string& name, const IndexConfig& conf) : name_(name) {
    neis_ = std::make_unique<HGraph>(conf.itype, conf.n, conf.nn);
    embedding_ =
        std::make_unique<EmbeddingHolder>(conf.dtype, conf.n, conf.dim);
}

AIndex::AIndex(int n, int dim) {
    neis_ = std::make_unique<HGraph>(DataType::DT_UINT32, n, n * dim);
    embedding_ = std::make_unique<EmbeddingHolder>(DataType::DT_FLOAT, n, dim);
}

}  // namespace common
}  // namespace annop