#pragma once

#include <faiss/Index.h>
#include <faiss/IndexFlat.h>  // 可能需要其他索引结构的头文件，如IndexFlat
#include <faiss/IndexHNSW.h>
#include <faiss/index_io.h>

#include <memory>
#include <string>

#include "nexus/cc/common/index.hh"

namespace nexus {
namespace common {

using annop::common::AIndex;
using annop::common::AIndexPtr;
using annop::common::IndexConfig;

class Adaptor {
  public:
    virtual faiss::Index* load_index(const std::string& model_path) {
        return faiss::read_index(model_path.c_str(), 0);
    }

    virtual AIndexPtr tansform(const std::string& model_path) = 0;
};

class FaissHNSWAdaptor : public Adaptor {
  public:
    AIndexPtr tansform(const std::string& model_path) override {
        auto index = Adaptor::load_index(model_path);

        auto hnsw = dynamic_cast<faiss::IndexHNSW*>(index);

        auto storage = dynamic_cast<faiss::IndexFlat*>(hnsw->storage);

        IndexConfig conf{.itype = tensorflow::DT_INT32,
                         .dtype = tensorflow::DT_FLOAT,
                         .n = static_cast<uint64_t>(hnsw->ntotal),
                         .dim = hnsw->d,
                         .h = hnsw->hnsw.max_level};

        auto hindex = std::make_shared<AIndex>(std::string("hnsw"), conf);
        LOG(INFO) << hnsw->ntotal << ", " << hnsw->d << " " << hindex->name();

        hindex->embedding_->set_embeddings(storage->get_xb(), conf.dim * conf.n * sizeof(float));

            return hindex;
    }
};
}  // namespace common
}  // namespace nexus
