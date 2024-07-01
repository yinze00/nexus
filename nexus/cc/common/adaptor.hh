#pragma once

#include <faiss/Index.h>
#include <faiss/IndexFlat.h>  // 可能需要其他索引结构的头文件，如IndexFlat
#include <faiss/IndexHNSW.h>
#include <faiss/index_io.h>

#include <cstdint>
#include <memory>
#include <string>

#include "graph.hh"
#include "nexus/cc/common/index.hh"

#define SSTR(x) std::string(#x) << ": " << std::to_string(x) << " "

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

        hindex->embedding_->set_embeddings(storage->get_xb(),
                                           conf.dim * conf.n * sizeof(float));

        {
            // prepare h-nsw neis
            for (auto level = 0; level < conf.h; ++level) {
                // auto neis =
                //     std::make_shared<annop::common::LinkedList<uint32_t>>();
                auto cum_nb_neighbors = hnsw->hnsw.cum_nb_neighbors(level);

                auto nb_neighbors = hnsw->hnsw.nb_neighbors(level);

                // auto nb_num_this_level = hnsw->hnsw.
                LOG(INFO) << SSTR(level) << SSTR(cum_nb_neighbors)
                          << SSTR(nb_neighbors);

                auto levels = hnsw->hnsw.levels;

                auto neighbors = hnsw->hnsw.neighbors;

                for (auto offset = 0; offset < levels.size(); ++offset) {
                    if (levels[offset] > level) {
                        size_t begin, end;
                        hnsw->hnsw.neighbor_range(offset, level, &begin, &end);
                        std::unordered_set<int> neighset;
                        for (size_t j = begin; j < end; j++) {
                            if (neighbors[j] < 0) break;
                            LOG(INFO) << SSTR(neighbors[j]);
                            neighset.insert(neighbors[j]);
                        }
                    }
                }

                hnsw->hnsw.print_neighbor_stats(level);
            }
        }

        return hindex;
    }
};
}  // namespace common
}  // namespace nexus
