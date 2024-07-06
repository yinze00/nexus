#pragma once

#include <faiss/Index.h>
#include <faiss/IndexFlat.h>  // 可能需要其他索引结构的头文件，如IndexFlat
#include <faiss/IndexHNSW.h>
#include <faiss/index_io.h>

#include <cstdint>
#include <cstring>
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
                         .n     = static_cast<uint64_t>(hnsw->ntotal),
                         .dim   = hnsw->d,
                         .h     = hnsw->hnsw.max_level,
                         .nn    = hnsw->hnsw.neighbors.size()};

        auto hindex = std::make_shared<AIndex>(std::string("hnsw"), conf);
        LOG(INFO) << hnsw->ntotal << ", " << hnsw->d << " " << hindex->name();

        hindex->embedding_->set_embeddings(storage->get_xb(),
                                           conf.dim * conf.n * sizeof(float));

        auto hgraph = hindex->neis_.get();

        hgraph->levels_             = hnsw->hnsw.levels;
        hgraph->ones_neis_at_level_ = hnsw->hnsw.cum_nneighbor_per_level;
        hgraph->offsets_            = hnsw->hnsw.offsets;
        hgraph->n_                  = hnsw->ntotal;
        hgraph->entry_point         = hnsw->hnsw.entry_point;

        LOG(INFO) << "neighbors: " << hnsw->hnsw.neighbors.size();
        LOG(INFO) << "cum_neis: " << hnsw->hnsw.cum_nneighbor_per_level.size();
        LOG(INFO) << "ones_neis_at_level.size = "
                  << hgraph->ones_neis_at_level_.size();

        // int i = 0;
        // for (auto val : hgraph->ones_neis_at_level_) {
        //     LOG(INFO) << "i " << i << " " << val;
        //     i++;
        // }

        auto dst = hgraph->h_linklist_->base<uint32_t>();
        std::memcpy(dst, hnsw->hnsw.neighbors.data(),
                    conf.nn * sizeof(uint32_t));
        // hgraph->set_h_linklist(HierachyLinkedListUPtrType &&ptr)

        // hgraph->h_linklist_.reset(
        //     new annop::common::HierachyLinkedList<uint32>(n_,
        //     hnsw->hnsw.neighbors.size() * sizeof(uint32_t))
        // )
        // std::make_unique<annop::common::HierachyLinkedList<uint32_t>,
        // annop::common::HierachyLinkedListDeleter<uint32_t>>(
        //     (uint32_t)hnsw->ntotal, hnsw->hnsw.neighbors.size() *
        //     sizeof(uint32_t));
        {
            // prepare h-nsw neis
            for (auto level = 0; level <= conf.h; ++level) {
                auto cum_nb_neighbors = hnsw->hnsw.cum_nb_neighbors(level);

                auto nb_neighbors = hnsw->hnsw.nb_neighbors(level);

                VLOG(1) << SSTR(level) << SSTR(cum_nb_neighbors)
                        << SSTR(nb_neighbors);

                // auto& levels = hnsw->hnsw.levels;

                // auto& neighbors = hnsw->hnsw.neighbors;

                // for (auto offset = 0; offset < levels.size(); ++offset) {
                //     if (levels[offset] > level) {
                //         size_t begin, end;
                //         hnsw->hnsw.neighbor_range(offset, level, &begin,
                //         &end);
                //         // std::unordered_set<int> neighset;
                //         for (size_t j = begin; j < end; j++) {
                //             if (neighbors[j] < 0) break;
                //             // LOG(INFO) << SSTR(neighbors[j]);
                //             // neighset.insert(neighbors[j]);
                //         }
                //     }
                // }
                if (VLOG_IS_ON(1)) hnsw->hnsw.print_neighbor_stats(level);
            }
        }

        return hindex;
    }
};
}  // namespace common
}  // namespace nexus
