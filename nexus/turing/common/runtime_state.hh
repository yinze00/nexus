#pragma once
#include <memory>
#include <queue>
#include <vector>

#include "nexus/cc/common/heap.hh"
#include "nexus/turing/proto/run_graph.pb.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/protobuf/config.pb.h"

namespace tensorflow {

class RuntimeState {
  public:
    RuntimeState()  = default;
    ~RuntimeState() = default;

    // disable assignment
    RuntimeState(const RuntimeState&)            = delete;
    RuntimeState& operator=(const RuntimeState&) = delete;

  public:
    struct AscentCompare {
        bool operator()(const std::pair<uint32_t, float>& a,
                        const std::pair<uint32_t, float>& b) {
            return a.second > b.second;  // 最小堆，float 小的优先
        }
    };

    struct DescentCompare {
        bool operator()(const std::pair<uint32_t, float>& a,
                        const std::pair<uint32_t, float>& b) {
            return a.second < b.second;  // 最大堆，float 小的优先
        }
    };

    // struct MinHeap {
    //     using HC = annop::CMax<float, uint32_t>;
    //     explicit MinHeap(int n) : n(n), k(0), nvalid(0), labels(n), scores(n)
    //     {}

    //     int n;
    //     int k;
    //     int nvalid;

    //     std::vector<uint32_t> labels;
    //     std::vector<float>    scores;

    //     void push(uint32_t i, float v) {
    //         if (k == n) {
    //             if (v >= scores[0]) return;
    //             if (labels[0] != -1) {
    //                 --nvalid;
    //             }
    //             annop::heap_pop<HC>(k--, scores.data(), labels.data());
    //         }
    //         annop::heap_push<HC>(++k, scores.data(), labels.data(), v, i);
    //         ++nvalid;
    //     }
    // };

    struct VisitedTable {
        std::vector<uint8_t> visited;
        uint8_t              visno;

        explicit VisitedTable(int size) : visited(size), visno(1) {}

        void set(int no) { visited[no] = visno; }

        bool get(int no) const { return visited[no] == visno; }

        void advance() {
            visno++;
            if (visno == 250) {
                // 250 rather than 255 because sometimes we use visno and
                // visno+1
                memset(visited.data(), 0, sizeof(visited[0]) * visited.size());
                visno = 1;
            }
        }
    };

    // maxheap
    std::priority_queue<std::pair<uint32_t, float>,
                        std::vector<std::pair<uint32_t, float>>, DescentCompare>
        candidates;

    // results
    std::priority_queue<std::pair<uint32_t, float>,
                        std::vector<std::pair<uint32_t, float>>, AscentCompare>
        results;

  public:
    //@getter
    const RunOptions& getRunOptions() const { return run_options_; }
    RunOptions&       getRunOptions() { return run_options_; }

    void setRunId(int64_t run_id) {
        run_options_.set_run_id(run_id);
        run_id_ = run_id;
    }

    //@setter
    void setRunOptions(const RunOptions& run_options) {
        run_options_ = run_options;
    }

    void addRunMetaData(nexus::turing::NamedRunMetadata*);
    const std::vector<std::shared_ptr<nexus::turing::NamedRunMetadata>>&
    getRunMetadata() const {
        return run_metas_;
    }
    std::vector<std::shared_ptr<nexus::turing::NamedRunMetadata>>&
    getRunMetadata() {
        return run_metas_;
    }

  private:
    std::vector<std::shared_ptr<nexus::turing::NamedRunMetadata>> run_metas_;

    RunOptions run_options_;
    int64_t    run_id_;
};

using RuntimeStatePtr  = std::shared_ptr<RuntimeState>;
using RuntimeStateUPtr = std::unique_ptr<RuntimeState>;

}  // namespace tensorflow