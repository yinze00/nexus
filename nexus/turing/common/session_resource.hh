#pragma once
#include <cstddef>
#include <memory>
#include <vector>

#include "nexus/cc/common/IndexCommonSingleton.hh"
#include "nexus/cc/common/index.hh"
#include "nexus/cc/common/singleton.h"
#include "nexus/utils/lock.hh"
#include "query_resource.hh"
#include "tf_session.hh"

#define likely(x) __builtin_expect(!!(x), 1)
#define unlikely(x) __builtin_expect(!!(x), 0)

#include "tensorflow/core/framework/tensor.h"

namespace tensorflow {

using annop::common::AIndexPtr;
using annop::common::ANNIndexHolder;

class SessionResource {
  public:
    explicit SessionResource(int max_session);
    SessionResource(const SessionResource&)            = delete;
    SessionResource& operator=(const SessionResource&) = delete;

  public:
    struct IndexManager {
        AIndexPtr get_index(const std::string& model) {
            return annop::common::CSingleton<ANNIndexHolder>::instance()
                ->get_index(model);
        }
    };

  public:
    void add_query_resource(int64_t, QueryResourcePtr);

    void remove_query_resource(int64_t);

    inline QueryResourcePtr get_query_resource(int64_t run_id) {
        if (likely(run_id >= 0 && run_id < query_resource.size())) {
            return query_resource[run_id];
        } else {
            LOG(ERROR) << "Failed to get query_resource " << run_id;
            return nullptr;
        }
    }

    AIndexPtr get_index(const std::string& name) {
        return indexmgr_.get_index(name);
    }

  public:
    std::vector<QueryResourcePtr> query_resource;
    IndexManager                  indexmgr_;

  private:
    int                            max_session_;
    mutable nexus::utils::Spinlock lock_;
};

using SessionResourcePtr  = std::shared_ptr<SessionResource>;
using SessionResourceUPtr = std::unique_ptr<SessionResource>;

}  // namespace tensorflow
