#pragma once
#include "nexus/utils/lock.hh"
#include "query_resource.hh"
#include "tf_session.hh"

#include <memory>
#include <vector>

#define likely(x) __builtin_expect(!!(x), 1)
#define unlikely(x) __builtin_expect(!!(x), 0)

#include "tensorflow/core/framework/tensor.h"

namespace tensorflow {
class SessionResource {

public:
  explicit SessionResource(int max_session);
  SessionResource(const SessionResource &) = delete;
  SessionResource &operator=(const SessionResource &) = delete;

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

public:
  std::vector<QueryResourcePtr> query_resource;

private:
  int max_session_;
  mutable nexus::utils::Spinlock lock_;
};

using SessionResourcePtr = std::shared_ptr<SessionResource>;
using SessionResourceUPtr = std::unique_ptr<SessionResource>;

} // namespace tensorflow

