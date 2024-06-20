#include "session_resource.hh"

namespace tensorflow {

SessionResource::SessionResource(int max_session) : max_session_(max_session) {
  query_resource.resize(max_session_);
}

void SessionResource::add_query_resource(int64_t run_id,
                                         QueryResourcePtr query) {
  if (unlikely(run_id < 0 && run_id > max_session_))
    return;
  query_resource[run_id] = query;
}

void SessionResource::remove_query_resource(int64_t run_id) {
  if (unlikely(run_id < 0 && run_id > max_session_))
    return;
  query_resource[run_id] = nullptr;
}

} // namespace tensorflow