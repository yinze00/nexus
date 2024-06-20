#include "runtime_state.hh"
#include <memory>

namespace tensorflow {
void RuntimeState::addRunMetaData(nexus::turing::NamedRunMetadata *data) {
  if (data) {
    run_metas_.push_back(
        std::shared_ptr<nexus::turing::NamedRunMetadata>(data));
  }
}
} // namespace tensorflow