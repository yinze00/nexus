#include "adaptor.hh"

#include <gtest/gtest.h>

#include <cstddef>
#include <string>
#include <vector>

namespace annop {
namespace common {

TEST(Adaptor, load_index) {
    std::string hnsw_path =
        "/home/yinze/dev/zenith/nexus/nexus/data/hnsw_model/data/"
        "hnsw_10000.dat";

    auto adaptor = new nexus::common::FaissHNSWAdaptor();

    std::vector<float> a;
    a.reserve(10000 * 64);
    auto index = adaptor->tansform(hnsw_path);


    delete adaptor;
    EXPECT_NE(nullptr, index);
}
}  // namespace common
}  // namespace annop