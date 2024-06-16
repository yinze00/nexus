#include <gtest/gtest.h>

#include <cstdint>
#include <random>
#include <vector>

#include "buffer.hh"
#include "embedding.hh"

// 定义一个测试用例
// TEST(ExampleTest, Addition) { EXPECT_EQ(2 + 2, 4); }

// Buffer
namespace annop {
namespace common {

TEST(Buffer, base) {
    std::vector<int8_t> arrs(1000);

    Buffer* buf = new Buffer(arrs.data());

    EXPECT_NE(buf, nullptr);

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<int> dis(-128, 127);

    for (int i = 0; i < arrs.size(); ++i) {
        arrs[i] = static_cast<int8_t>(dis(gen));
    }

    auto p = buf->base<int8_t>();

    for (int i = 0; i < 1000; ++i) {
        EXPECT_EQ(p[i], arrs[i]);
    }

    buf->Unref();
}

TEST(EmbeddingHolder, gather_embedding) {
    EmbeddingHolder holder(tensorflow::DT_INT8, 10, 28);
    auto p = holder.gather_embedding(10);
    EXPECT_NE(p, nullptr);
}

}  // namespace common
}  // namespace annop
