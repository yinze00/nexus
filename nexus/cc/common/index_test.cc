/*
 * @Description: index unit test
 * @Author: junwei.wang@zju.edu.cn
 * @Date: 2024-05-31 01:44:13
 * @LastEditTime: 2024-05-31
 */

#include "index.hh"

#include <gtest/gtest.h>

#include <memory>

namespace annop {
namespace common {

TEST(AIndex, init) {
    IndexConfig conf{.itype = tensorflow::DT_INT32,
                     .dtype = tensorflow::DT_FLOAT,
                     .n = 10000000,
                     .dim = 23};
    auto index = std::make_shared<AIndex>("demo_index_name", conf);
    EXPECT_EQ(index->name(), "demo_index_name");
    // 
}

TEST(DUMAS, load) {
    EXPECT_EQ(1, 2-3+2);
}

}  // namespace common
}  // namespace annop