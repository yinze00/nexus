#include "heap.hh"

#include <gtest/gtest.h>

#include <cstdint>
#include <random>
#include <vector>

#include "buffer.hh"
#include "embedding.hh"
#include "result_handler.hh"

namespace annop {
using CMx = CMax<float, uint32_t>;
using CMn = CMin<float, uint32_t>;

TEST(CMAXMIN, maxmin) {
    EXPECT_EQ(true, CMn::cmp(1.0, 2.0));
    EXPECT_EQ(true, CMx::cmp(2.0, 1.0));
}

TEST(Heap, Heapify) {
    int                   n = 8;
    std::vector<float>    s(n);
    std::vector<uint32_t> l(n);

    auto print = [&]() {
        std::stringstream ss;
        for (auto i = 0; i < n; ++i) {
            ss << i << ": " << l[i] << "\t" << s[i] << "\n";
        }
        VLOG(1) << " Printing ...";
        std::cout << ss.str();
    };

    HeapResultHandler<CMn> hd(n, s.data(), l.data());

    print();

    hd.add_result(0.1, 224);
    hd.add_result(0.2, 225);

    print();

    EXPECT_TRUE(hd.add_result(0.1, 1));
    hd.add_result(0.2, 2);
    hd.add_result(0.3, 3);
    hd.add_result(0.4, 4);
    hd.add_result(0.5, 5);
    hd.add_result(0.6, 6);
    hd.add_result(0.7, 7);
    hd.add_result(0.8, 8);
    print();

    EXPECT_TRUE(hd.add_result(0.55, 88));
    print();
    EXPECT_FALSE(hd.add_result(0.01, 99));

    print();
    EXPECT_TRUE(hd.add_result(0.9, 9));
    print();
}

// For MiniMaxHeap

// 测试构造函数
TEST(MinimaxHeapTest, Constructor) {
    MinimaxHeap heap(5);
    EXPECT_EQ(heap.size(), 0);
    EXPECT_EQ(heap.max(),
              0);  // 初始状态下 max() 的值未定义，但这里我们假设初始为 0
}

void print_heap(MinimaxHeap& heap) { VLOG(1) << heap.print(); }

// 测试插入和最大值
TEST(MinimaxHeapTest, PushAndMax) {
    MinimaxHeap heap(5);
    heap.push(1, 10.0f);
    EXPECT_EQ(heap.size(), 1);
    EXPECT_EQ(heap.max(), 10.0f);

    heap.push(2, 20.0f);
    EXPECT_EQ(heap.size(), 2);
    EXPECT_EQ(heap.max(), 20.0f);

    heap.push(3, 5.0f);
    EXPECT_EQ(heap.size(), 3);
    EXPECT_EQ(heap.max(), 20.0f);
}

// 测试插入超出容量
TEST(MinimaxHeapTest, PushOverCapacity) {
    MinimaxHeap heap(3);
    heap.push(1, 10.0f);
    heap.push(2, 20.0f);
    heap.push(3, 5.0f);
    EXPECT_EQ(heap.size(), 3);
    EXPECT_EQ(heap.max(), 20.0f);

    print_heap(heap);

    // 插入一个值大于当前最大值
    heap.push(4, 25.0f);
    print_heap(heap);

    EXPECT_EQ(heap.size(), 3);
    EXPECT_EQ(heap.max(), 25.0f);

    // 插入一个值小于当前最大值
    heap.push(5, 15.0f);
    EXPECT_EQ(heap.size(), 3);
    EXPECT_EQ(heap.max(), 25.0f);
}

// 测试弹出最小值
TEST(MinimaxHeapTest, PopMin) {
    MinimaxHeap heap(5);
    heap.push(1, 10.0f);
    heap.push(2, 20.0f);
    heap.push(3, 5.0f);

    float vmin;
    int   id = heap.pop_min(&vmin);
    EXPECT_EQ(id, 3);
    EXPECT_EQ(vmin, 5.0f);
    EXPECT_EQ(heap.size(), 2);

    id = heap.pop_min(&vmin);
    EXPECT_EQ(id, 1);
    EXPECT_EQ(vmin, 10.0f);
    EXPECT_EQ(heap.size(), 1);
}

// 测试清空堆
TEST(MinimaxHeapTest, Clear) {
    MinimaxHeap heap(5);
    heap.push(1, 10.0f);
    heap.push(2, 20.0f);
    EXPECT_EQ(heap.size(), 2);

    heap.clear();
    EXPECT_EQ(heap.size(), 0);
}

// 测试统计小于给定阈值的元素数目
TEST(MinimaxHeapTest, CountBelow) {
    MinimaxHeap heap(5);
    heap.push(1, 10.0f);
    heap.push(2, 20.0f);
    heap.push(3, 5.0f);
    heap.push(4, 15.0f);
    heap.push(5, 8.0f);

    EXPECT_EQ(heap.count_below(10.0f), 2);  // 5.0f 和 8.0f 小于 10.0f
    EXPECT_EQ(heap.count_below(15.0f),
              4);  // 5.0f、8.0f、10.0f 和 15.0f 小于 15.0f
    EXPECT_EQ(heap.count_below(5.0f), 0);  // 没有小于 5.0f 的值
}

// // main 函数，用于运行所有测试
// int main(int argc, char **argv) {
//     ::testing::InitGoogleTest(&argc, argv);
//     return RUN_ALL_TESTS();
// }

}  // namespace annop