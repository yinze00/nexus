#pragma once
#include <functional>
#include <iostream>

namespace nexus {
namespace common {

template <typename T, typename Deleter = std::function<void(T*)>>
class ScopeDeleter {
  public:
    ScopeDeleter(
        T* ptr, Deleter deleter = [](T* p) { delete p; })
        : _ptr(ptr), _deleter(deleter) {}

    ScopeDeleter(const ScopeDeleter&) = delete;
    ScopeDeleter& operator=(const ScopeDeleter&) = delete;

    ScopeDeleter(ScopeDeleter&& other)
        : _ptr(other._ptr), _deleter(std::move(other._deleter)) {
        other._ptr = nullptr;
    }

    ScopeDeleter& operator=(ScopeDeleter&& other) {
        if (this != &other) {
            reset();
            _ptr = other._ptr;
            _deleter = std::move(other._deleter);
            other._ptr = nullptr;
        }
        return *this;
    }

    ~ScopeDeleter() { reset(); }

    void reset(T* ptr = nullptr) {
        if (_ptr) {
            _deleter(_ptr);
        }
        _ptr = ptr;
    }

    T* get() const { return _ptr; }

    T* operator->() const { return _ptr; }

  private:
    T* _ptr;
    Deleter _deleter;
};

// // 示例用法
// int main() {
//     // 使用默认的delete deleter
//     {
//         ScopeDeleter<int> scopePtr(new int(42));
//         std::cout << *scopePtr << std::endl;
//     }  // 离开作用域时自动删除int指针

//     // 使用自定义的deleter
//     {
//         FILE* file = fopen("example.txt", "w");
//         ScopeDeleter<FILE> scopeFile(file, [](FILE* f) {
//             if (f) {
//                 fclose(f);
//                 std::cout << "File closed" << std::endl;
//             }
//         });
//         // 在这里可以使用file
//     }  // 离开作用域时自动关闭文件

//     return 0;
// };

}  // namespace common
}  // namespace nexus