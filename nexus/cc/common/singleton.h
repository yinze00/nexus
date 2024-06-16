/*
 * @Description: Singleton for stateful data
 * @Author: junwei.wang@zju.edu.cn
 * @Date: 2024-04-01 15:46:05
 * @LastEditTime: 2024-04-01
 */

#pragma once

#include <iostream>
#include <memory>
#include <mutex>

namespace annop {
namespace common {

template <typename T>
class Singleton {
  public:
    Singleton(const Singleton&) = delete;
    Singleton& operator=(const Singleton&) = delete;

    static T& instance() {
        std::call_once(initflags, &Singleton::initSingleton);
        return *instance_;
    }

    static T* pinstance() {
        std::call_once(initflags, &Singleton::initSingleton);
        return instance_.get();
    }

  private:
    Singleton() = default;
    ~Singleton() = default;

    static void initSingleton() { instance_.reset(new T); }

  private:
    static std::unique_ptr<T> instance_;
    static std::once_flag initflags;
};

template <typename Type>
class CSingleton {
  public:
    static Type* instance() {
        if (NULL == instance_) {
            std::lock_guard<std::mutex> lock(mutex_);
            if (NULL == instance_) {
                try {
                    instance_ = new Type();
                } catch (...) {
                    instance_ = NULL;
                    std::cerr << "Singleton::m_instance new exception";
                }
            }
            destructor_.nothing();  // force m_destructor to instantiated
            mutex_.unlock();
        }
        return instance_;
    }
    template <typename TArg1>
    static Type* instance(const TArg1& arg1) {
        if (NULL == instance_) {
            std::lock_guard<std::mutex> lock(mutex_);
            if (NULL == instance_) {
                try {
                    instance_ = new Type(arg1);
                } catch (...) {
                    instance_ = NULL;
                    std::cerr << "Singleton::m_instance new exception";
                }
            }
            destructor_.nothing();  // force m_destructor to instantiated
        }
        return instance_;
    }
    template <typename TArg1, typename TArg2>
    static Type* instance(const TArg1& arg1, const TArg2& arg2) {
        if (NULL == instance_) {
            std::lock_guard<std::mutex> lock(mutex_);
            if (NULL == instance_) {
                try {
                    instance_ = new Type(arg1, arg2);
                } catch (...) {
                    instance = NULL;
                    std::cerr << "Singleton::m_instance new exception";
                }
            }
            destructor_.nothing();  // force m_destructor to instantiated
        }
        return instance_;
    }
    template <typename TArg1, typename TArg2, typename TArg3>
    static Type* instance(const TArg1& arg1, const TArg2& arg2,
                          const TArg3& arg3) {
        if (NULL == instance_) {
            std::lock_guard<std::mutex> lock(mutex_);
            if (NULL == instance_) {
                try {
                    instance_ = new Type(arg1, arg2, arg3);
                } catch (...) {
                    instance_ = NULL;
                    std::cerr << "Singleton::m_instance new exception";
                }
            }
            destructor_.nothing();  // force m_destructor to instantiated
        }
        return instance_;
    }
    template <typename TArg1, typename TArg2, typename TArg3, typename TArg4>
    static Type* instance(const TArg1& arg1, const TArg2& arg2,
                          const TArg3& arg3, const TArg4& arg4) {
        if (NULL == instance_) {
            std::lock_guard<std::mutex> lock(mutex_);
            if (NULL == instance_) {
                try {
                    instance_ = new Type(arg1, arg2, arg3, arg4);
                } catch (...) {
                    instance_ = NULL;
                    std::cerr << "Singleton::m_instance new exception";
                }
            }
            destructor_.nothing();  // force m_destructor to instantiated
        }
        return instance_;
    }

  private:
    class Destructor {
      public:
        Destructor() {}
        ~Destructor() {
            if (CSingleton<Type>::instance_ != NULL) {
                delete CSingleton<Type>::instance_;
            }
        }
        void nothing() {}
    };

  private:
    static Type* instance_;
    static std::mutex mutex_;
    static Destructor destructor_;

  private:
    CSingleton();
    ~CSingleton();
    CSingleton(const CSingleton&);
    void operator=(const CSingleton&);
    void operator=(const CSingleton&) const;
    CSingleton* operator&();
    const CSingleton* operator&() const;
};
template <typename Type>
Type* CSingleton<Type>::instance_ = NULL;
template <typename Type>
typename CSingleton<Type>::Destructor CSingleton<Type>::destructor_;
template <typename Type>
std::mutex CSingleton<Type>::mutex_;

}  // namespace common
}  // namespace annop