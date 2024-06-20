#pragma once

#include <atomic>
#include <cassert>
#include <pthread.h>

namespace nexus {
namespace utils {

class ThreadMutex {
public:
  ThreadMutex(const pthread_mutexattr_t *mta = NULL) {
    pthread_mutex_init(&_mutex, mta);
  }

  ~ThreadMutex() { pthread_mutex_destroy(&_mutex); }

  int lock() { return pthread_mutex_lock(&_mutex); }

  int trylock() { return pthread_mutex_trylock(&_mutex); }

  int unlock() { return pthread_mutex_unlock(&_mutex); }

private:
  ThreadMutex(const ThreadMutex &);
  ThreadMutex &operator=(const ThreadMutex &);

protected:
  pthread_mutex_t _mutex;
};

class ScopedLock {
private:
  ThreadMutex &_lock;

private:
  ScopedLock(const ScopedLock &);
  ScopedLock &operator=(const ScopedLock &);

public:
  explicit ScopedLock(ThreadMutex &lock) : _lock(lock) {
    int ret = _lock.lock();
    assert(ret == 0);
    (void)ret;
  }

  ~ScopedLock() {
    int ret = _lock.unlock();
    assert(ret == 0);
    (void)ret;
  }
};

class Spinlock {
public:
  Spinlock() : flag_(0) {}

  void lock() {
    while (flag_.test_and_set(std::memory_order_acquire))
      ;
  }

  void unlock() { flag_.clear(std::memory_order_release); }

private:
  std::atomic_flag flag_;
};

class ScopedSpinlock {
public:
  ScopedSpinlock(Spinlock &spinlock) : lock_(spinlock) { lock_.lock(); }

  ~ScopedSpinlock() { lock_.unlock(); }

private:
  Spinlock &lock_;
};

} // namespace utils
} // namespace nexus