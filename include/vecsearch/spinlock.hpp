#pragma once

#include <atomic>
#include <cstdint>
#include <thread>

namespace vecsearch {

// A one-byte lock for the per-node locks in HNSW.
//
// HNSW construction holds a node's lock only while copying or rewriting its neighbor list
// (well under a microsecond), and contention on any single node is rare, so spinning is
// cheaper than the futex path of std::mutex. It is also 1 byte instead of 40, which matters
// with one lock per node (40 MB of mutexes for 1M vectors).
//
// Test-and-test-and-set: spin on a plain load (which stays in the local cache) and only try the
// exchange when the lock looks free, so waiting threads do not bounce the cache line.
class SpinLock {
 public:
  void lock() noexcept {
    for (int spins = 0;; ++spins) {
      if (!flag_.exchange(1, std::memory_order_acquire)) return;
      while (flag_.load(std::memory_order_relaxed)) {
        if (++spins > 64) std::this_thread::yield();  // holder was probably preempted
      }
    }
  }
  void unlock() noexcept { flag_.store(0, std::memory_order_release); }

 private:
  std::atomic<std::uint8_t> flag_{0};
};

}  // namespace vecsearch
