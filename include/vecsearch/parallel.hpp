#pragma once

#include <algorithm>
#include <atomic>
#include <cstddef>
#include <exception>
#include <mutex>
#include <thread>
#include <vector>

namespace vecsearch {

// 0 means "all hardware threads".
inline std::size_t resolve_threads(std::size_t requested) {
  if (requested != 0) return requested;
  return std::max<std::size_t>(1, std::thread::hardware_concurrency());
}

// Calls fn(i, worker) for every i in [0, n), using up to `num_threads` threads.
//
// Work is handed out dynamically in chunks from a shared atomic counter, so threads that get
// cheap items keep pulling more (HNSW queries vary a lot in cost). `worker` is in
// [0, num_threads) and is stable for the duration of one call, so callers can index per-thread
// scratch buffers with it. The calling thread acts as worker 0. The first exception thrown by
// any fn is rethrown here after all threads have joined.
//
// Threads are created per call rather than kept in a pool: a batch of thousands of queries costs
// milliseconds, while starting 16 threads costs tens of microseconds, and there is no idle pool
// state to reason about.
template <class Fn>
void parallel_for(std::size_t n, std::size_t num_threads, Fn&& fn) {
  num_threads = std::min(resolve_threads(num_threads), std::max<std::size_t>(n, 1));
  if (num_threads <= 1) {
    for (std::size_t i = 0; i < n; ++i) fn(i, std::size_t{0});
    return;
  }
  const std::size_t chunk = std::clamp<std::size_t>(n / (num_threads * 16), 1, 256);
  std::atomic<std::size_t> next{0};
  std::exception_ptr error;
  std::mutex error_mu;

  auto run = [&](std::size_t worker) {
    try {
      for (;;) {
        const std::size_t begin = next.fetch_add(chunk, std::memory_order_relaxed);
        if (begin >= n) break;
        const std::size_t end = std::min(n, begin + chunk);
        for (std::size_t i = begin; i < end; ++i) fn(i, worker);
      }
    } catch (...) {
      std::lock_guard lock(error_mu);
      if (!error) error = std::current_exception();
      next.store(n, std::memory_order_relaxed);  // stop handing out work
    }
  };

  std::vector<std::thread> threads;
  threads.reserve(num_threads - 1);
  for (std::size_t t = 1; t < num_threads; ++t) threads.emplace_back(run, t);
  run(0);
  for (auto& t : threads) t.join();
  if (error) std::rethrow_exception(error);
}

}  // namespace vecsearch
