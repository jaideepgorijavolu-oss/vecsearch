#include <gtest/gtest.h>

#include <atomic>
#include <stdexcept>
#include <vector>

#include "vecsearch/parallel.hpp"

using namespace vecsearch;

TEST(ParallelFor, VisitsEveryIndexOnce) {
  for (std::size_t threads : {1u, 2u, 7u, 16u}) {
    std::vector<std::atomic<int>> hits(1000);
    parallel_for(hits.size(), threads, [&](std::size_t i, std::size_t worker) {
      EXPECT_LT(worker, threads);
      hits[i].fetch_add(1);
    });
    for (auto& h : hits) EXPECT_EQ(h.load(), 1);
  }
}

TEST(ParallelFor, ZeroItems) {
  int calls = 0;
  parallel_for(0, 4, [&](std::size_t, std::size_t) { ++calls; });
  EXPECT_EQ(calls, 0);
}

TEST(ParallelFor, PropagatesException) {
  EXPECT_THROW(parallel_for(100, 4,
                            [](std::size_t i, std::size_t) {
                              if (i == 42) throw std::runtime_error("boom");
                            }),
               std::runtime_error);
}
