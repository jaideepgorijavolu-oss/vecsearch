// How much does a parallel build hurt graph quality, and does the sequential prefix fix it?
// For each configuration, counts how many of the first 2000 inserted vectors (and the last
// 2000) are returned as their own nearest neighbor at ef = 50. Random Gaussian data, dim 16.
//
// Usage: parallel_build_quality [n=100000]
#include <cstdio>
#include <random>
#include <string>
#include <vector>

#include "vecsearch/hnsw_index.hpp"

using namespace vecsearch;

int main(int argc, char** argv) {
  const std::size_t n = argc > 1 ? std::stoul(argv[1]) : 100000, dim = 16, probe = 2000;
  std::mt19937 rng(10);
  std::normal_distribution<float> nd;
  std::vector<float> base(n * dim);
  for (auto& x : base) x = nd(rng);

  std::printf("n %zu, dim %zu, M 16, ef_construction 200, self-hits at ef 50 (of %zu)\n", n, dim,
              probe);
  std::printf("%8s %18s %16s %16s\n", "threads", "sequential_prefix", "first 2000", "last 2000");
  for (std::size_t threads : {1u, 8u, 16u}) {
    for (std::size_t prefix : {1u, 1000u}) {
      if (threads == 1 && prefix == 1000) continue;  // identical to prefix 1
      HnswParams p;
      p.sequential_prefix = prefix;
      HnswIndex index(dim, Metric::L2, p);
      index.add(base.data(), n, nullptr, threads);
      const auto first = index.search(base.data(), probe, 1, 50);
      const auto last = index.search(base.data() + (n - probe) * dim, probe, 1, 50);
      std::size_t hf = 0, hl = 0;
      for (std::size_t i = 0; i < probe; ++i) {
        hf += first.ids[i] == static_cast<std::int64_t>(i);
        hl += last.ids[i] == static_cast<std::int64_t>(n - probe + i);
      }
      std::printf("%8zu %18zu %16zu %16zu\n", threads, prefix, hf, hl);
    }
  }
}
