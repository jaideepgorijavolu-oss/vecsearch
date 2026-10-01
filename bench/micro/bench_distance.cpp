// Distance kernel microbenchmarks.
//
// Two scenarios per kernel and dimension:
//   Hot  : the same two vectors every call, so both live in L1. Measures pure compute.
//   Scan : one query against a 256 MiB database, read sequentially. Measures what happens when
//          every call has to pull a new vector from DRAM (memory-bandwidth bound).
// "bytes_per_second" counts the floats the kernel reads: 2*dim*4 bytes per call for Hot, and
// dim*4 bytes per call for Scan (the query stays in L1).
#include <benchmark/benchmark.h>

#include <random>
#include <string>
#include <vector>

#include "vecsearch/aligned_allocator.hpp"
#include "vecsearch/distance.hpp"

namespace autovec {
float l2_sq(const float*, const float*, std::size_t);
float dot(const float*, const float*, std::size_t);
}  // namespace autovec
namespace autovec_fastmath {
float l2_sq(const float*, const float*, std::size_t);
float dot(const float*, const float*, std::size_t);
}  // namespace autovec_fastmath

using namespace vecsearch;

namespace {

AlignedVector<float> random_floats(std::size_t n, std::uint32_t seed) {
  std::mt19937 rng(seed);
  std::uniform_real_distribution<float> u(-1.0f, 1.0f);
  AlignedVector<float> v(n);
  for (auto& x : v) x = u(rng);
  return v;
}

void hot(benchmark::State& state, KernelFn fn) {
  const auto dim = static_cast<std::size_t>(state.range(0));
  const auto a = random_floats(dim, 1), b = random_floats(dim, 2);
  for (auto _ : state) {
    benchmark::DoNotOptimize(fn(a.data(), b.data(), dim));
  }
  state.SetBytesProcessed(static_cast<int64_t>(state.iterations() * 2 * dim * sizeof(float)));
}

void scan(benchmark::State& state, KernelFn fn) {
  const auto dim = static_cast<std::size_t>(state.range(0));
  const std::size_t n = (256u << 20) / (dim * sizeof(float));  // 256 MiB >> last-level cache
  static AlignedVector<float> db;
  if (db.size() != n * dim) db = random_floats(n * dim, 3);
  const auto q = random_floats(dim, 4);
  std::size_t i = 0;
  for (auto _ : state) {
    benchmark::DoNotOptimize(fn(q.data(), db.data() + i * dim, dim));
    if (++i == n) i = 0;
  }
  state.SetBytesProcessed(static_cast<int64_t>(state.iterations() * dim * sizeof(float)));
}

void register_all() {
  struct Entry {
    std::string name;
    KernelFn l2, ip;
  };
  std::vector<Entry> entries{
      {"scalar", scalar::l2_sq, scalar::dot},
      {"autovec_O3_native", autovec::l2_sq, autovec::dot},
      {"autovec_O3_native_fastmath", autovec_fastmath::l2_sq, autovec_fastmath::dot},
  };
  for (const Kernels* k : supported_kernels()) {
    if (std::string(k->name) != "scalar") entries.push_back({k->name, k->l2_sq, k->dot});
  }
  for (const auto& e : entries) {
    for (auto [metric, fn] : {std::pair{"l2", e.l2}, std::pair{"dot", e.ip}}) {
      for (const char* mode : {"hot", "scan"}) {
        const std::string name = std::string(mode) + "/" + metric + "/" + e.name;
        auto* b = std::string(mode) == "hot"
                      ? benchmark::RegisterBenchmark(name, hot, fn)
                      : benchmark::RegisterBenchmark(name, scan, fn);
        for (int d : {32, 64, 128, 384, 768, 1536}) b->Arg(d);
      }
    }
  }
}

}  // namespace

int main(int argc, char** argv) {
  register_all();
  benchmark::Initialize(&argc, argv);
  benchmark::RunSpecifiedBenchmarks();
  benchmark::Shutdown();
  return 0;
}
