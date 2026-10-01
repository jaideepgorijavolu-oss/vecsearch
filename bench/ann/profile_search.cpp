// Single-threaded search loop for profiling (perf stat / perf record).
//
// Usage: profile_search base.fbin query.fbin index.bin ef prefetch(0|1) [seconds=10]
// Builds the index (M=16, ef_construction=200, all threads) and saves it to index.bin if that
// file does not exist yet, so repeated profiling runs share one graph. Then searches all
// queries repeatedly on one thread for about `seconds` and prints QPS.
#include <chrono>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "vecsearch/hnsw_index.hpp"

using namespace vecsearch;
using Clock = std::chrono::steady_clock;

namespace {

std::vector<float> read_fbin(const std::string& path, std::uint32_t& n, std::uint32_t& dim) {
  std::ifstream in(path, std::ios::binary);
  if (!in) throw std::runtime_error("cannot open " + path);
  in.read(reinterpret_cast<char*>(&n), 4);
  in.read(reinterpret_cast<char*>(&dim), 4);
  std::vector<float> v(std::size_t{n} * dim);
  in.read(reinterpret_cast<char*>(v.data()), v.size() * 4);
  if (!in) throw std::runtime_error("truncated " + path);
  return v;
}

}  // namespace

int main(int argc, char** argv) try {
  if (argc < 6) {
    std::fprintf(stderr, "usage: %s base.fbin query.fbin index.bin ef prefetch [seconds]\n",
                 argv[0]);
    return 2;
  }
  const std::string index_path = argv[3];
  const std::size_t ef = std::stoul(argv[4]);
  const bool prefetch = std::stoi(argv[5]) != 0;
  const double seconds = argc > 6 ? std::stod(argv[6]) : 10.0;

  std::uint32_t nq, qdim;
  const auto queries = read_fbin(argv[2], nq, qdim);

  std::unique_ptr<HnswIndex> index;
  if (std::filesystem::exists(index_path)) {
    index = HnswIndex::load(index_path);
  } else {
    std::uint32_t n, dim;
    const auto base = read_fbin(argv[1], n, dim);
    index = std::make_unique<HnswIndex>(dim, Metric::L2);
    const auto t0 = Clock::now();
    index->add(base.data(), n);
    std::fprintf(stderr, "built %u vectors in %.1f s\n", n,
                 std::chrono::duration<double>(Clock::now() - t0).count());
    index->save(index_path);
  }
  index->set_prefetch(prefetch);

  // Warm up (page in the index), then measure.
  index->search(queries.data(), nq, 10, ef, 1);
  std::size_t done = 0;
  const auto t0 = Clock::now();
  double elapsed = 0;
  while (elapsed < seconds) {
    index->search(queries.data(), nq, 10, ef, 1);
    done += nq;
    elapsed = std::chrono::duration<double>(Clock::now() - t0).count();
  }
  std::printf("ef %zu prefetch %d: %zu queries in %.2f s = %.0f QPS (1 thread)\n", ef,
              prefetch ? 1 : 0, done, elapsed, done / elapsed);
  return 0;
} catch (const std::exception& e) {
  std::fprintf(stderr, "error: %s\n", e.what());
  return 1;
}
