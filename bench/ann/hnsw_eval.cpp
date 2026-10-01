// Build an HNSW index over a .fbin dataset, compute exact ground truth with FlatIndex, and
// report recall@10 and QPS for a sweep of ef_search values. Also checks that save -> load gives
// identical results. Used for the Phase 3 acceptance test (SIFT 100k subset).
//
// Usage: hnsw_eval base.fbin query.fbin [metric=l2] [M=16] [ef_construction=200] [threads=0]
//                  [ef list, comma separated=10,20,40,80,160]
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "vecsearch/flat_index.hpp"
#include "vecsearch/hnsw_index.hpp"
#include "vecsearch/parallel.hpp"

using namespace vecsearch;
using Clock = std::chrono::steady_clock;

namespace {

struct Matrix {
  std::uint32_t n = 0, dim = 0;
  std::vector<float> data;
};

Matrix read_fbin(const std::string& path) {
  std::ifstream in(path, std::ios::binary);
  if (!in) throw std::runtime_error("cannot open " + path);
  Matrix m;
  in.read(reinterpret_cast<char*>(&m.n), 4);
  in.read(reinterpret_cast<char*>(&m.dim), 4);
  m.data.resize(std::size_t{m.n} * m.dim);
  in.read(reinterpret_cast<char*>(m.data.data()), m.data.size() * 4);
  if (!in) throw std::runtime_error("truncated " + path);
  return m;
}

double seconds_since(Clock::time_point t0) {
  return std::chrono::duration<double>(Clock::now() - t0).count();
}

double recall(const SearchResult& got, const SearchResult& truth, std::size_t k) {
  double hits = 0;
  for (std::size_t q = 0; q < truth.num_queries; ++q) {
    std::set<std::int64_t> t(truth.row_ids(q), truth.row_ids(q) + k);
    for (std::size_t j = 0; j < k; ++j) hits += t.count(got.row_ids(q)[j]);
  }
  return hits / double(truth.num_queries * k);
}

}  // namespace

int main(int argc, char** argv) try {
  if (argc < 3) {
    std::fprintf(stderr, "usage: %s base.fbin query.fbin [metric M efc threads efs]\n", argv[0]);
    return 2;
  }
  const Matrix base = read_fbin(argv[1]), queries = read_fbin(argv[2]);
  const Metric metric = parse_metric(argc > 3 ? argv[3] : "l2");
  HnswParams params;
  params.M = argc > 4 ? std::stoul(argv[4]) : 16;
  params.ef_construction = argc > 5 ? std::stoul(argv[5]) : 200;
  const std::size_t threads = resolve_threads(argc > 6 ? std::stoul(argv[6]) : 0);
  std::vector<std::size_t> efs;
  std::stringstream ss(argc > 7 ? argv[7] : "10,20,40,80,160");
  for (std::string tok; std::getline(ss, tok, ',');) efs.push_back(std::stoul(tok));
  const std::size_t k = 10;

  std::printf(
      "base %u x %u, queries %u, metric %s, M %zu, ef_construction %zu, threads %zu, "
      "kernels %s\n",
      base.n, base.dim, queries.n, metric_name(metric), params.M, params.ef_construction, threads,
      active_kernels().name);

  auto t0 = Clock::now();
  FlatIndex flat(base.dim, metric);
  flat.add(base.data.data(), base.n);
  const SearchResult truth = flat.search(queries.data.data(), queries.n, k, threads);
  std::printf("ground truth (FlatIndex): %.2f s\n", seconds_since(t0));

  t0 = Clock::now();
  HnswIndex index(base.dim, metric, params);
  index.add(base.data.data(), base.n, nullptr, threads);
  std::printf("build: %.2f s, max level %d, memory %.1f MiB\n", seconds_since(t0),
              index.max_level(), index.memory_bytes() / 1048576.0);

  std::printf("%8s %10s %12s %12s\n", "ef", "recall@10", "QPS(1 thr)", "QPS(all)");
  for (std::size_t ef : efs) {
    t0 = Clock::now();
    const SearchResult r1 = index.search(queries.data.data(), queries.n, k, ef, 1);
    const double qps1 = queries.n / seconds_since(t0);
    t0 = Clock::now();
    const SearchResult rn = index.search(queries.data.data(), queries.n, k, ef, threads);
    const double qpsn = queries.n / seconds_since(t0);
    if (r1.ids != rn.ids) throw std::runtime_error("parallel search differs from serial");
    std::printf("%8zu %10.4f %12.0f %12.0f\n", ef, recall(r1, truth, k), qps1, qpsn);
  }

  const std::string path = "/tmp/hnsw_eval_index.bin";
  index.save(path);
  const auto loaded = HnswIndex::load(path);
  std::remove(path.c_str());
  const auto a = index.search(queries.data.data(), queries.n, k, efs.back(), threads);
  const auto b = loaded->search(queries.data.data(), queries.n, k, efs.back(), threads);
  const bool same = a.ids == b.ids && a.distances == b.distances;
  std::printf("save -> load -> search: %s\n", same ? "identical" : "DIFFERENT");
  return same ? 0 : 1;
} catch (const std::exception& e) {
  std::fprintf(stderr, "error: %s\n", e.what());
  return 1;
}
