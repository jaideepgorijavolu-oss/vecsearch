// Experiment driver for adaptive (early-terminating) HNSW search; see docs/adaptive_search/.
//
//   adaptive_eval build    base.fbin out.hnsw M ef_construction seed [metric]  (1 thread)
//   adaptive_eval gt       base.fbin queries.fbin k out.gt [metric]           (exact, FlatIndex)
//   adaptive_eval trace    index.hnsw queries.fbin ef checkpoint out.trace
//   adaptive_eval run      index.hnsw queries.fbin configs.txt passes out_dir
//   adaptive_eval overhead model.txt inputs.f32 reps   (rows: features, then the query if used)
//   adaptive_eval batch    index.hnsw queries.fbin configs.txt reps threads  (throughput)
//
// trace: one unbudgeted search per query (all threads) recording stats, the features at the
// checkpoint and the top-k event log, from which recall at any budget is computed offline.
// run: one query per call on one thread, `passes` timed passes after one warmup pass, config
// order rotated each pass. configs.txt lines: name fixed|adaptive ef max_evals patience model mult
// (model "-" = none). Writes <name>.ids / .lat / .stats per config.
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <memory>
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

constexpr std::size_t kK = 10;

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

template <class T>
void put(std::ofstream& out, const std::vector<T>& v) {
  out.write(reinterpret_cast<const char*>(v.data()), v.size() * sizeof(T));
}
template <class T>
void put(std::ofstream& out, T x) {
  out.write(reinterpret_cast<const char*>(&x), sizeof(T));
}

double seconds_since(Clock::time_point t0) {
  return std::chrono::duration<double>(Clock::now() - t0).count();
}

int build(char** a, int nargs) {
  const Metric metric = parse_metric(nargs > 5 ? a[5] : "l2");
  const Matrix base = read_fbin(a[0]);
  HnswParams p;
  p.M = std::stoul(a[2]);
  p.ef_construction = std::stoul(a[3]);
  p.seed = std::stoull(a[4]);
  const auto t0 = Clock::now();
  HnswIndex index(base.dim, metric, p);
  index.add(base.data.data(), base.n, nullptr, 1);
  std::printf(
      "{\"build_seconds\": %.6f, \"threads\": 1, \"max_level\": %d, \"memory_bytes\": %zu}\n",
      seconds_since(t0), index.max_level(), index.memory_bytes());
  index.save(a[1]);
  return 0;
}

int gt(char** a, int nargs) {
  const Metric metric = parse_metric(nargs > 4 ? a[4] : "l2");
  const Matrix base = read_fbin(a[0]), q = read_fbin(a[1]);
  const std::size_t k = std::stoul(a[2]);
  const auto t0 = Clock::now();
  FlatIndex flat(base.dim, metric);
  flat.add(base.data.data(), base.n);
  const SearchResult r = flat.search(q.data.data(), q.n, k, 0);
  std::printf("{\"gt_seconds\": %.6f, \"threads\": %zu, \"queries\": %u}\n", seconds_since(t0),
              resolve_threads(0), q.n);
  std::ofstream out(a[3], std::ios::binary);
  put<std::uint64_t>(out, q.n);
  put<std::uint64_t>(out, k);
  put(out, r.ids);
  put(out, r.distances);
  return out ? 0 : 1;
}

int trace(char** a) {
  const auto index = HnswIndex::load(a[0]);
  const Matrix q = read_fbin(a[1]);
  AdaptiveParams p;
  p.ef = std::stoul(a[2]);
  p.checkpoint = std::stoul(a[3]);
  std::vector<AdaptiveStats> st(q.n);
  std::vector<std::vector<TopkEvent>> events;
  const auto t0 = Clock::now();
  index->search_adaptive(q.data.data(), q.n, kK, p, st.data(), &events, 0);
  std::fprintf(stderr, "trace: %.2f s\n", seconds_since(t0));

  std::vector<std::uint32_t> evals, expansions, ev_eval;
  std::vector<std::uint8_t> valid;
  std::vector<float> features, ev_dist;
  std::vector<std::int64_t> ev_id;
  std::vector<std::uint64_t> offsets{0};
  for (std::size_t i = 0; i < q.n; ++i) {
    evals.push_back(st[i].evals);
    expansions.push_back(st[i].expansions);
    valid.push_back(st[i].features_valid);
    features.insert(features.end(), st[i].features, st[i].features + kNumTerminationFeatures);
    for (const auto& e : events[i]) {
      ev_eval.push_back(e.eval);
      ev_id.push_back(e.id);
      ev_dist.push_back(e.dist);
    }
    offsets.push_back(ev_eval.size());
  }
  std::ofstream out(a[4], std::ios::binary);
  put<std::uint64_t>(out, q.n);
  put<std::uint64_t>(out, kNumTerminationFeatures);
  put<std::uint64_t>(out, ev_eval.size());
  put(out, evals);
  put(out, expansions);
  put(out, valid);
  put(out, features);
  put(out, offsets);
  put(out, ev_eval);
  put(out, ev_id);
  put(out, ev_dist);
  return out ? 0 : 1;
}

struct Config {
  std::string name, kind, model_path;
  AdaptiveParams p;
  std::unique_ptr<TerminationModel> model;
};

std::vector<Config> read_configs(const std::string& path) {
  std::vector<Config> configs;
  std::ifstream cf(path);
  for (std::string line; std::getline(cf, line);) {
    if (line.empty() || line[0] == '#') continue;
    std::istringstream ls(line);
    Config c;
    ls >> c.name >> c.kind >> c.p.ef >> c.p.max_evals >> c.p.patience >> c.model_path >>
        c.p.multiplier;
    if (!ls || (c.kind != "fixed" && c.kind != "adaptive"))
      throw std::runtime_error("bad config: " + line);
    if (c.model_path != "-") {
      c.model = std::make_unique<TerminationModel>(TerminationModel::load(c.model_path));
      c.p.model = c.model.get();
    }
    configs.push_back(std::move(c));
  }
  return configs;
}

int run(char** a) {
  const auto index = HnswIndex::load(a[0]);
  const Matrix q = read_fbin(a[1]);
  std::vector<Config> configs = read_configs(a[2]);
  const std::size_t passes = std::stoul(a[3]);
  const std::filesystem::path dir(a[4]);
  std::filesystem::create_directories(dir);

  const std::size_t nc = configs.size(), nq = q.n;
  std::vector<std::vector<std::int64_t>> lat(nc, std::vector<std::int64_t>(passes * nq));
  std::vector<std::vector<std::int64_t>> ids(nc);
  std::vector<std::vector<AdaptiveStats>> stats(nc, std::vector<AdaptiveStats>(nq));
  for (std::size_t pass = 0; pass <= passes; ++pass) {  // pass 0 = warmup
    for (std::size_t j = 0; j < nc; ++j) {
      const std::size_t ci = (j + pass) % nc;
      Config& c = configs[ci];
      std::vector<std::int64_t> got(nq * kK);
      for (std::size_t i = 0; i < nq; ++i) {
        const float* x = q.data.data() + i * q.dim;
        const auto t0 = Clock::now();
        const SearchResult r =
            c.kind == "fixed" ? index->search(x, 1, kK, c.p.ef, 1)
                              : index->search_adaptive(x, 1, kK, c.p, &stats[ci][i], nullptr, 1);
        const auto ns =
            std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now() - t0).count();
        std::copy(r.ids.begin(), r.ids.end(), got.begin() + i * kK);
        if (pass > 0) lat[ci][(pass - 1) * nq + i] = ns;
      }
      if (pass == 0)
        ids[ci] = got;
      else if (got != ids[ci])
        throw std::runtime_error("non-deterministic results: " + c.name);
    }
    std::fprintf(stderr, "pass %zu/%zu done\n", pass, passes);
  }
  for (std::size_t ci = 0; ci < nc; ++ci) {
    const auto base = dir / configs[ci].name;
    std::ofstream o1(base.string() + ".ids", std::ios::binary),
        o2(base.string() + ".lat", std::ios::binary),
        o3(base.string() + ".stats", std::ios::binary);
    put(o1, ids[ci]);
    put(o2, lat[ci]);
    for (const auto& s : stats[ci]) {  // evals, expansions, budget, stop, model_used
      put<std::uint32_t>(o3, s.evals);
      put<std::uint32_t>(o3, s.expansions);
      put<std::uint32_t>(o3, s.budget);
      put<std::uint8_t>(o3, static_cast<std::uint8_t>(s.stop));
      put<std::uint8_t>(o3, s.model_used);
    }
  }
  return 0;
}

// Cost of one model prediction (inference only; feature extraction is measured end to end).
int overhead(char** a) {
  const auto model = TerminationModel::load(a[0]);
  std::ifstream in(a[1], std::ios::binary | std::ios::ate);
  std::vector<float> feats(static_cast<std::size_t>(in.tellg()) / sizeof(float));
  in.seekg(0);
  in.read(reinterpret_cast<char*>(feats.data()), feats.size() * sizeof(float));
  const std::size_t width = kNumTerminationFeatures + model.query_dim();
  const std::size_t n = feats.size() / width, reps = std::stoul(a[2]);
  double sink = 0;
  const auto t0 = Clock::now();
  for (std::size_t r = 0; r < reps; ++r)
    for (std::size_t i = 0; i < n; ++i)
      sink +=
          model.predict_log_evals(&feats[i * width], &feats[i * width + kNumTerminationFeatures]);
  const double ns = seconds_since(t0) * 1e9 / double(n * reps);
  std::printf("{\"predict_ns\": %.3f, \"n\": %zu, \"reps\": %zu, \"checksum\": %.6f}\n", ns, n,
              reps, sink);
  return 0;
}

// Multithreaded throughput: all queries in one call on `threads` threads, best of `reps` runs.
// Prints one JSON line per config. Results must equal the single-thread ones (checked).
int batch(char** a) {
  const auto index = HnswIndex::load(a[0]);
  const Matrix q = read_fbin(a[1]);
  const std::vector<Config> configs = read_configs(a[2]);
  const std::size_t reps = std::stoul(a[3]), threads = std::stoul(a[4]);
  for (const Config& c : configs) {
    auto search = [&](std::size_t t) {
      return c.kind == "fixed"
                 ? index->search(q.data.data(), q.n, kK, c.p.ef, t)
                 : index->search_adaptive(q.data.data(), q.n, kK, c.p, nullptr, nullptr, t);
    };
    const SearchResult serial = search(1);
    double best = 1e300;
    for (std::size_t r = 0; r < reps; ++r) {
      const auto t0 = Clock::now();
      const SearchResult got = search(threads);
      best = std::min(best, seconds_since(t0));
      if (got.ids != serial.ids) throw std::runtime_error("parallel != serial: " + c.name);
    }
    std::printf(
        "{\"name\": \"%s\", \"threads\": %zu, \"reps\": %zu, \"best_seconds\": %.6f, "
        "\"qps\": %.1f}\n",
        c.name.c_str(), threads, reps, best, q.n / best);
  }
  return 0;
}

}  // namespace

int main(int argc, char** argv) try {
  const std::string cmd = argc > 1 ? argv[1] : "";
  const int nargs = argc - 2;
  if (cmd == "build" && (nargs == 5 || nargs == 6)) return build(argv + 2, nargs);
  if (cmd == "gt" && (nargs == 4 || nargs == 5)) return gt(argv + 2, nargs);
  if (cmd == "trace" && nargs == 5) return trace(argv + 2);
  if (cmd == "run" && nargs == 5) return run(argv + 2);
  if (cmd == "overhead" && nargs == 3) return overhead(argv + 2);
  if (cmd == "batch" && nargs == 5) return batch(argv + 2);
  std::fprintf(stderr, "usage: see the comment at the top of adaptive_eval.cpp\n");
  return 2;
} catch (const std::exception& e) {
  std::fprintf(stderr, "error: %s\n", e.what());
  return 1;
}
