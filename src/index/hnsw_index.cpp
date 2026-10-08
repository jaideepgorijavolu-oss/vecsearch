#include "vecsearch/hnsw_index.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <functional>
#include <limits>
#include <stdexcept>

#include "vecsearch/io.hpp"
#include "vecsearch/parallel.hpp"

namespace vecsearch {

namespace {

constexpr char kMagic[9] = "VSHNSW01";
constexpr std::uint32_t kVersion = 1;
constexpr int kMaxLevel = 31;

struct AcceptAll {
  bool operator()(std::uint32_t) const { return true; }
};

}  // namespace

// A free list of Scratch objects, so repeated searches reuse their buffers (including the
// visited list, which is as large as the index) instead of allocating per query.
class HnswIndex::ScratchPool {
 public:
  std::unique_ptr<Scratch> acquire() {
    std::lock_guard lock(mu_);
    if (free_.empty()) return std::make_unique<Scratch>();
    auto s = std::move(free_.back());
    free_.pop_back();
    return s;
  }
  void release(std::unique_ptr<Scratch> s) {
    std::lock_guard lock(mu_);
    free_.push_back(std::move(s));
  }

  // Borrows one Scratch per worker for the duration of a parallel call.
  class Lease {
   public:
    Lease(ScratchPool& pool, std::size_t n) : pool_(pool) {
      for (std::size_t i = 0; i < n; ++i) items_.push_back(pool.acquire());
    }
    ~Lease() {
      for (auto& s : items_) pool_.release(std::move(s));
    }
    Scratch& operator[](std::size_t i) { return *items_[i]; }

   private:
    ScratchPool& pool_;
    std::vector<std::unique_ptr<Scratch>> items_;
  };

 private:
  std::mutex mu_;
  std::vector<std::unique_ptr<Scratch>> free_;
};

HnswIndex::HnswIndex(std::size_t dim, Metric metric, HnswParams params)
    : dim_(dim),
      stride_(padded_dim(dim)),
      metric_(metric),
      dist_(make_distance(metric)),
      params_(params),
      M0_(2 * params.M),
      level_mult_(params.M > 1 ? 1.0 / std::log(static_cast<double>(params.M)) : 0.0),
      ef_search_(params.ef_search),
      rng_(params.seed),
      scratch_pool_(std::make_shared<ScratchPool>()) {
  if (dim == 0) throw std::invalid_argument("dim must be positive");
  if (params.M < 2) throw std::invalid_argument("M must be at least 2");
  if (params.ef_construction == 0) throw std::invalid_argument("ef_construction must be positive");
}

void HnswIndex::reserve(std::size_t capacity) {
  if (capacity <= capacity_) return;
  capacity = std::max(capacity, capacity_ * 2);  // geometric growth: amortized O(1) per add
  data_.resize(capacity * stride_, 0.0f);
  level0_.resize(capacity * (1 + M0_), 0);
  upper_.resize(capacity);
  levels_.resize(capacity, 0);
  deleted_.resize(capacity, 0);
  labels_.resize(capacity, kNoId);
  // All locks are free between add() calls, so they can simply be recreated.
  node_locks_ = std::make_unique<SpinLock[]>(capacity);
  capacity_ = capacity;
}

int HnswIndex::random_level() {
  // level = floor(-ln(U) * mL), U uniform in (0, 1]. P(level >= l) = M^-l, so each layer has
  // about 1/M of the nodes of the layer below.
  std::uniform_real_distribution<double> u(0.0, 1.0);
  const double r = -std::log(1.0 - u(rng_)) * level_mult_;
  return std::min(static_cast<int>(r), kMaxLevel);
}

void HnswIndex::add(const float* vectors, std::size_t n, const std::int64_t* labels,
                    std::size_t num_threads) {
  if (n == 0) return;
  if (count_ + n > std::numeric_limits<std::uint32_t>::max())
    throw std::length_error("HNSW index is limited to 2^32 - 1 vectors");
  if (labels) {
    for (std::size_t i = 0; i < n; ++i) {
      if (labels[i] == kNoId) throw std::invalid_argument("id -1 is reserved for 'no result'");
    }
  }
  const std::size_t first = count_;
  reserve(count_ + n);

  // Sequential part: copy vectors, assign labels and levels, allocate upper-layer blocks.
  // Doing this before any thread starts means every node's vector, level and storage are
  // immutable while the parallel phase runs; only neighbor lists change, under node locks.
  for (std::size_t i = 0; i < n; ++i) {
    const auto id = static_cast<std::uint32_t>(first + i);
    float* row = data_.data() + std::size_t{id} * stride_;
    std::memcpy(row, vectors + i * dim_, dim_ * sizeof(float));
    if (metric_ == Metric::Cosine) normalize(row, dim_);

    const std::int64_t label = labels ? labels[i] : static_cast<std::int64_t>(id);
    if (auto it = label_to_id_.find(label); it != label_to_id_.end()) {
      deleted_[it->second] = 1;  // upsert: the old vector becomes a tombstone
      ++num_deleted_;
      it->second = id;
    } else {
      label_to_id_.emplace(label, id);
    }
    labels_[id] = label;

    const int level = random_level();
    levels_[id] = static_cast<std::uint8_t>(level);
    upper_[id].assign(std::size_t(level) * (1 + params_.M), 0);
  }
  count_ += n;

  std::size_t start = 0;
  ScratchPool::Lease scratch(*scratch_pool_, std::min(resolve_threads(num_threads), n));
  // While the graph is small, insert sequentially. With T threads, each insert cannot see the
  // up to T-1 nodes being inserted at the same moment; in a graph of a few hundred nodes that
  // is a large fraction of it, and those early nodes end up poorly linked (measured by
  // bench/ann/parallel_build_quality, see DESIGN.md). Once the graph has
  // params_.sequential_prefix nodes, concurrent inserts barely affect each other. The first
  // node is always inserted alone: it becomes the entry point.
  while (start < n && (first + start == 0 || first + start < params_.sequential_prefix)) {
    insert(static_cast<std::uint32_t>(first + start), scratch[0]);
    ++start;
  }
  parallel_for(n - start, num_threads, [&](std::size_t i, std::size_t worker) {
    insert(static_cast<std::uint32_t>(first + start + i), scratch[worker]);
  });
}

void HnswIndex::insert(std::uint32_t id, Scratch& s) {
  const int level = levels_[id];
  std::unique_lock global(global_mu_);
  const int top = max_level_;
  const std::uint32_t entry = entry_point_;
  if (top < 0) {
    entry_point_ = id;
    max_level_ = level;
    return;
  }
  // A node that raises the maximum level keeps the global lock for its whole insertion, so no
  // other insert can start from an entry point that is about to change. This happens for about
  // 1 in M^top inserts.
  if (level <= top) global.unlock();

  const float* q = vec(id);
  std::uint32_t ep = greedy_descent<true>(q, entry, top, level, s);
  for (int l = std::min(level, top); l >= 0; --l) {
    search_layer<true>(q, ep, params_.ef_construction, l, s, AcceptAll{});
    std::sort_heap(s.results.begin(), s.results.end());  // closest first
    ep = s.results.front().id;
    connect(id, s.results, l, s);
  }
  if (level > top) {
    entry_point_ = id;
    max_level_ = level;
  }
}

template <bool kLocked>
std::uint32_t HnswIndex::greedy_descent(const float* q, std::uint32_t ep, int top, int bottom,
                                        Scratch& s) const {
  std::uint32_t cur = ep;
  float cur_dist = dist(q, cur);
  for (int l = top; l > bottom; --l) {
    for (bool changed = true; changed;) {
      changed = false;
      const std::uint32_t* nb;
      std::size_t n;
      if constexpr (kLocked) {
        std::lock_guard lock(node_locks_[cur]);
        const std::uint32_t* block = links(cur, l);
        s.neighbor_copy.assign(block + 1, block + 1 + block[0]);
        nb = s.neighbor_copy.data();
        n = s.neighbor_copy.size();
      } else {
        const std::uint32_t* block = links(cur, l);
        nb = block + 1;
        n = block[0];
      }
      for (std::size_t j = 0; j < n; ++j) {
        const float d = dist(q, nb[j]);
        if (d < cur_dist) {
          cur_dist = d;
          cur = nb[j];
          changed = true;
        }
      }
    }
  }
  return cur;
}

void HnswIndex::prefetch_address(const void* p) {
#if defined(__GNUC__) || defined(__clang__)
  __builtin_prefetch(p);
#else
  (void)p;
#endif
}

void HnswIndex::prefetch_vector(std::uint32_t id) const {
#if defined(__GNUC__) || defined(__clang__)
  const char* p = reinterpret_cast<const char*>(vec(id));
  for (std::size_t off = 0; off < dim_ * sizeof(float); off += kCacheLine)
    __builtin_prefetch(p + off);
#else
  (void)id;
#endif
}

template <bool kLocked, class Accept>
void HnswIndex::search_layer(const float* q, std::uint32_t ep, std::size_t ef, int level,
                             Scratch& s, const Accept& accept) const {
  auto& candidates = s.candidates;  // min-heap: std::greater puts the closest on top
  auto& results = s.results;        // max-heap: the furthest kept result on top
  candidates.clear();
  results.clear();
  s.visited.reset(count_);

  const float d0 = dist(q, ep);
  s.visited.test_and_set(ep);
  candidates.push_back({d0, ep});
  if (accept(ep)) results.push_back({d0, ep});
  // Distance of the furthest kept result: a candidate further than this cannot improve them.
  float bound = results.empty() ? std::numeric_limits<float>::infinity() : d0;

  while (!candidates.empty()) {
    const Cand c = candidates.front();
    // Every remaining candidate is further than our worst result: done.
    if (c.dist > bound && results.size() >= ef) break;
    std::pop_heap(candidates.begin(), candidates.end(), std::greater<>{});
    candidates.pop_back();
    // The new heap top is most likely the next node we expand: start loading its neighbor block.
    if (prefetch_ && !candidates.empty()) prefetch_address(links(candidates.front().id, level));

    const std::uint32_t* nb;
    std::size_t n;
    if constexpr (kLocked) {
      // Copy the neighbor list under the node lock, then compute distances without holding it.
      std::lock_guard lock(node_locks_[c.id]);
      const std::uint32_t* block = links(c.id, level);
      s.neighbor_copy.assign(block + 1, block + 1 + block[0]);
      nb = s.neighbor_copy.data();
      n = s.neighbor_copy.size();
    } else {
      const std::uint32_t* block = links(c.id, level);
      nb = block + 1;
      n = block[0];
    }

    // Each neighbor costs two likely cache misses: its visited mark (a random spot in an array
    // as large as the index) and its vector (random, 512 bytes for SIFT). Done naively, they
    // are taken one neighbor at a time, each waiting ~100 ns for DRAM. With prefetching we issue
    // all the visited-mark loads, then all the vector loads for unvisited neighbors, and only
    // then compute distances, so the misses overlap (memory-level parallelism).
    // See RESULTS.md, Phase 4, for the profile that motivated this.
    if (prefetch_) {
      for (std::size_t j = 0; j < n; ++j) prefetch_address(s.visited.address(nb[j]));
    }
    auto& todo = s.unvisited;
    todo.clear();
    for (std::size_t j = 0; j < n; ++j) {
      if (!s.visited.test_and_set(nb[j])) todo.push_back(nb[j]);
    }
    if (prefetch_) {
      for (const std::uint32_t id : todo) prefetch_vector(id);
    }

    for (const std::uint32_t id : todo) {
      const float d = dist(q, id);
      if (results.size() < ef || d < bound) {
        candidates.push_back({d, id});
        std::push_heap(candidates.begin(), candidates.end(), std::greater<>{});
        // Rejected nodes (deleted, filtered out) are still expanded as candidates so the search
        // can route through them; they just never enter the results.
        if (accept(id)) {
          results.push_back({d, id});
          std::push_heap(results.begin(), results.end());
          if (results.size() > ef) {
            std::pop_heap(results.begin(), results.end());
            results.pop_back();
          }
          bound = results.front().dist;
        }
      }
    }
  }
}

template <class Accept>
void HnswIndex::search_layer_adaptive(const float* q, std::uint32_t ep, std::size_t k,
                                      const AdaptiveParams& p, const TerminationModel* model,
                                      Scratch& s, const Accept& accept, AdaptiveStats& st,
                                      std::vector<TopkEvent>* events) const {
  constexpr std::size_t kNever = std::numeric_limits<std::size_t>::max();
  constexpr float kInf = std::numeric_limits<float>::infinity();
  const std::size_t ef = p.ef;
  auto& candidates = s.candidates;
  auto& results = s.results;
  auto& topk = s.topk;  // max-heap of the best k accepted nodes: exactly what would be returned
  candidates.clear();
  results.clear();
  topk.clear();
  s.visited.reset(count_);
  st = AdaptiveStats{};

  // All termination rules are checked against the layer-0 evaluation count. To keep the hot
  // loop at one comparison per evaluation, `next_event` is the next count at which anything can
  // happen (the checkpoint or the budget).
  std::size_t evals = 0;
  std::size_t budget = p.max_evals != 0 ? p.max_evals : kNever;
  std::size_t checkpoint =
      model ? model->checkpoint() : (p.checkpoint != 0 ? p.checkpoint : kNever);
  std::size_t next_event = std::min(budget, checkpoint);
  std::size_t last_change = 0, changes = 0, stale_expansions = 0;
  bool changed = false;  // the top-k changed during the current expansion
  float best = kInf;
  float d_entry = 0;

  auto offer_topk = [&](float d, std::uint32_t id) {
    if (topk.size() == k && !(d < topk.front().dist)) return;
    topk.push_back({d, id});
    std::push_heap(topk.begin(), topk.end());
    if (topk.size() > k) {
      std::pop_heap(topk.begin(), topk.end());
      topk.pop_back();
    }
    best = std::min(best, d);
    last_change = evals;
    ++changes;
    changed = true;
    if (events) events->push_back({static_cast<std::uint32_t>(evals), labels_[id], d});
  };

  // Runs when evals reaches next_event. Returns true if the search must stop now.
  auto on_event = [&]() {
    if (evals >= checkpoint) {
      checkpoint = kNever;
      if (topk.size() == k) {
        const float dk = topk.front().dist;
        const float cand = candidates.empty() ? dk : candidates.front().dist;
        float* f = st.features;
        f[kLogEntryDist] = std::log1p(d_entry);
        f[kLogBestDist] = std::log1p(best);
        f[kLogKthDist] = std::log1p(dk);
        f[kBestOverEntry] = best / d_entry;
        f[kKthOverEntry] = dk / d_entry;
        f[kKthOverBest] = dk / best;
        f[kCandidateOverKth] = cand / dk;
        f[kStaleFraction] = float(evals - last_change) / float(evals);
        f[kTopkChanges] = float(changes);
        st.features_valid =
            std::all_of(f, f + kNumTerminationFeatures, [](float x) { return std::isfinite(x); });
      }
      // Invalid features (e.g. a zero distance): ignore the model; explicit limits still apply.
      if (st.features_valid && model) {
        const double predicted = p.multiplier * std::exp(model->predict_log_evals(st.features, q));
        if (predicted >= 0) {  // false for NaN
          const double capped = std::min(predicted, 4e9);
          budget = std::min(budget, std::max(evals, static_cast<std::size_t>(std::ceil(capped))));
          st.model_used = true;
        }
      }
    }
    if (evals >= budget) {
      st.stop = StopReason::Budget;
      return true;
    }
    next_event = std::min(budget, checkpoint);
    return false;
  };

  const float d0 = dist(q, ep);
  ++evals;
  d_entry = d0;
  s.visited.test_and_set(ep);
  candidates.push_back({d0, ep});
  if (accept(ep)) {
    results.push_back({d0, ep});
    offer_topk(d0, ep);
  }
  float bound = results.empty() ? kInf : d0;
  bool stop = evals >= next_event && on_event();

  // From here on this is search_layer (same order of expansions and distance evaluations), with
  // the termination checks added.
  while (!stop && !candidates.empty()) {
    const Cand c = candidates.front();
    if (c.dist > bound && results.size() >= ef) break;
    std::pop_heap(candidates.begin(), candidates.end(), std::greater<>{});
    candidates.pop_back();
    if (prefetch_ && !candidates.empty()) prefetch_address(links(candidates.front().id, 0));
    ++st.expansions;
    changed = false;

    const std::uint32_t* block = links(c.id, 0);
    const std::uint32_t* nb = block + 1;
    const std::size_t n = block[0];
    if (prefetch_) {
      for (std::size_t j = 0; j < n; ++j) prefetch_address(s.visited.address(nb[j]));
    }
    auto& todo = s.unvisited;
    todo.clear();
    for (std::size_t j = 0; j < n; ++j) {
      if (!s.visited.test_and_set(nb[j])) todo.push_back(nb[j]);
    }
    if (prefetch_) {
      for (const std::uint32_t id : todo) prefetch_vector(id);
    }

    for (const std::uint32_t id : todo) {
      const float d = dist(q, id);
      ++evals;
      if (results.size() < ef || d < bound) {
        candidates.push_back({d, id});
        std::push_heap(candidates.begin(), candidates.end(), std::greater<>{});
        if (accept(id)) {
          results.push_back({d, id});
          std::push_heap(results.begin(), results.end());
          if (results.size() > ef) {
            std::pop_heap(results.begin(), results.end());
            results.pop_back();
          }
          bound = results.front().dist;
          offer_topk(d, id);
        }
      }
      if (evals >= next_event && on_event()) {
        stop = true;
        break;
      }
    }
    if (stop) break;
    if (p.patience != 0) {
      stale_expansions = changed ? 0 : stale_expansions + 1;
      if (stale_expansions >= p.patience) {
        st.stop = StopReason::Patience;
        break;
      }
    }
  }
  st.evals = static_cast<std::uint32_t>(evals);
  st.budget =
      budget == kNever ? 0 : static_cast<std::uint32_t>(std::min<std::size_t>(budget, UINT32_MAX));
}

void HnswIndex::select_neighbors(std::vector<Cand>& candidates, std::size_t m) const {
  if (candidates.size() <= m) return;
  if (!params_.use_heuristic) {
    candidates.resize(m);  // simple: the m closest
    return;
  }
  // Algorithm 4 (without the optional extendCandidates / keepPrunedConnections steps).
  // Walk candidates from closest to furthest; keep c only if it is closer to the base node than
  // to every neighbor already kept. Otherwise some kept neighbor already "covers" c's direction,
  // and the graph is better off spending the link on a different direction. This keeps links to
  // other clusters that a plain closest-M rule would drop, which is what keeps the graph
  // navigable on clustered data.
  std::size_t kept = 0;
  for (std::size_t i = 0; i < candidates.size() && kept < m; ++i) {
    const Cand c = candidates[i];
    bool good = true;
    for (std::size_t j = 0; j < kept; ++j) {
      if (dist_(vec(candidates[j].id), vec(c.id), dim_) < c.dist) {
        good = false;
        break;
      }
    }
    if (good) candidates[kept++] = c;
  }
  candidates.resize(kept);
}

void HnswIndex::connect(std::uint32_t id, std::vector<Cand>& candidates, int level, Scratch& s) {
  select_neighbors(candidates, params_.M);
  {
    std::lock_guard lock(node_locks_[id]);
    std::uint32_t* block = links(id, level);
    block[0] = static_cast<std::uint32_t>(candidates.size());
    for (std::size_t i = 0; i < candidates.size(); ++i) block[1 + i] = candidates[i].id;
  }

  // Add the reverse edge to each chosen neighbor. A thread holds at most one node lock at a
  // time, so there is no lock ordering to get wrong and no deadlock.
  const std::size_t mmax = max_links(level);
  auto& pool = s.candidates;  // free to reuse: search_layer is done with it
  for (const Cand& nbr : candidates) {
    std::lock_guard lock(node_locks_[nbr.id]);
    std::uint32_t* block = links(nbr.id, level);
    const std::size_t n = block[0];
    if (n < mmax) {
      block[1 + n] = id;
      block[0] = static_cast<std::uint32_t>(n + 1);
      continue;
    }
    // Full: re-select among the old neighbors plus the new node, from nbr's point of view.
    const float* v = vec(nbr.id);
    pool.clear();
    pool.push_back({nbr.dist, id});
    for (std::size_t j = 0; j < n; ++j)
      pool.push_back({dist_(v, vec(block[1 + j]), dim_), block[1 + j]});
    std::sort(pool.begin(), pool.end());
    select_neighbors(pool, mmax);
    block[0] = static_cast<std::uint32_t>(pool.size());
    for (std::size_t j = 0; j < pool.size(); ++j) block[1 + j] = pool[j].id;
  }
}

SearchResult HnswIndex::search(const float* queries, std::size_t nq, std::size_t k, std::size_t ef,
                               std::size_t num_threads, const std::int64_t* allowed_labels,
                               std::size_t num_allowed) const {
  SearchResult result(nq, k);
  if (k == 0 || nq == 0 || max_level_ < 0) return result;
  ef = std::max(ef == 0 ? ef_search() : ef, k);

  std::vector<std::uint8_t> allowed;  // by internal id; deleted nodes are never in label_to_id_
  if (allowed_labels) {
    allowed.assign(count_, 0);
    for (std::size_t i = 0; i < num_allowed; ++i) {
      if (auto it = label_to_id_.find(allowed_labels[i]); it != label_to_id_.end())
        allowed[it->second] = 1;
    }
  }

  const std::size_t threads = std::min(resolve_threads(num_threads), nq);
  ScratchPool::Lease scratch(*scratch_pool_, threads);
  parallel_for(nq, threads, [&](std::size_t qi, std::size_t worker) {
    Scratch& s = scratch[worker];
    const float* q = queries + qi * dim_;
    if (metric_ == Metric::Cosine) {
      s.query.assign(q, q + dim_);
      normalize(s.query.data(), dim_);
      q = s.query.data();
    }
    const std::uint32_t ep = greedy_descent<false>(q, entry_point_, max_level_, 0, s);
    if (allowed_labels) {
      search_layer<false>(q, ep, ef, 0, s, [&](std::uint32_t id) { return allowed[id] != 0; });
    } else if (num_deleted_ > 0) {
      search_layer<false>(q, ep, ef, 0, s, [&](std::uint32_t id) { return deleted_[id] == 0; });
    } else {
      search_layer<false>(q, ep, ef, 0, s, AcceptAll{});
    }
    std::sort_heap(s.results.begin(), s.results.end());
    const std::size_t found = std::min(k, s.results.size());
    for (std::size_t j = 0; j < found; ++j) {
      result.ids[qi * k + j] = labels_[s.results[j].id];
      result.distances[qi * k + j] = s.results[j].dist;
    }
  });
  return result;
}

SearchResult HnswIndex::search_adaptive(const float* queries, std::size_t nq, std::size_t k,
                                        const AdaptiveParams& params, AdaptiveStats* stats,
                                        std::vector<std::vector<TopkEvent>>* events,
                                        std::size_t num_threads) const {
  if (params.model && params.model->query_dim() != 0 && params.model->query_dim() != dim_)
    throw std::invalid_argument("termination model query_dim does not match the index");
  SearchResult result(nq, k);
  if (events) events->assign(nq, {});
  if (stats) std::fill(stats, stats + nq, AdaptiveStats{});
  if (k == 0 || nq == 0 || max_level_ < 0) return result;
  AdaptiveParams p = params;
  p.ef = std::max(p.ef == 0 ? ef_search() : p.ef, k);
  // The model was trained on static, unfiltered indexes: with deleted nodes, ignore it (explicit
  // max_evals / patience still apply).
  const TerminationModel* model = num_deleted_ == 0 ? p.model : nullptr;

  const std::size_t threads = std::min(resolve_threads(num_threads), nq);
  ScratchPool::Lease scratch(*scratch_pool_, threads);
  parallel_for(nq, threads, [&](std::size_t qi, std::size_t worker) {
    Scratch& s = scratch[worker];
    const float* q = queries + qi * dim_;
    if (metric_ == Metric::Cosine) {
      s.query.assign(q, q + dim_);
      normalize(s.query.data(), dim_);
      q = s.query.data();
    }
    const std::uint32_t ep = greedy_descent<false>(q, entry_point_, max_level_, 0, s);
    AdaptiveStats st;
    std::vector<TopkEvent>* ev = events ? &(*events)[qi] : nullptr;
    if (num_deleted_ > 0) {
      search_layer_adaptive(
          q, ep, k, p, model, s, [&](std::uint32_t id) { return deleted_[id] == 0; }, st, ev);
    } else {
      search_layer_adaptive(q, ep, k, p, model, s, AcceptAll{}, st, ev);
    }
    std::sort_heap(s.results.begin(), s.results.end());
    const std::size_t found = std::min(k, s.results.size());
    for (std::size_t j = 0; j < found; ++j) {
      result.ids[qi * k + j] = labels_[s.results[j].id];
      result.distances[qi * k + j] = s.results[j].dist;
    }
    if (stats) stats[qi] = st;
  });
  return result;
}

bool HnswIndex::remove(std::int64_t label) {
  const auto it = label_to_id_.find(label);
  if (it == label_to_id_.end()) return false;
  deleted_[it->second] = 1;
  ++num_deleted_;
  label_to_id_.erase(it);
  return true;
}

std::vector<std::uint32_t> HnswIndex::neighbors(std::uint32_t id, int level) const {
  const std::uint32_t* block = links(id, level);
  return {block + 1, block + 1 + block[0]};
}

std::size_t HnswIndex::memory_bytes() const {
  std::size_t bytes = data_.capacity() * sizeof(float) + level0_.capacity() * sizeof(std::uint32_t);
  bytes += upper_.capacity() * sizeof(upper_[0]);
  for (std::size_t i = 0; i < count_; ++i) bytes += upper_[i].capacity() * sizeof(std::uint32_t);
  bytes += levels_.capacity() + deleted_.capacity() + labels_.capacity() * sizeof(std::int64_t);
  bytes += capacity_ * sizeof(SpinLock);
  // unordered_map: bucket array + one heap node (key, value, next pointer, hash) per entry.
  bytes += label_to_id_.bucket_count() * sizeof(void*) + label_to_id_.size() * 32;
  return bytes;
}

void HnswIndex::save(const std::string& path) const {
  auto out = io::open_out(path);
  io::write_header(out, kMagic, kVersion);
  io::write_pod<std::uint64_t>(out, dim_);
  io::write_pod<std::uint8_t>(out, static_cast<std::uint8_t>(metric_));
  io::write_pod<std::uint64_t>(out, params_.M);
  io::write_pod<std::uint64_t>(out, params_.ef_construction);
  io::write_pod<std::uint64_t>(out, ef_search());
  io::write_pod<std::uint64_t>(out, params_.seed);
  io::write_pod<std::uint8_t>(out, params_.use_heuristic ? 1 : 0);
  io::write_pod<std::uint64_t>(out, count_);
  io::write_pod<std::int32_t>(out, max_level_);
  io::write_pod<std::uint32_t>(out, entry_point_);
  io::write_array(out, labels_.data(), count_);
  io::write_array(out, levels_.data(), count_);
  io::write_array(out, deleted_.data(), count_);
  io::write_array(out, level0_.data(), count_ * (1 + M0_));
  for (std::size_t i = 0; i < count_; ++i) io::write_array(out, upper_[i].data(), upper_[i].size());
  io::write_array(out, data_.data(), count_ * stride_);
  if (!out) throw std::runtime_error("failed writing '" + path + "'");
}

std::unique_ptr<HnswIndex> HnswIndex::load(const std::string& path) {
  // Every value read here is untrusted. Each one is checked before it is used for an
  // allocation or as an index, so a malformed file throws instead of crashing a later search.
  auto in = io::open_in(path);
  io::check_header(in, kMagic, kVersion);
  const auto dim = io::read_pod<std::uint64_t>(in);
  const auto metric = io::read_pod<std::uint8_t>(in);
  HnswParams p;
  p.M = io::read_pod<std::uint64_t>(in);
  p.ef_construction = io::read_pod<std::uint64_t>(in);
  p.ef_search = io::read_pod<std::uint64_t>(in);
  p.seed = io::read_pod<std::uint64_t>(in);
  p.use_heuristic = io::read_pod<std::uint8_t>(in) != 0;
  if (metric > 2) io::corrupt("unknown metric");
  if (dim == 0 || dim > io::kMaxFileDim) io::corrupt("dimension out of range");
  if (p.M < 2 || p.M > io::kMaxFileM) io::corrupt("M out of range");
  if (p.ef_construction == 0) io::corrupt("ef_construction is 0");
  auto index = std::make_unique<HnswIndex>(dim, static_cast<Metric>(metric), p);
  const auto count = io::read_pod<std::uint64_t>(in);
  const auto max_level = io::read_pod<std::int32_t>(in);
  const auto entry = io::read_pod<std::uint32_t>(in);

  // Fixed bytes per node: label, level, deleted flag, layer-0 block, vector. The upper-layer
  // blocks come on top, so this is a lower bound that also caps count before we allocate.
  const std::uint64_t link_bytes = (1 + index->M0_) * sizeof(std::uint32_t);
  const std::uint64_t vector_bytes = index->stride_ * sizeof(float);
  const std::uint64_t fixed_per_node = sizeof(std::int64_t) + 2 + link_bytes + vector_bytes;
  const std::uint64_t remaining = io::remaining_bytes(in);
  if (count > std::numeric_limits<std::uint32_t>::max() || count > remaining / fixed_per_node)
    io::corrupt("node count does not match file size");

  if (count == 0) {
    if (max_level != -1 || entry != 0) io::corrupt("empty index has an entry point");
    if (remaining != 0) io::corrupt("unexpected trailing data");
    return index;
  }
  if (max_level < 0 || max_level > kMaxLevel) io::corrupt("max level out of range");
  if (entry >= count) io::corrupt("entry point out of range");

  index->reserve(count);
  io::read_array(in, index->labels_.data(), count);
  io::read_array(in, index->levels_.data(), count);
  io::read_array(in, index->deleted_.data(), count);

  int top = 0;
  std::uint64_t upper_ints = 0;
  for (std::size_t i = 0; i < count; ++i) {
    if (index->levels_[i] > kMaxLevel) io::corrupt("node level out of range");
    if (index->deleted_[i] > 1) io::corrupt("invalid deleted flag");
    top = std::max<int>(top, index->levels_[i]);
    upper_ints += std::uint64_t{index->levels_[i]} * (1 + p.M);
  }
  if (top != max_level) io::corrupt("max level does not match node levels");
  if (index->levels_[entry] != max_level) io::corrupt("entry point is not on the top layer");

  // Now the exact size of the rest of the file is known.
  const std::uint64_t level0_ints = io::checked_mul(count, 1 + index->M0_);
  const std::uint64_t floats = io::checked_mul(count, index->stride_);
  const std::uint64_t expected =
      io::checked_add(io::checked_mul(io::checked_add(level0_ints, upper_ints), 4),
                      io::checked_mul(floats, sizeof(float)));
  if (io::remaining_bytes(in) != expected) io::corrupt("file size does not match its header");

  io::read_array(in, index->level0_.data(), level0_ints);
  for (std::size_t i = 0; i < count; ++i) {
    index->upper_[i].resize(std::size_t(index->levels_[i]) * (1 + p.M));
    io::read_array(in, index->upper_[i].data(), index->upper_[i].size());
  }
  io::read_array(in, index->data_.data(), floats);

  // Every edge on layer l must point to an existing node that is itself on layer l, and no
  // neighbor list may exceed its layer's capacity: search relies on both without checking.
  for (std::uint32_t id = 0; id < count; ++id) {
    for (int l = 0; l <= index->levels_[id]; ++l) {
      const std::uint32_t* block = index->links(id, l);
      if (block[0] > index->max_links(l)) io::corrupt("neighbor count exceeds capacity");
      for (std::uint32_t j = 0; j < block[0]; ++j) {
        const std::uint32_t nb = block[1 + j];
        if (nb >= count) io::corrupt("edge to a node that does not exist");
        if (index->levels_[nb] < l) io::corrupt("edge to a node not on that layer");
      }
    }
  }

  index->count_ = count;
  index->max_level_ = max_level;
  index->entry_point_ = entry;
  for (std::uint32_t id = 0; id < count; ++id) {
    if (index->deleted_[id]) {
      ++index->num_deleted_;
      continue;
    }
    const std::int64_t label = index->labels_[id];
    if (label == kNoId) io::corrupt("label -1 is reserved");
    if (!index->label_to_id_.emplace(label, id).second) io::corrupt("duplicate live label");
  }
  // Continue the level sequence deterministically for vectors added after loading.
  index->rng_.seed(p.seed + count);
  return index;
}

}  // namespace vecsearch
