#pragma once

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <random>
#include <string>
#include <unordered_map>
#include <vector>

#include "vecsearch/aligned_allocator.hpp"
#include "vecsearch/distance.hpp"
#include "vecsearch/search_result.hpp"
#include "vecsearch/spinlock.hpp"
#include "vecsearch/topk.hpp"
#include "vecsearch/visited_list.hpp"

namespace vecsearch {

struct HnswParams {
  std::size_t M = 16;                    // max neighbors per node on layers >= 1 (layer 0: 2*M)
  std::size_t ef_construction = 200;     // beam width while building
  std::size_t ef_search = 50;            // default beam width while searching
  std::uint64_t seed = 100;              // level assignment RNG seed
  bool use_heuristic = true;             // Algorithm 4 neighbor selection; false = plain closest-M
  std::size_t sequential_prefix = 1000;  // the first nodes of an index are inserted on one thread
};

// Hierarchical Navigable Small World graph (Malkov & Yashunin, 2018).
//
// Thread safety: add() and search() each run in parallel internally, but add() must not run
// concurrently with search() or remove() on the same index (the service layer wraps each index
// in a reader-writer lock). Concurrent search() calls are safe.
class HnswIndex {
 public:
  HnswIndex(std::size_t dim, Metric metric, HnswParams params = {});

  // Inserts n vectors. labels[i] is the user-visible id of vector i; with labels == nullptr,
  // vector i gets label element_count() + i. Adding an existing label replaces that vector
  // (the old node is soft-deleted). num_threads = 0 uses every core.
  void add(const float* vectors, std::size_t n, const std::int64_t* labels = nullptr,
           std::size_t num_threads = 0);

  // k nearest live neighbors per query. ef = 0 uses ef_search(); the effective beam width is
  // max(ef, k). If allowed_labels is given, only those labels can be returned (filtered nodes
  // are still traversed); see DESIGN.md for the recall cost of restrictive filters.
  SearchResult search(const float* queries, std::size_t nq, std::size_t k, std::size_t ef = 0,
                      std::size_t num_threads = 0, const std::int64_t* allowed_labels = nullptr,
                      std::size_t num_allowed = 0) const;

  // Soft delete: the node stays in the graph (and is traversed) but is never returned.
  // Returns false if the label is not present.
  bool remove(std::int64_t label);
  bool contains(std::int64_t label) const { return label_to_id_.count(label) != 0; }

  void set_ef_search(std::size_t ef) { ef_search_.store(ef, std::memory_order_relaxed); }
  std::size_t ef_search() const { return ef_search_.load(std::memory_order_relaxed); }
  // Software prefetching of neighbor vectors during search (see RESULTS.md, Phase 4).
  void set_prefetch(bool on) { prefetch_ = on; }
  bool prefetch() const { return prefetch_; }

  std::size_t size() const { return count_ - num_deleted_; }  // live vectors
  std::size_t element_count() const { return count_; }        // including deleted
  std::size_t dim() const { return dim_; }
  Metric metric() const { return metric_; }
  const HnswParams& params() const { return params_; }
  int max_level() const { return max_level_; }
  std::size_t memory_bytes() const;

  void save(const std::string& path) const;
  static std::unique_ptr<HnswIndex> load(const std::string& path);

  // ---- Introspection for tests ----
  int level_of(std::uint32_t id) const { return levels_[id]; }
  std::vector<std::uint32_t> neighbors(std::uint32_t id, int level) const;

 private:
  using Cand = Scored<std::uint32_t>;

  // Per-query working memory. Each thread uses its own, so search shares no mutable state.
  struct Scratch {
    VisitedList visited;
    std::vector<Cand> candidates;  // min-heap (closest on top) of nodes to expand
    std::vector<Cand> results;     // max-heap (furthest on top) of the best ef found
    std::vector<std::uint32_t> neighbor_copy;
    std::vector<std::uint32_t> unvisited;  // neighbors of the node being expanded, not yet seen
    std::vector<float> query;              // normalized query copy (cosine)
  };
  class ScratchPool;

  // Layout of a neighbor block: [count, n_0, n_1, ..., n_{max-1}] as uint32.
  std::uint32_t* links(std::uint32_t id, int level) {
    return level == 0 ? level0_.data() + std::size_t{id} * (1 + M0_)
                      : upper_[id].data() + std::size_t(level - 1) * (1 + params_.M);
  }
  const std::uint32_t* links(std::uint32_t id, int level) const {
    return const_cast<HnswIndex*>(this)->links(id, level);
  }
  std::size_t max_links(int level) const { return level == 0 ? M0_ : params_.M; }
  const float* vec(std::uint32_t id) const { return data_.data() + std::size_t{id} * stride_; }
  float dist(const float* a, std::uint32_t b) const { return dist_(a, vec(b), dim_); }

  void reserve(std::size_t capacity);
  int random_level();
  void insert(std::uint32_t id, Scratch& s);

  // Algorithm 2: beam search on one layer starting from ep. Leaves the best ef nodes that pass
  // `accept` in s.results as a max-heap. kLocked = take node locks (needed during add()).
  template <bool kLocked, class Accept>
  void search_layer(const float* q, std::uint32_t ep, std::size_t ef, int level, Scratch& s,
                    const Accept& accept) const;

  // Greedy descent (ef = 1) from ep through layers top..(bottom+1). Returns the closest node.
  template <bool kLocked>
  std::uint32_t greedy_descent(const float* q, std::uint32_t ep, int top, int bottom,
                               Scratch& s) const;

  // Algorithm 4: picks up to m diverse neighbors from candidates (sorted closest first).
  void select_neighbors(std::vector<Cand>& candidates, std::size_t m) const;
  // Links id to the best of `candidates` (sorted closest first) on `level`, and back.
  void connect(std::uint32_t id, std::vector<Cand>& candidates, int level, Scratch& s);
  void prefetch_vector(std::uint32_t id) const;
  static void prefetch_address(const void* p);

  std::size_t dim_;
  std::size_t stride_;
  Metric metric_;
  Distance dist_;
  HnswParams params_;
  std::size_t M0_;
  double level_mult_;  // mL = 1 / ln(M)
  std::atomic<std::size_t> ef_search_;
  bool prefetch_ = true;

  std::size_t count_ = 0;     // nodes [0, count_) exist
  std::size_t capacity_ = 0;  // storage allocated for this many nodes
  std::size_t num_deleted_ = 0;

  AlignedVector<float> data_;                      // capacity_ rows of stride_ floats
  AlignedVector<std::uint32_t> level0_;            // capacity_ blocks of (1 + M0) uint32
  std::vector<std::vector<std::uint32_t>> upper_;  // layers 1..level, (1 + M) uint32 each
  std::vector<std::uint8_t> levels_;
  std::vector<std::uint8_t> deleted_;
  std::vector<std::int64_t> labels_;                             // internal id -> label
  std::unordered_map<std::int64_t, std::uint32_t> label_to_id_;  // live labels only
  std::unique_ptr<SpinLock[]> node_locks_;

  std::mutex global_mu_;  // guards entry_point_ / max_level_ during add()
  std::uint32_t entry_point_ = 0;
  int max_level_ = -1;  // -1: empty
  std::mt19937_64 rng_;

  std::shared_ptr<ScratchPool> scratch_pool_;
};

}  // namespace vecsearch
