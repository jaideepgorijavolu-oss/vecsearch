#pragma once

#include <cstddef>
#include <cstdint>
#include <istream>
#include <string>
#include <vector>

namespace vecsearch {

// Features a learned early-termination policy sees, computed from the search state when the
// layer-0 search reaches its checkpoint (a fixed number of distance evaluations). Every one of
// them is available at query time; nothing depends on ground truth or on how the search ends.
enum TerminationFeature : std::size_t {
  kLogEntryDist,      // log1p(distance to the layer-0 entry point, i.e. after greedy descent)
  kLogBestDist,       // log1p(d_1): closest result so far
  kLogKthDist,        // log1p(d_k): k-th closest result so far
  kBestOverEntry,     // d_1 / d_entry
  kKthOverEntry,      // d_k / d_entry
  kKthOverBest,       // d_k / d_1
  kCandidateOverKth,  // closest unexpanded candidate / d_k
  kStaleFraction,     // evaluations since the top-k last changed / evaluations so far
  kTopkChanges,       // number of times the top-k set changed so far
  kNumTerminationFeatures
};

// A small regression model that predicts log(total layer-0 distance evaluations a query needs),
// as in Li et al., "Improving Approximate Nearest Neighbor Search through Learned Adaptive
// Early Termination" (SIGMOD 2020). Two kinds: a linear model and a gradient-boosted tree
// ensemble, both trained offline (bench/ann/adaptive/study.py) and stored in a versioned text
// file. Optionally the raw query vector is appended to the features, as in the paper.
// Immutable after loading, so concurrent searches can share one model.
class TerminationModel {
 public:
  static TerminationModel load(const std::string& path);
  static TerminationModel parse(std::istream& in);  // throws std::runtime_error if malformed

  // Layer-0 distance evaluations after which the model is consulted.
  std::size_t checkpoint() const { return checkpoint_; }
  // Dimension of the query vector appended to the features (0 = features only).
  std::size_t query_dim() const { return query_dim_; }

  // features: kNumTerminationFeatures values; query: query_dim() floats (may be null if 0).
  double predict_log_evals(const float* features, const float* query) const;

 private:
  struct Node {
    std::int32_t feature;  // -1 = leaf
    double threshold;      // go left if x <= threshold
    std::int32_t left, right;
    double value;  // leaf output (already scaled by the learning rate)
  };
  double input(std::size_t i, const float* features, const float* query) const {
    return i < kNumTerminationFeatures ? features[i] : query[i - kNumTerminationFeatures];
  }

  bool linear_ = true;
  std::size_t checkpoint_ = 0;
  std::size_t query_dim_ = 0;
  double bias_ = 0;
  std::vector<double> weights_;            // linear
  std::vector<Node> nodes_;                // trees, concatenated
  std::vector<std::uint32_t> tree_roots_;  // index of each tree's root in nodes_
};

}  // namespace vecsearch
