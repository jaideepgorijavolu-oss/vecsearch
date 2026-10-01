#pragma once

#include <cstddef>
#include <cstdint>
#include <limits>
#include <vector>

namespace vecsearch {

inline constexpr std::int64_t kNoId = -1;
inline constexpr float kNoDistance = std::numeric_limits<float>::infinity();

// Results for a batch of queries, row-major: row q holds the k best ids (and distances) for
// query q, best first. When fewer than k results exist the row is padded with kNoId / +inf.
struct SearchResult {
  std::size_t num_queries = 0;
  std::size_t k = 0;
  std::vector<std::int64_t> ids;
  std::vector<float> distances;

  SearchResult() = default;
  SearchResult(std::size_t nq, std::size_t k_)
      : num_queries(nq), k(k_), ids(nq * k_, kNoId), distances(nq * k_, kNoDistance) {}

  const std::int64_t* row_ids(std::size_t q) const { return ids.data() + q * k; }
  const float* row_distances(std::size_t q) const { return distances.data() + q * k; }
};

}  // namespace vecsearch
