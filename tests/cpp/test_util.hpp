#pragma once

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <random>
#include <set>
#include <string>
#include <vector>

#include <gtest/gtest.h>
#include <unistd.h>

#include "vecsearch/search_result.hpp"

namespace vecsearch::test {

// A temporary file path unique to this process and test, so `ctest -j` can run tests in
// parallel without two of them writing the same file.
inline std::string unique_temp_path(const std::string& name) {
  const auto* info = ::testing::UnitTest::GetInstance()->current_test_info();
  std::string tag = std::to_string(::getpid());
  if (info) tag += std::string("_") + info->test_suite_name() + "_" + info->name();
  for (char& c : tag)
    if (c == '/') c = '_';
  return (std::filesystem::temp_directory_path() / (tag + "_" + name)).string();
}

inline std::vector<float> random_vectors(std::size_t n, std::size_t dim, std::uint32_t seed) {
  std::mt19937 rng(seed);
  std::normal_distribution<float> nd(0.0f, 1.0f);
  std::vector<float> v(n * dim);
  for (auto& x : v) x = nd(rng);
  return v;
}

// Gaussian clusters: `clusters` random centers, points = center + sigma * noise.
inline std::vector<float> clustered_vectors(std::size_t n, std::size_t dim, std::size_t clusters,
                                            float sigma, std::uint32_t seed) {
  std::mt19937 rng(seed);
  std::normal_distribution<float> nd(0.0f, 1.0f);
  std::vector<float> centers(clusters * dim);
  for (auto& x : centers) x = nd(rng) * 10.0f;
  std::vector<float> v(n * dim);
  for (std::size_t i = 0; i < n; ++i) {
    const std::size_t c = rng() % clusters;
    for (std::size_t j = 0; j < dim; ++j) v[i * dim + j] = centers[c * dim + j] + sigma * nd(rng);
  }
  return v;
}

// Fraction of the true top-k (from `truth`) found in `got`, averaged over queries.
inline double recall_at_k(const SearchResult& got, const SearchResult& truth, std::size_t k) {
  double hits = 0.0;
  for (std::size_t q = 0; q < truth.num_queries; ++q) {
    std::set<std::int64_t> expected(truth.row_ids(q), truth.row_ids(q) + k);
    for (std::size_t j = 0; j < k; ++j) hits += expected.count(got.row_ids(q)[j]);
  }
  return hits / static_cast<double>(truth.num_queries * k);
}

}  // namespace vecsearch::test
