#pragma once

#include <cstddef>
#include <string>

#include "vecsearch/aligned_allocator.hpp"
#include "vecsearch/distance.hpp"
#include "vecsearch/search_result.hpp"

namespace vecsearch {

// Exact (brute-force) search. Every query is compared with every stored vector.
//
// This is the ground truth for every recall measurement of the approximate index, so it is kept
// deliberately simple. Ids are insertion order: the i-th vector ever added has id i.
class FlatIndex {
 public:
  FlatIndex(std::size_t dim, Metric metric);

  // Appends n vectors (row-major, n * dim floats). Cosine vectors are normalized on insert.
  void add(const float* vectors, std::size_t n);

  // k nearest neighbors of each of the nq queries. num_threads = 0 uses every core.
  SearchResult search(const float* queries, std::size_t nq, std::size_t k,
                      std::size_t num_threads = 0) const;

  std::size_t size() const { return size_; }
  std::size_t dim() const { return dim_; }
  Metric metric() const { return metric_; }
  // The stored (possibly normalized) vector with id i.
  const float* vector(std::size_t i) const { return data_.data() + i * stride_; }
  std::size_t memory_bytes() const { return data_.capacity() * sizeof(float); }

  void save(const std::string& path) const;
  static FlatIndex load(const std::string& path);

 private:
  std::size_t dim_;
  std::size_t stride_;  // floats per stored row (dim rounded up to 16, see padded_dim)
  Metric metric_;
  std::size_t size_ = 0;
  AlignedVector<float> data_;  // size_ rows of stride_ floats, contiguous
};

}  // namespace vecsearch
