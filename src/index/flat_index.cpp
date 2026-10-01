#include "vecsearch/flat_index.hpp"

#include <algorithm>
#include <cstring>
#include <stdexcept>
#include <vector>

#include "vecsearch/io.hpp"
#include "vecsearch/parallel.hpp"
#include "vecsearch/topk.hpp"

namespace vecsearch {

namespace {
constexpr char kMagic[9] = "VSFLAT01";
constexpr std::uint32_t kVersion = 1;
}  // namespace

FlatIndex::FlatIndex(std::size_t dim, Metric metric)
    : dim_(dim), stride_(padded_dim(dim)), metric_(metric) {
  if (dim == 0) throw std::invalid_argument("dim must be positive");
}

void FlatIndex::add(const float* vectors, std::size_t n) {
  if (n == 0) return;
  // Grow geometrically so repeated small adds stay amortized O(1) per vector.
  const std::size_t needed = (size_ + n) * stride_;
  if (needed > data_.capacity()) data_.reserve(std::max(needed, data_.capacity() * 2));
  data_.resize(needed, 0.0f);  // padding floats stay zero
  for (std::size_t i = 0; i < n; ++i) {
    float* row = data_.data() + (size_ + i) * stride_;
    std::memcpy(row, vectors + i * dim_, dim_ * sizeof(float));
    if (metric_ == Metric::Cosine) normalize(row, dim_);
  }
  size_ += n;
}

SearchResult FlatIndex::search(const float* queries, std::size_t nq, std::size_t k,
                               std::size_t num_threads) const {
  SearchResult result(nq, k);
  if (k == 0 || nq == 0) return result;
  const Distance dist = make_distance(metric_);

  parallel_for(nq, num_threads, [&](std::size_t q, std::size_t) {
    const float* query = queries + q * dim_;
    std::vector<float> normalized;
    if (metric_ == Metric::Cosine) {
      normalized.assign(query, query + dim_);
      normalize(normalized.data(), dim_);
      query = normalized.data();
    }
    TopK<std::int64_t> top(k);
    for (std::size_t i = 0; i < size_; ++i) {
      top.push(dist(query, vector(i), dim_), static_cast<std::int64_t>(i));
    }
    const auto best = top.take_sorted();
    for (std::size_t j = 0; j < best.size(); ++j) {
      result.ids[q * k + j] = best[j].id;
      result.distances[q * k + j] = best[j].dist;
    }
  });
  return result;
}

void FlatIndex::save(const std::string& path) const {
  auto out = io::open_out(path);
  io::write_header(out, kMagic, kVersion);
  io::write_pod<std::uint64_t>(out, dim_);
  io::write_pod<std::uint8_t>(out, static_cast<std::uint8_t>(metric_));
  io::write_pod<std::uint64_t>(out, size_);
  io::write_array(out, data_.data(), size_ * stride_);
  if (!out) throw std::runtime_error("failed writing '" + path + "'");
}

FlatIndex FlatIndex::load(const std::string& path) {
  auto in = io::open_in(path);
  io::check_header(in, kMagic, kVersion);
  const auto dim = io::read_pod<std::uint64_t>(in);
  const auto metric = io::read_pod<std::uint8_t>(in);
  if (metric > 2) throw std::runtime_error("index file has an unknown metric");
  FlatIndex index(dim, static_cast<Metric>(metric));
  index.size_ = io::read_pod<std::uint64_t>(in);
  index.data_.resize(index.size_ * index.stride_);
  io::read_array(in, index.data_.data(), index.data_.size());
  return index;
}

}  // namespace vecsearch
