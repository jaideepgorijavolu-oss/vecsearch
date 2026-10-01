#include <gtest/gtest.h>

#include <cstdint>
#include <filesystem>
#include <fstream>
#include <vector>

#include "test_util.hpp"
#include "vecsearch/aligned_allocator.hpp"
#include "vecsearch/flat_index.hpp"
#include "vecsearch/topk.hpp"

using namespace vecsearch;

TEST(TopK, KeepsSmallestSorted) {
  TopK<int> top(3);
  const float d[] = {5, 1, 4, 2, 8, 0.5f, 3};
  for (int i = 0; i < 7; ++i) top.push(d[i], i);
  const auto best = top.take_sorted();
  ASSERT_EQ(best.size(), 3u);
  EXPECT_EQ(best[0].id, 5);
  EXPECT_EQ(best[1].id, 1);
  EXPECT_EQ(best[2].id, 3);
}

TEST(TopK, TiesBrokenById) {
  TopK<int> top(2);
  for (int i = 4; i >= 0; --i) top.push(1.0f, i);
  const auto best = top.take_sorted();
  EXPECT_EQ(best[0].id, 0);
  EXPECT_EQ(best[1].id, 1);
}

TEST(TopK, ZeroK) {
  TopK<int> top(0);
  top.push(1.0f, 1);
  EXPECT_EQ(top.size(), 0u);
}

TEST(AlignedStorage, RowsAreCacheLineAligned) {
  FlatIndex index(37, Metric::L2);
  const auto data = test::random_vectors(10, 37, 1);
  index.add(data.data(), 10);
  for (std::size_t i = 0; i < 10; ++i) {
    EXPECT_EQ(reinterpret_cast<std::uintptr_t>(index.vector(i)) % kCacheLine, 0u);
  }
  EXPECT_EQ(padded_dim(37), 48u);
  EXPECT_EQ(padded_dim(128), 128u);
}

namespace {

struct Fixture {
  std::int32_t n, dim, nq, k;
  std::vector<float> base, queries;
  std::vector<std::int64_t> ids[3];
};

Fixture load_fixture() {
  const auto path = std::filesystem::path(VECSEARCH_TEST_DIR) / "fixtures/flat_numpy.bin";
  std::ifstream in(path, std::ios::binary);
  if (!in) throw std::runtime_error("missing fixture " + path.string());
  Fixture f{};
  in.read(reinterpret_cast<char*>(&f.n), 16);
  f.base.resize(static_cast<std::size_t>(f.n) * f.dim);
  f.queries.resize(static_cast<std::size_t>(f.nq) * f.dim);
  in.read(reinterpret_cast<char*>(f.base.data()), f.base.size() * 4);
  in.read(reinterpret_cast<char*>(f.queries.data()), f.queries.size() * 4);
  for (auto& ids : f.ids) {
    ids.resize(static_cast<std::size_t>(f.nq) * f.k);
    in.read(reinterpret_cast<char*>(ids.data()), ids.size() * 8);
  }
  if (!in) throw std::runtime_error("truncated fixture");
  return f;
}

}  // namespace

class FlatVsNumpy : public ::testing::TestWithParam<Metric> {};

TEST_P(FlatVsNumpy, IdsMatchExactly) {
  const Fixture f = load_fixture();
  const Metric metric = GetParam();
  FlatIndex index(f.dim, metric);
  index.add(f.base.data(), f.n);
  for (std::size_t threads : {1u, 4u}) {
    const auto res = index.search(f.queries.data(), f.nq, f.k, threads);
    EXPECT_EQ(res.ids, f.ids[static_cast<int>(metric)]) << "threads=" << threads;
  }
}

INSTANTIATE_TEST_SUITE_P(AllMetrics, FlatVsNumpy,
                         ::testing::Values(Metric::L2, Metric::InnerProduct, Metric::Cosine),
                         [](const auto& info) { return std::string(metric_name(info.param)); });

TEST(Flat, KLargerThanSizePads) {
  FlatIndex index(4, Metric::L2);
  const auto data = test::random_vectors(3, 4, 2);
  index.add(data.data(), 3);
  const auto res = index.search(data.data(), 1, 5);
  EXPECT_EQ(res.ids[0], 0);
  for (int j = 0; j < 3; ++j) EXPECT_NE(res.ids[j], kNoId);
  EXPECT_EQ(res.ids[3], kNoId);
  EXPECT_EQ(res.ids[4], kNoId);
  EXPECT_EQ(res.distances[4], kNoDistance);
}

TEST(Flat, EmptyIndex) {
  FlatIndex index(4, Metric::L2);
  const float q[4] = {1, 2, 3, 4};
  const auto res = index.search(q, 1, 3);
  EXPECT_EQ(res.ids, std::vector<std::int64_t>(3, kNoId));
}

TEST(Flat, SelfIsNearest) {
  FlatIndex index(16, Metric::L2);
  const auto data = test::random_vectors(100, 16, 3);
  index.add(data.data(), 50);
  index.add(data.data() + 50 * 16, 50);  // two adds behave like one
  const auto res = index.search(data.data(), 100, 1);
  for (std::size_t i = 0; i < 100; ++i) {
    EXPECT_EQ(res.ids[i], static_cast<std::int64_t>(i));
    EXPECT_FLOAT_EQ(res.distances[i], 0.0f);
  }
}

TEST(Flat, CosineIsScaleInvariant) {
  FlatIndex index(8, Metric::Cosine);
  auto data = test::random_vectors(20, 8, 4);
  index.add(data.data(), 20);
  for (auto& x : data) x *= 7.5f;  // scaled queries must give identical neighbors
  const auto a = index.search(data.data(), 20, 5);
  for (std::size_t i = 0; i < 20; ++i) EXPECT_EQ(a.ids[i * 5], static_cast<std::int64_t>(i));
}

TEST(Flat, SaveLoadRoundTrip) {
  FlatIndex index(37, Metric::InnerProduct);
  const auto data = test::random_vectors(200, 37, 5);
  index.add(data.data(), 200);
  const auto path = (std::filesystem::temp_directory_path() / "vecsearch_flat_test.bin").string();
  index.save(path);
  const FlatIndex loaded = FlatIndex::load(path);
  std::filesystem::remove(path);
  EXPECT_EQ(loaded.size(), 200u);
  EXPECT_EQ(loaded.metric(), Metric::InnerProduct);
  const auto a = index.search(data.data(), 20, 10);
  const auto b = loaded.search(data.data(), 20, 10);
  EXPECT_EQ(a.ids, b.ids);
  EXPECT_EQ(a.distances, b.distances);
}

TEST(Flat, LoadRejectsGarbage) {
  const auto path = (std::filesystem::temp_directory_path() / "vecsearch_garbage.bin").string();
  {
    std::ofstream out(path, std::ios::binary);
    out << "definitely not an index";
  }
  EXPECT_THROW(FlatIndex::load(path), std::runtime_error);
  std::filesystem::remove(path);
}
