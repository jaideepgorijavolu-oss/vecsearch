#include <gtest/gtest.h>

#include <algorithm>
#include <filesystem>
#include <set>
#include <vector>

#include "test_util.hpp"
#include "vecsearch/flat_index.hpp"
#include "vecsearch/hnsw_index.hpp"
#include "vecsearch/visited_list.hpp"

using namespace vecsearch;

namespace {

SearchResult exact(const std::vector<float>& base, std::size_t dim, Metric m,
                   const std::vector<float>& queries, std::size_t k) {
  FlatIndex flat(dim, m);
  flat.add(base.data(), base.size() / dim);
  return flat.search(queries.data(), queries.size() / dim, k);
}

std::string temp_path(const char* name) {
  return vecsearch::test::unique_temp_path(name);
}

}  // namespace

TEST(VisitedList, EpochReset) {
  VisitedList v;
  v.reset(10);
  EXPECT_FALSE(v.test_and_set(3));
  EXPECT_TRUE(v.test_and_set(3));
  v.reset(10);
  EXPECT_FALSE(v.contains(3));
  // Survives the 16-bit epoch wrapping around.
  for (int i = 0; i < 70000; ++i) {
    v.reset(10);
    ASSERT_FALSE(v.test_and_set(5));
  }
  v.reset(20);  // growing keeps working
  EXPECT_FALSE(v.contains(19));
}

class HnswRecall : public ::testing::TestWithParam<Metric> {};

TEST_P(HnswRecall, HighRecallOnRandomData) {
  const std::size_t n = 5000, dim = 32, nq = 200, k = 10;
  const auto base = test::random_vectors(n, dim, 1);
  const auto queries = test::random_vectors(nq, dim, 2);
  HnswIndex index(dim, GetParam(), {.M = 16, .ef_construction = 200});
  index.add(base.data(), n, nullptr, 4);
  const auto truth = exact(base, dim, GetParam(), queries, k);
  const auto got = index.search(queries.data(), nq, k, 200);
  EXPECT_GE(test::recall_at_k(got, truth, k), 0.95);
}

INSTANTIATE_TEST_SUITE_P(AllMetrics, HnswRecall,
                         ::testing::Values(Metric::L2, Metric::InnerProduct, Metric::Cosine),
                         [](const auto& info) { return std::string(metric_name(info.param)); });

TEST(Hnsw, RecallGrowsWithEf) {
  const std::size_t n = 5000, dim = 32, nq = 200, k = 10;
  const auto base = test::random_vectors(n, dim, 3);
  const auto queries = test::random_vectors(nq, dim, 4);
  HnswIndex index(dim, Metric::L2, {.M = 8, .ef_construction = 100});
  index.add(base.data(), n, nullptr, 4);
  const auto truth = exact(base, dim, Metric::L2, queries, k);
  const double low = test::recall_at_k(index.search(queries.data(), nq, k, 10), truth, k);
  const double high = test::recall_at_k(index.search(queries.data(), nq, k, 300), truth, k);
  EXPECT_LT(low, high);
  EXPECT_GE(high, 0.97);
}

TEST(Hnsw, GraphInvariants) {
  const std::size_t n = 3000, dim = 16;
  const auto base = test::random_vectors(n, dim, 5);
  HnswIndex index(dim, Metric::L2, {.M = 8, .ef_construction = 64});
  index.add(base.data(), n, nullptr, 8);
  std::vector<std::size_t> per_level(32, 0);
  for (std::uint32_t id = 0; id < n; ++id) {
    const int top = index.level_of(id);
    ++per_level[top];
    for (int l = 0; l <= top; ++l) {
      const auto nb = index.neighbors(id, l);
      EXPECT_LE(nb.size(), l == 0 ? 16u : 8u);
      std::set<std::uint32_t> unique(nb.begin(), nb.end());
      EXPECT_EQ(unique.size(), nb.size()) << "duplicate neighbor";
      for (auto x : nb) {
        EXPECT_NE(x, id) << "self loop";
        ASSERT_LT(x, n);
        EXPECT_GE(index.level_of(x), l) << "neighbor not present on this layer";
      }
      if (l == 0) {
        EXPECT_GE(nb.size(), 1u) << "isolated node";
      }
    }
  }
  // Level sizes shrink by about M each layer: expect level-0-only nodes ~ (1 - 1/8) of n.
  EXPECT_NEAR(static_cast<double>(per_level[0]) / n, 1.0 - 1.0 / 8, 0.03);
  EXPECT_GE(index.max_level(), 2);
}

// Algorithm 4's diversity heuristic keeps links between clusters that "closest M" drops.
// On tightly clustered low-dimensional data that difference is what keeps the graph connected.
TEST(Hnsw, HeuristicBeatsClosestM) {
  const std::size_t n = 20000, dim = 8, nq = 500, k = 10;
  const auto base = test::clustered_vectors(n, dim, 100, 0.5f, 6);
  const auto queries = test::clustered_vectors(nq, dim, 100, 0.5f, 6 + 1000);
  const auto truth = exact(base, dim, Metric::L2, queries, k);
  double recall[2];
  for (int h = 0; h < 2; ++h) {
    HnswIndex index(dim, Metric::L2, {.M = 6, .ef_construction = 40, .use_heuristic = h == 1});
    index.add(base.data(), n, nullptr, 1);
    recall[h] = test::recall_at_k(index.search(queries.data(), nq, k, 20), truth, k);
  }
  std::printf("recall@10 at ef=20: closest-M %.4f, heuristic %.4f\n", recall[0], recall[1]);
  EXPECT_GT(recall[1], recall[0] + 0.02);
}

TEST(Hnsw, SingleThreadBuildIsDeterministic) {
  const auto base = test::random_vectors(1000, 16, 7);
  HnswIndex a(16, Metric::L2), b(16, Metric::L2);
  a.add(base.data(), 1000, nullptr, 1);
  b.add(base.data(), 1000, nullptr, 1);
  for (std::uint32_t id = 0; id < 1000; ++id) EXPECT_EQ(a.neighbors(id, 0), b.neighbors(id, 0));
}

TEST(Hnsw, ParallelSearchMatchesSerial) {
  const auto base = test::random_vectors(4000, 24, 8);
  const auto queries = test::random_vectors(300, 24, 9);
  HnswIndex index(24, Metric::L2);
  index.add(base.data(), 4000, nullptr, 8);
  const auto serial = index.search(queries.data(), 300, 10, 64, 1);
  const auto parallel = index.search(queries.data(), 300, 10, 64, 8);
  EXPECT_EQ(serial.ids, parallel.ids);
  EXPECT_EQ(serial.distances, parallel.distances);
}

TEST(Hnsw, EmptyAndTiny) {
  HnswIndex index(4, Metric::L2);
  const float q[4] = {0, 0, 0, 0};
  EXPECT_EQ(index.search(q, 1, 3).ids, std::vector<std::int64_t>(3, kNoId));
  const float one[4] = {1, 1, 1, 1};
  index.add(one, 1);
  const auto r = index.search(q, 1, 3);
  EXPECT_EQ(r.ids[0], 0);
  EXPECT_EQ(r.ids[1], kNoId);  // k > n pads
  EXPECT_FLOAT_EQ(r.distances[0], 4.0f);
}

TEST(Hnsw, IncrementalAddsAndLabels) {
  const auto base = test::random_vectors(2000, 16, 10);
  HnswIndex index(16, Metric::L2);
  std::vector<std::int64_t> labels(2000);
  for (std::size_t i = 0; i < 2000; ++i) labels[i] = 1000000 + static_cast<std::int64_t>(i) * 3;
  for (std::size_t off = 0; off < 2000; off += 500) {  // several adds, growing storage
    index.add(base.data() + off * 16, 500, labels.data() + off, 4);
  }
  EXPECT_EQ(index.size(), 2000u);
  const auto r = index.search(base.data(), 2000, 1, 50);
  std::size_t self = 0;
  for (std::size_t i = 0; i < 2000; ++i) self += r.ids[i] == labels[i];
  EXPECT_GE(self, 1990u);
}

TEST(Hnsw, SoftDeleteExcludesFromResults) {
  const std::size_t n = 3000, dim = 16, k = 10;
  const auto base = test::random_vectors(n, dim, 11);
  const auto queries = test::random_vectors(100, dim, 12);
  HnswIndex index(dim, Metric::L2);
  index.add(base.data(), n, nullptr, 4);
  for (std::int64_t id = 0; id < static_cast<std::int64_t>(n); id += 2)
    ASSERT_TRUE(index.remove(id));
  EXPECT_FALSE(index.remove(0));  // already gone
  EXPECT_EQ(index.size(), n / 2);
  const auto r = index.search(queries.data(), 100, k, 100);
  for (auto id : r.ids) EXPECT_EQ(id % 2, 1) << "deleted id returned";

  // Recall against brute force over the surviving (odd) vectors.
  std::vector<float> odd;
  for (std::size_t i = 1; i < n; i += 2)
    odd.insert(odd.end(), &base[i * dim], &base[i * dim] + dim);
  auto truth = exact(odd, dim, Metric::L2, queries, k);
  for (auto& id : truth.ids) id = id * 2 + 1;
  EXPECT_GE(test::recall_at_k(r, truth, k), 0.95);
}

TEST(Hnsw, UpsertReplacesVector) {
  const auto base = test::random_vectors(500, 8, 13);
  HnswIndex index(8, Metric::L2);
  index.add(base.data(), 500);
  const float moved[8] = {100, 100, 100, 100, 100, 100, 100, 100};
  const std::int64_t label = 42;
  index.add(moved, 1, &label);
  EXPECT_EQ(index.size(), 500u);
  EXPECT_EQ(index.element_count(), 501u);
  EXPECT_EQ(index.search(moved, 1, 1).ids[0], 42);
  const auto r = index.search(base.data() + 42 * 8, 1, 1);  // the old vector is gone
  EXPECT_NE(r.ids[0], 42);
}

TEST(Hnsw, FilteredSearchReturnsOnlyAllowed) {
  const std::size_t n = 4000, dim = 16, k = 10;
  const auto base = test::random_vectors(n, dim, 14);
  const auto queries = test::random_vectors(50, dim, 15);
  HnswIndex index(dim, Metric::L2);
  index.add(base.data(), n, nullptr, 4);
  std::vector<std::int64_t> allowed;
  for (std::int64_t i = 0; i < static_cast<std::int64_t>(n); i += 5) allowed.push_back(i);  // 20%
  const auto r = index.search(queries.data(), 50, k, 100, 0, allowed.data(), allowed.size());
  for (auto id : r.ids) EXPECT_EQ(id % 5, 0);

  std::vector<float> subset;
  for (auto i : allowed) subset.insert(subset.end(), &base[i * dim], &base[i * dim] + dim);
  auto truth = exact(subset, dim, Metric::L2, queries, k);
  for (auto& id : truth.ids) id *= 5;
  EXPECT_GE(test::recall_at_k(r, truth, k), 0.9);
}

TEST(Hnsw, SaveLoadIdenticalResults) {
  const auto base = test::random_vectors(3000, 20, 16);
  const auto queries = test::random_vectors(100, 20, 17);
  HnswIndex index(20, Metric::Cosine, {.M = 12, .ef_construction = 100, .ef_search = 40});
  index.add(base.data(), 3000, nullptr, 4);
  index.remove(7);
  const auto path = temp_path("vecsearch_hnsw_test.bin");
  index.save(path);
  const auto loaded = HnswIndex::load(path);
  std::filesystem::remove(path);

  EXPECT_EQ(loaded->size(), index.size());
  EXPECT_EQ(loaded->ef_search(), 40u);
  EXPECT_FALSE(loaded->contains(7));
  const auto a = index.search(queries.data(), 100, 10);
  const auto b = loaded->search(queries.data(), 100, 10);
  EXPECT_EQ(a.ids, b.ids);
  EXPECT_EQ(a.distances, b.distances);

  // A loaded index keeps accepting vectors.
  loaded->add(queries.data(), 100);
  EXPECT_EQ(loaded->size(), index.size() + 100);
}

TEST(Hnsw, LoadRejectsWrongFile) {
  FlatIndex flat(4, Metric::L2);
  const auto path = temp_path("vecsearch_wrong_type.bin");
  flat.save(path);
  EXPECT_THROW(HnswIndex::load(path), std::runtime_error);
  std::filesystem::remove(path);
}

TEST(Hnsw, PrefetchDoesNotChangeResults) {
  const auto base = test::random_vectors(2000, 40, 18);
  HnswIndex index(40, Metric::L2);
  index.add(base.data(), 2000, nullptr, 4);
  index.set_prefetch(false);
  const auto a = index.search(base.data(), 100, 10);
  index.set_prefetch(true);
  const auto b = index.search(base.data(), 100, 10);
  EXPECT_EQ(a.ids, b.ids);
}

TEST(Hnsw, RejectsBadParams) {
  EXPECT_THROW(HnswIndex(0, Metric::L2), std::invalid_argument);
  EXPECT_THROW(HnswIndex(4, Metric::L2, {.M = 1}), std::invalid_argument);
}
