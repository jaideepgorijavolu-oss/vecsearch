// search_adaptive: early termination by distance budget, patience or a learned budget.
#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <sstream>
#include <vector>

#include "test_util.hpp"
#include "vecsearch/hnsw_index.hpp"
#include "vecsearch/termination_model.hpp"

using namespace vecsearch;
using vecsearch::test::clustered_vectors;
using vecsearch::test::random_vectors;

namespace {

constexpr std::size_t kDim = 16, kN = 4000, kQ = 50, kK = 10, kEf = 64;

struct Fixture {
  std::vector<float> base = clustered_vectors(kN, kDim, 20, 0.3f, 1);
  std::vector<float> queries = clustered_vectors(kQ, kDim, 20, 0.3f, 2);
  HnswIndex index{kDim, Metric::L2, {.M = 8, .ef_construction = 100}};
  Fixture() { index.add(base.data(), kN, nullptr, 1); }
};

Fixture& fixture() {
  static Fixture f;
  return f;
}

TerminationModel parse(const std::string& text) {
  std::istringstream in(text);
  return TerminationModel::parse(in);
}

// A model that always predicts `evals` evaluations.
TerminationModel constant_model(double evals, std::size_t checkpoint) {
  std::ostringstream s;
  s.precision(17);
  s << "vecsearch-termination-model 1 kind linear checkpoint " << checkpoint << " query_dim 0 bias "
    << std::log(evals) << " weights " << kNumTerminationFeatures;
  for (std::size_t i = 0; i < kNumTerminationFeatures; ++i) s << " 0";
  s << " end";
  return parse(s.str());
}

// Sorted distances of the top-k after `budget` evaluations, replayed from the event log.
std::vector<float> replay(const std::vector<TopkEvent>& events, std::size_t budget) {
  std::vector<float> d;
  for (const auto& e : events)
    if (e.eval <= budget) d.push_back(e.dist);
  std::sort(d.begin(), d.end());
  d.resize(std::min(d.size(), kK));
  return d;
}

std::vector<float> row_dists(const SearchResult& r, std::size_t q) {
  std::vector<float> d(r.row_distances(q), r.row_distances(q) + kK);
  std::sort(d.begin(), d.end());
  return d;
}

}  // namespace

TEST(Adaptive, WithoutTerminationEqualsSearch) {
  for (Metric m : {Metric::L2, Metric::InnerProduct, Metric::Cosine}) {
    auto base = random_vectors(2000, kDim, 3), queries = random_vectors(kQ, kDim, 4);
    HnswIndex index(kDim, m, {.M = 8, .ef_construction = 100});
    index.add(base.data(), 2000, nullptr, 1);
    const auto a = index.search(queries.data(), kQ, kK, kEf, 1);
    const auto b = index.search_adaptive(queries.data(), kQ, kK, {.ef = kEf}, nullptr, nullptr, 1);
    EXPECT_EQ(a.ids, b.ids) << metric_name(m);
    EXPECT_EQ(a.distances, b.distances) << metric_name(m);
    // Deleted nodes take the accept-filter path in both.
    for (std::int64_t l = 0; l < 2000; l += 7) index.remove(l);
    const auto c = index.search(queries.data(), kQ, kK, kEf, 1);
    const auto d = index.search_adaptive(queries.data(), kQ, kK, {.ef = kEf}, nullptr, nullptr, 1);
    EXPECT_EQ(c.ids, d.ids) << metric_name(m);
  }
}

// A budgeted search is a prefix of the unbudgeted one: its result is exactly the top-k after B
// evaluations. Training labels and the offline budget simulation rely on this.
TEST(Adaptive, BudgetIsPrefixOfFullRun) {
  auto& f = fixture();
  std::vector<AdaptiveStats> full(kQ);
  std::vector<std::vector<TopkEvent>> events;
  f.index.search_adaptive(f.queries.data(), kQ, kK, {.ef = 256}, full.data(), &events, 1);
  for (std::size_t budget : {1, 10, 37, 100, 250, 600, 5000}) {
    std::vector<AdaptiveStats> st(kQ);
    const auto r = f.index.search_adaptive(f.queries.data(), kQ, kK,
                                           {.ef = 256, .max_evals = budget}, st.data(), nullptr, 1);
    for (std::size_t q = 0; q < kQ; ++q) {
      ASSERT_EQ(st[q].evals, std::min<std::size_t>(budget, full[q].evals));
      EXPECT_EQ(st[q].stop, budget < full[q].evals ? StopReason::Budget : StopReason::Converged);
      if (budget >= kK) {
        EXPECT_EQ(row_dists(r, q), replay(events[q], budget)) << q << " " << budget;
      }
    }
  }
}

TEST(Adaptive, PatienceStopsEarlyAndHugePatienceChangesNothing) {
  auto& f = fixture();
  std::vector<AdaptiveStats> full(kQ), p1(kQ), big(kQ);
  const auto a =
      f.index.search_adaptive(f.queries.data(), kQ, kK, {.ef = kEf}, full.data(), nullptr, 1);
  f.index.search_adaptive(f.queries.data(), kQ, kK, {.ef = kEf, .patience = 1}, p1.data(), nullptr,
                          1);
  const auto c = f.index.search_adaptive(f.queries.data(), kQ, kK,
                                         {.ef = kEf, .patience = 1u << 30}, big.data(), nullptr, 1);
  EXPECT_EQ(a.ids, c.ids);
  std::size_t stopped = 0;
  for (std::size_t q = 0; q < kQ; ++q) {
    EXPECT_LE(p1[q].evals, full[q].evals);
    stopped += p1[q].stop == StopReason::Patience;
    EXPECT_EQ(big[q].evals, full[q].evals);
  }
  EXPECT_GT(stopped, 0u);
}

TEST(Adaptive, FeaturesAtCheckpoint) {
  auto& f = fixture();
  std::vector<AdaptiveStats> st(kQ);
  const auto r = f.index.search_adaptive(f.queries.data(), kQ, kK, {.ef = kEf, .checkpoint = 50},
                                         st.data(), nullptr, 1);
  const auto plain = f.index.search(f.queries.data(), kQ, kK, kEf, 1);
  EXPECT_EQ(r.ids, plain.ids);  // recording features does not change the search
  for (const auto& s : st) {
    ASSERT_TRUE(s.features_valid);
    EXPECT_FALSE(s.model_used);
    EXPECT_LE(s.features[kBestOverEntry], 1.0f);
    EXPECT_GE(s.features[kKthOverBest], 1.0f);
    EXPECT_GE(s.features[kStaleFraction], 0.0f);
    EXPECT_LT(s.features[kStaleFraction], 1.0f);
    EXPECT_GE(s.features[kTopkChanges], float(kK));
  }
  // A checkpoint the search never reaches leaves the features invalid.
  f.index.search_adaptive(f.queries.data(), kQ, kK, {.ef = kEf, .checkpoint = 1u << 30}, st.data(),
                          nullptr, 1);
  for (const auto& s : st) EXPECT_FALSE(s.features_valid);
}

TEST(Adaptive, ModelSetsBudgetAndScalesWithMultiplier) {
  auto& f = fixture();
  const auto model = constant_model(200, 40);
  for (double mult : {0.5, 1.0, 2.0}) {
    std::vector<AdaptiveStats> full(kQ), st(kQ);
    f.index.search_adaptive(f.queries.data(), kQ, kK, {.ef = 256}, full.data(), nullptr, 1);
    f.index.search_adaptive(f.queries.data(), kQ, kK,
                            {.ef = 256, .model = &model, .multiplier = mult}, st.data(), nullptr,
                            1);
    const auto budget = static_cast<std::uint32_t>(std::ceil(200 * mult - 1e-9));
    for (std::size_t q = 0; q < kQ; ++q) {
      ASSERT_TRUE(st[q].model_used);
      EXPECT_NEAR(double(st[q].budget), double(std::max<std::uint32_t>(budget, 40)), 1.0);
      EXPECT_EQ(st[q].evals, std::min(st[q].budget, full[q].evals));
    }
  }
}

TEST(Adaptive, ZeroDistanceFallsBackToPlainSearch) {
  auto& f = fixture();
  // Queries equal to indexed vectors: d_1 = 0, so d_k / d_1 is not finite.
  std::vector<float> q(f.base.begin(), f.base.begin() + 5 * kDim);
  const auto model = constant_model(20, 30);
  std::vector<AdaptiveStats> st(5);
  const auto r =
      f.index.search_adaptive(q.data(), 5, kK, {.ef = kEf, .model = &model}, st.data(), nullptr, 1);
  const auto plain = f.index.search(q.data(), 5, kK, kEf, 1);
  EXPECT_EQ(r.ids, plain.ids);
  for (const auto& s : st) {
    EXPECT_FALSE(s.features_valid);
    EXPECT_FALSE(s.model_used);
  }
}

TEST(Adaptive, ModelIgnoredWithDeletedNodes) {
  auto base = random_vectors(1000, kDim, 5), queries = random_vectors(10, kDim, 6);
  HnswIndex index(kDim, Metric::L2, {.M = 8, .ef_construction = 50});
  index.add(base.data(), 1000, nullptr, 1);
  index.remove(3);
  const auto model = constant_model(15, 12);
  std::vector<AdaptiveStats> st(10);
  const auto r = index.search_adaptive(queries.data(), 10, kK, {.ef = kEf, .model = &model},
                                       st.data(), nullptr, 1);
  EXPECT_EQ(r.ids, index.search(queries.data(), 10, kK, kEf, 1).ids);
  for (const auto& s : st) EXPECT_FALSE(s.model_used);
}

TEST(Adaptive, ParallelEqualsSerial) {
  auto& f = fixture();
  const auto model = constant_model(150, 40);
  const AdaptiveParams p{.ef = 128, .patience = 30, .model = &model};
  std::vector<AdaptiveStats> s1(kQ), s8(kQ);
  const auto a = f.index.search_adaptive(f.queries.data(), kQ, kK, p, s1.data(), nullptr, 1);
  const auto b = f.index.search_adaptive(f.queries.data(), kQ, kK, p, s8.data(), nullptr, 8);
  EXPECT_EQ(a.ids, b.ids);
  for (std::size_t q = 0; q < kQ; ++q) EXPECT_EQ(s1[q].evals, s8[q].evals);
}

TEST(Adaptive, QueryDimMismatchThrows) {
  auto& f = fixture();
  std::ostringstream s;
  s << "vecsearch-termination-model 1 kind linear checkpoint 10 query_dim 3 bias 1 weights "
    << kNumTerminationFeatures + 3;
  for (std::size_t i = 0; i < kNumTerminationFeatures + 3; ++i) s << " 0";
  s << " end";
  const auto model = parse(s.str());
  EXPECT_THROW(f.index.search_adaptive(f.queries.data(), 1, kK, {.model = &model}),
               std::invalid_argument);
}

TEST(TerminationModel, TreePrediction) {
  // One stump on feature 1 (log d_1) plus a second tree that is a single leaf.
  const auto m = parse(
      "vecsearch-termination-model 1 kind gbdt checkpoint 7 query_dim 0 bias 1.5 trees 2 "
      "nodes 3  1 2.0 1 2 0  -1 0 0 0 0.25  -1 0 0 0 -0.5 "
      "nodes 1  -1 0 0 0 0.125 end");
  EXPECT_EQ(m.checkpoint(), 7u);
  float f[kNumTerminationFeatures] = {};
  f[1] = 2.0f;  // x <= threshold goes left
  EXPECT_DOUBLE_EQ(m.predict_log_evals(f, nullptr), 1.5 + 0.25 + 0.125);
  f[1] = 2.5f;
  EXPECT_DOUBLE_EQ(m.predict_log_evals(f, nullptr), 1.5 - 0.5 + 0.125);
}

TEST(TerminationModel, LinearUsesQueryVector) {
  std::ostringstream s;
  s << "vecsearch-termination-model 1 kind linear checkpoint 5 query_dim 2 bias 0.5 weights "
    << kNumTerminationFeatures + 2;
  for (std::size_t i = 0; i < kNumTerminationFeatures; ++i) s << (i == 0 ? " 2" : " 0");
  s << " 10 -1 end";
  const auto m = parse(s.str());
  float f[kNumTerminationFeatures] = {3.0f};
  const float q[2] = {1.0f, 4.0f};
  EXPECT_DOUBLE_EQ(m.predict_log_evals(f, q), 0.5 + 6 + 10 - 4);
}

TEST(TerminationModel, RejectsMalformedFiles) {
  const std::string head =
      "vecsearch-termination-model 1 kind gbdt checkpoint 7 query_dim 0 bias 0 ";
  for (
      const std::string& bad : std::vector<std::string>{
          "",
          "vecsearch-termination-model 2 kind linear",               // future version
          "vecsearch-termination-model 1 kind forest checkpoint 1",  // unknown kind
          "vecsearch-termination-model 1 kind linear checkpoint 0 query_dim 0 bias 0 weights 0 end",
          "vecsearch-termination-model 1 kind linear checkpoint 3 query_dim 0 bias 0 weights 2 1 1 "
          "end",
          head + "trees 1 nodes 3  0 1 0 2 0  -1 0 0 0 1  -1 0 0 0 1 end",  // child loops to itself
          head + "trees 1 nodes 3  0 1 1 5 0  -1 0 0 0 1  -1 0 0 0 1 end",  // child out of range
          head + "trees 1 nodes 3  99 1 1 2 0  -1 0 0 0 1  -1 0 0 0 1 end",   // bad feature
          head + "trees 1 nodes 3  0 nan 1 2 0  -1 0 0 0 1  -1 0 0 0 1 end",  // NaN threshold
          head + "trees 1 nodes 1  -1 0 0 0 inf end",                         // non-finite leaf
          head + "trees 1 nodes 3  0 1 1 2 0  -1 0 0 0 1",                    // truncated
      }) {
    EXPECT_THROW(parse(bad), std::runtime_error) << bad;
  }
}
