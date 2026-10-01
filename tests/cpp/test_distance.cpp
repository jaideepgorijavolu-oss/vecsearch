#include <gtest/gtest.h>

#include <cmath>
#include <string>

#include "test_util.hpp"
#include "vecsearch/distance.hpp"

using namespace vecsearch;

namespace {

double ref_l2(const float* a, const float* b, std::size_t d) {
  double s = 0;
  for (std::size_t i = 0; i < d; ++i) s += (double(a[i]) - b[i]) * (double(a[i]) - b[i]);
  return s;
}

double ref_dot(const float* a, const float* b, std::size_t d) {
  double s = 0;
  for (std::size_t i = 0; i < d; ++i) s += double(a[i]) * b[i];
  return s;
}

}  // namespace

TEST(Distance, ScalarMatchesDoubleReference) {
  const auto a = test::random_vectors(1, 100, 1), b = test::random_vectors(1, 100, 2);
  EXPECT_NEAR(scalar::l2_sq(a.data(), b.data(), 100), ref_l2(a.data(), b.data(), 100), 1e-3);
  EXPECT_NEAR(scalar::dot(a.data(), b.data(), 100), ref_dot(a.data(), b.data(), 100), 1e-3);
}

TEST(Distance, ActiveKernelIsListed) {
  const auto all = supported_kernels();
  ASSERT_FALSE(all.empty());
  EXPECT_STREQ(all.front()->name, "scalar");
  bool found = false;
  for (const auto* k : all) found |= (k == &active_kernels());
  EXPECT_TRUE(found);
  std::printf("active kernels: %s\n", active_kernels().name);
}

// Every SIMD kernel must agree with the scalar kernel to 1e-4 relative error, for every dim
// from 1 to 200 (covers all tail lengths) and the benchmark dims. Inputs are offset by one float
// to exercise unaligned loads.
TEST(Distance, SimdMatchesScalar) {
  std::vector<std::size_t> dims;
  for (std::size_t d = 1; d <= 200; ++d) dims.push_back(d);
  for (std::size_t d : {384u, 768u, 1536u, 1537u}) dims.push_back(d);

  for (const Kernels* k : supported_kernels()) {
    for (std::size_t d : dims) {
      for (std::uint32_t trial = 0; trial < 4; ++trial) {
        const auto a = test::random_vectors(1, d + 1, 100 + trial);
        const auto b = test::random_vectors(1, d + 1, 200 + trial);
        const float* pa = a.data() + 1;
        const float* pb = b.data() + 1;

        const double l2_ref = scalar::l2_sq(pa, pb, d);
        const double l2 = k->l2_sq(pa, pb, d);
        EXPECT_LE(std::abs(l2 - l2_ref), 1e-4 * std::max(1.0, std::abs(l2_ref)))
            << k->name << " l2 dim=" << d;

        // Dot products can be near zero, so measure error relative to the sum of |a_i b_i|
        // (the scale of the rounding error), not relative to the result itself.
        double scale = 0;
        for (std::size_t i = 0; i < d; ++i) scale += std::abs(double(pa[i]) * pb[i]);
        const double ip_ref = scalar::dot(pa, pb, d);
        const double ip = k->dot(pa, pb, d);
        EXPECT_LE(std::abs(ip - ip_ref), 1e-4 * std::max(1.0, scale))
            << k->name << " dot dim=" << d;
      }
    }
  }
}

TEST(Distance, InnerProductDistanceConvention) {
  const float a[3] = {1, 0, 0}, b[3] = {1, 0, 0}, c[3] = {0, 1, 0};
  const Distance d = make_distance(Metric::InnerProduct);
  EXPECT_FLOAT_EQ(d(a, b, 3), 0.0f);  // identical unit vectors: distance 0
  EXPECT_FLOAT_EQ(d(a, c, 3), 1.0f);  // orthogonal: distance 1
}

TEST(Distance, NormalizeAndMetricNames) {
  float v[2] = {3, 4};
  normalize(v, 2);
  EXPECT_FLOAT_EQ(v[0], 0.6f);
  EXPECT_FLOAT_EQ(v[1], 0.8f);
  float z[2] = {0, 0};
  normalize(z, 2);
  EXPECT_EQ(z[0], 0.0f);
  for (Metric m : {Metric::L2, Metric::InnerProduct, Metric::Cosine}) {
    EXPECT_EQ(parse_metric(metric_name(m)), m);
  }
  EXPECT_THROW(parse_metric("hamming"), std::invalid_argument);
}
