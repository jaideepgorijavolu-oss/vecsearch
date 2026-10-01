// AVX2 + FMA kernels. This file alone is compiled with -mavx2 -mfma (see CMakeLists.txt); the
// dispatcher only calls into it after checking that the CPU supports both.
#include <immintrin.h>

#include "vecsearch/distance.hpp"

namespace vecsearch::avx2 {

namespace {

// Sum of the 8 lanes of a 256-bit register.
inline float hsum(__m256 v) {
  __m128 lo = _mm256_castps256_ps128(v);
  const __m128 hi = _mm256_extractf128_ps(v, 1);
  lo = _mm_add_ps(lo, hi);             // 4 lanes
  __m128 shuf = _mm_movehdup_ps(lo);   // (1,1,3,3)
  __m128 sums = _mm_add_ps(lo, shuf);  // (0+1, _, 2+3, _)
  shuf = _mm_movehl_ps(shuf, sums);    // (2+3, ...)
  sums = _mm_add_ss(sums, shuf);
  return _mm_cvtss_f32(sums);
}

}  // namespace

// Four independent accumulators (32 floats per iteration). An FMA has ~4 cycles latency but the
// CPU can start two per cycle, so with a single accumulator every iteration would wait for the
// previous one. Four chains keep the FMA units busy. Then one 8-wide step, then a scalar tail
// for dims that are not a multiple of 8.
float l2_sq(const float* a, const float* b, std::size_t dim) {
  __m256 acc0 = _mm256_setzero_ps(), acc1 = _mm256_setzero_ps();
  __m256 acc2 = _mm256_setzero_ps(), acc3 = _mm256_setzero_ps();
  std::size_t i = 0;
  for (; i + 32 <= dim; i += 32) {
    const __m256 d0 = _mm256_sub_ps(_mm256_loadu_ps(a + i), _mm256_loadu_ps(b + i));
    const __m256 d1 = _mm256_sub_ps(_mm256_loadu_ps(a + i + 8), _mm256_loadu_ps(b + i + 8));
    const __m256 d2 = _mm256_sub_ps(_mm256_loadu_ps(a + i + 16), _mm256_loadu_ps(b + i + 16));
    const __m256 d3 = _mm256_sub_ps(_mm256_loadu_ps(a + i + 24), _mm256_loadu_ps(b + i + 24));
    acc0 = _mm256_fmadd_ps(d0, d0, acc0);
    acc1 = _mm256_fmadd_ps(d1, d1, acc1);
    acc2 = _mm256_fmadd_ps(d2, d2, acc2);
    acc3 = _mm256_fmadd_ps(d3, d3, acc3);
  }
  for (; i + 8 <= dim; i += 8) {
    const __m256 d = _mm256_sub_ps(_mm256_loadu_ps(a + i), _mm256_loadu_ps(b + i));
    acc0 = _mm256_fmadd_ps(d, d, acc0);
  }
  float sum = hsum(_mm256_add_ps(_mm256_add_ps(acc0, acc1), _mm256_add_ps(acc2, acc3)));
  for (; i < dim; ++i) {
    const float d = a[i] - b[i];
    sum += d * d;
  }
  return sum;
}

float dot(const float* a, const float* b, std::size_t dim) {
  __m256 acc0 = _mm256_setzero_ps(), acc1 = _mm256_setzero_ps();
  __m256 acc2 = _mm256_setzero_ps(), acc3 = _mm256_setzero_ps();
  std::size_t i = 0;
  for (; i + 32 <= dim; i += 32) {
    acc0 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i), _mm256_loadu_ps(b + i), acc0);
    acc1 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i + 8), _mm256_loadu_ps(b + i + 8), acc1);
    acc2 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i + 16), _mm256_loadu_ps(b + i + 16), acc2);
    acc3 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i + 24), _mm256_loadu_ps(b + i + 24), acc3);
  }
  for (; i + 8 <= dim; i += 8) {
    acc0 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i), _mm256_loadu_ps(b + i), acc0);
  }
  float sum = hsum(_mm256_add_ps(_mm256_add_ps(acc0, acc1), _mm256_add_ps(acc2, acc3)));
  for (; i < dim; ++i) sum += a[i] * b[i];
  return sum;
}

}  // namespace vecsearch::avx2
