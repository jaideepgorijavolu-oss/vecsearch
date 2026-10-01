// The scalar loops again, compiled with -O3 -march=native (and, in a second object, also with
// -ffast-math). This answers "how close does the compiler get on its own?" for the
// microbenchmarks. VECSEARCH_AUTOVEC_NS names the namespace for each build.
#include <cstddef>

namespace VECSEARCH_AUTOVEC_NS {

float l2_sq(const float* a, const float* b, std::size_t dim) {
  float sum = 0.0f;
  for (std::size_t i = 0; i < dim; ++i) {
    const float d = a[i] - b[i];
    sum += d * d;
  }
  return sum;
}

float dot(const float* a, const float* b, std::size_t dim) {
  float sum = 0.0f;
  for (std::size_t i = 0; i < dim; ++i) sum += a[i] * b[i];
  return sum;
}

}  // namespace VECSEARCH_AUTOVEC_NS
