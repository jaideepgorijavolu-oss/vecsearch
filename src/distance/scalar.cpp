#include <cmath>
#include <stdexcept>
#include <string>

#include "vecsearch/distance.hpp"

namespace vecsearch {

namespace scalar {

// Reference kernels. One accumulator, strict left-to-right order. Without -ffast-math the
// compiler may not reorder these float additions, so it cannot vectorize the loop: this is the
// true scalar baseline that the SIMD kernels are measured against.
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

}  // namespace scalar

const Kernels& scalar_kernels() {
  static const Kernels k{&scalar::l2_sq, &scalar::dot, "scalar"};
  return k;
}

Metric parse_metric(std::string_view name) {
  if (name == "l2") return Metric::L2;
  if (name == "ip") return Metric::InnerProduct;
  if (name == "cosine") return Metric::Cosine;
  throw std::invalid_argument("unknown metric '" + std::string(name) +
                              "' (expected l2, ip or cosine)");
}

const char* metric_name(Metric m) {
  switch (m) {
    case Metric::L2:
      return "l2";
    case Metric::InnerProduct:
      return "ip";
    case Metric::Cosine:
      return "cosine";
  }
  return "unknown";
}

Distance make_distance(Metric m, const Kernels& k) {
  if (m == Metric::L2) return Distance{k.l2_sq, false};
  return Distance{k.dot, true};
}

void normalize(float* v, std::size_t dim) {
  double norm_sq = 0.0;  // double: avoids precision loss for large dims
  for (std::size_t i = 0; i < dim; ++i) norm_sq += static_cast<double>(v[i]) * v[i];
  if (norm_sq == 0.0) return;
  const float inv = static_cast<float>(1.0 / std::sqrt(norm_sq));
  for (std::size_t i = 0; i < dim; ++i) v[i] *= inv;
}

}  // namespace vecsearch
