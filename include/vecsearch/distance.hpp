#pragma once

#include <cstddef>
#include <cstdint>
#include <string_view>
#include <vector>

namespace vecsearch {

enum class Metric : std::uint8_t {
  L2 = 0,            // squared Euclidean distance
  InnerProduct = 1,  // distance = 1 - <a, b>
  Cosine = 2,        // vectors normalized at insert, then InnerProduct
};

Metric parse_metric(std::string_view name);  // "l2", "ip", "cosine"; throws on anything else
const char* metric_name(Metric m);

// A raw kernel: returns sum((a-b)^2) or sum(a*b) over `dim` floats.
using KernelFn = float (*)(const float* a, const float* b, std::size_t dim);

namespace scalar {
float l2_sq(const float* a, const float* b, std::size_t dim);
float dot(const float* a, const float* b, std::size_t dim);
}  // namespace scalar

// A set of kernels for one instruction set.
struct Kernels {
  KernelFn l2_sq;
  KernelFn dot;
  const char* name;
};

const Kernels& scalar_kernels();

// The kernels used by the indexes. Chosen once, on first call.
const Kernels& active_kernels();

// Every kernel set this binary contains that the current CPU can run (scalar first).
std::vector<const Kernels*> supported_kernels();

// A distance where smaller means closer, for any metric.
struct Distance {
  KernelFn fn;
  bool is_ip;
  float operator()(const float* a, const float* b, std::size_t dim) const {
    return is_ip ? 1.0f - fn(a, b, dim) : fn(a, b, dim);
  }
};

Distance make_distance(Metric m, const Kernels& k = active_kernels());

// Scales v to unit L2 norm in place. Zero vectors are left unchanged.
void normalize(float* v, std::size_t dim);

}  // namespace vecsearch
