// Runtime kernel selection.
//
// The library is compiled for baseline x86-64 (or AArch64). SIMD kernels live in their own
// translation units compiled with extra -m flags. On the first call to active_kernels() we ask
// the CPU what it supports and pick the best kernel set; after that every distance call is one
// indirect call through a function pointer, with no per-call feature checks.
#include <cstdlib>
#include <string_view>

#include "vecsearch/distance.hpp"

namespace vecsearch {

#if VECSEARCH_HAVE_AVX2
namespace avx2 {
float l2_sq(const float* a, const float* b, std::size_t dim);
float dot(const float* a, const float* b, std::size_t dim);
}  // namespace avx2

static const Kernels kAvx2{&avx2::l2_sq, &avx2::dot, "avx2"};

static bool cpu_has_avx2_fma() {
#if defined(__GNUC__) || defined(__clang__)
  // libgcc / compiler-rt also check (via XGETBV) that the OS saves the YMM registers.
  __builtin_cpu_init();
  return __builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma");
#else
  return false;
#endif
}
#endif

#if VECSEARCH_HAVE_NEON
namespace neon {
float l2_sq(const float* a, const float* b, std::size_t dim);
float dot(const float* a, const float* b, std::size_t dim);
}  // namespace neon

static const Kernels kNeon{&neon::l2_sq, &neon::dot, "neon"};
#endif

std::vector<const Kernels*> supported_kernels() {
  std::vector<const Kernels*> out{&scalar_kernels()};
#if VECSEARCH_HAVE_AVX2
  if (cpu_has_avx2_fma()) out.push_back(&kAvx2);
#endif
#if VECSEARCH_HAVE_NEON
  out.push_back(&kNeon);
#endif
  return out;
}

static const Kernels& choose() {
  const auto all = supported_kernels();
  // VECSEARCH_KERNELS=scalar (or avx2/neon) forces a kernel set, for testing and benchmarks.
  if (const char* forced = std::getenv("VECSEARCH_KERNELS")) {
    for (const Kernels* k : all) {
      if (std::string_view(forced) == k->name) return *k;
    }
  }
  return *all.back();  // the list is ordered from slowest to fastest
}

const Kernels& active_kernels() {
  static const Kernels& chosen = choose();  // thread-safe one-time initialization (C++11)
  return chosen;
}

}  // namespace vecsearch
