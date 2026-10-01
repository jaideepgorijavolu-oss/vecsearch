#include "vecsearch/distance.hpp"

namespace vecsearch {

const Kernels& active_kernels() { return scalar_kernels(); }

std::vector<const Kernels*> supported_kernels() { return {&scalar_kernels()}; }

}  // namespace vecsearch
