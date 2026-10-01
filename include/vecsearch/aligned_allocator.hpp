#pragma once

#include <cstddef>
#include <limits>
#include <new>
#include <vector>

namespace vecsearch {

inline constexpr std::size_t kCacheLine = 64;

// Allocator that returns memory aligned to `Alignment` bytes (default: one cache line).
//
// Why: a 64-byte-aligned vector never straddles more cache lines than necessary, and its start
// is aligned for 256-bit (AVX2) and 512-bit loads. Combined with padding each row to a multiple
// of 16 floats (see padded_dim), every stored vector starts on its own cache-line boundary.
template <class T, std::size_t Alignment = kCacheLine>
struct AlignedAllocator {
  static_assert(Alignment >= alignof(T), "alignment must satisfy T");
  static_assert((Alignment & (Alignment - 1)) == 0, "alignment must be a power of two");

  using value_type = T;
  template <class U>
  struct rebind {
    using other = AlignedAllocator<U, Alignment>;
  };

  AlignedAllocator() noexcept = default;
  template <class U>
  AlignedAllocator(const AlignedAllocator<U, Alignment>&) noexcept {}

  T* allocate(std::size_t n) {
    if (n > std::numeric_limits<std::size_t>::max() / sizeof(T)) throw std::bad_array_new_length();
    return static_cast<T*>(::operator new(n * sizeof(T), std::align_val_t{Alignment}));
  }
  void deallocate(T* p, std::size_t) noexcept { ::operator delete(p, std::align_val_t{Alignment}); }

  friend bool operator==(const AlignedAllocator&, const AlignedAllocator&) noexcept { return true; }
};

template <class T>
using AlignedVector = std::vector<T, AlignedAllocator<T>>;

// Row stride (in floats) used to store a `dim`-dimensional vector: rounded up to 16 floats
// (64 bytes) so consecutive rows stay cache-line aligned. Padding floats are always zero.
inline constexpr std::size_t padded_dim(std::size_t dim) { return (dim + 15) / 16 * 16; }

}  // namespace vecsearch
